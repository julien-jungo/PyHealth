"""Cox proportional hazards model over dense per-patient features.

Reproduces the survival models of Song et al., "Multimodal Cancer Modeling in
the Age of Foundation Model Embeddings", ML4H 2025
(https://arxiv.org/abs/2505.07683): a linear Cox model on precomputed,
standardized and PCA-reduced foundation-model embeddings.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.trajsurv import cox_partial_likelihood_loss
from pyhealth.processors.base_processor import FeatureProcessor

__all__ = ["CoxSurvivalModel"]


def _query(processor: FeatureProcessor, method: str, name: str) -> Any:
    """Calls a processor's shape method, turning its absence into ValueError."""
    if not hasattr(processor, method):
        raise ValueError(f"{name} has no {method}(), so it is not a dense vector")
    try:
        return getattr(processor, method)()
    except NotImplementedError as err:
        raise ValueError(
            f"{name}.{method}() is not implemented or not fitted: {err}"
        ) from err


class CoxSurvivalModel(BaseModel):
    """Linear Cox model over concatenated dense features.

    Each sample's feature fields are flattened and concatenated, and a linear
    head maps them to one log hazard ratio (``risk``; higher means earlier
    events). The head has no bias, since the Cox partial likelihood is
    invariant to shifting all risks. Training minimizes the negative Cox
    partial likelihood (Breslow ties).

    The risk sets are those of each batch, so mini-batch training optimizes a
    different objective than the full partial likelihood. Trained full-batch
    (``batch_size`` = training set size) with the paper's ridge penalty
    rescaled to ``weight_decay``, this is the ridge CoxPH of the paper.

    Labels follow :class:`~pyhealth.models.TrajSurv`: ``duration`` and
    ``event`` tensor outputs. ``mode`` is ``None``, so evaluate with
    :func:`pyhealth.metrics.concordance_index` on ``Trainer.inference`` output.

    Args:
        dataset: Fitted ``SampleDataset`` with dense feature fields and
            ``duration``/``event`` tensor outputs.
        feature_keys: Input fields to use. Defaults to all input fields; pass
            one field for a unimodal model.
        duration_key: Output field with the observed time.
        event_key: Output field with 1 for an observed event, 0 for censoring.

    Raises:
        ValueError: If ``feature_keys`` is empty or not a subset of the input
            fields, if ``duration_key`` or ``event_key`` is not an output
            field, or if a feature field is not one dense vector of fixed
            width per sample (e.g. token indices or a matrix).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "expr": [0.1, 0.2],
        ...      "duration": [100.0], "event": [1.0]},
        ...     {"patient_id": "p1", "expr": [0.3, -0.1],
        ...      "duration": [250.0], "event": [0.0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"expr": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = CoxSurvivalModel(dataset)
        >>> model.mode is None
        True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: list[str] | None = None,
        duration_key: str = "duration",
        event_key: str = "event",
    ) -> None:
        super().__init__(dataset)
        self.mode = None

        feature_keys = (
            list(dataset.input_schema) if feature_keys is None else feature_keys
        )
        if not feature_keys:
            raise ValueError("feature_keys must not be empty")
        for key in feature_keys:
            if key not in dataset.input_schema:
                raise ValueError(f"{key!r} is not in dataset.input_schema")
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in dataset.output_schema")

        self.feature_keys = list(feature_keys)
        self.duration_key = duration_key
        self.event_key = event_key

        width = sum(self._feature_width(key) for key in self.feature_keys)
        self.head = nn.Linear(width, 1, bias=False)

    def _feature_width(self, key: str) -> int:
        processor = self.dataset.input_processors[key]
        name = f"feature {key!r} ({type(processor).__name__})"

        size = _query(processor, "size", name)
        if not size:
            raise ValueError(f"{name} has no fixed width")

        if _query(processor, "is_token", name):
            raise ValueError(f"{name} holds token indices, not dense values")

        schema = _query(processor, "schema", name)
        if "value" not in schema:
            raise ValueError(f"{name} has no 'value' output")

        value_dim = _query(processor, "dim", name)[schema.index("value")]
        if value_dim != 1:
            raise ValueError(f"{name} has {value_dim}-D values per sample")

        return size

    def _feature_value(self, key: str, feature: Tensor | tuple[Tensor, ...]) -> Tensor:
        if isinstance(feature, Tensor):
            feature = (feature,)
        schema = self.dataset.input_processors[key].schema()
        value = feature[schema.index("value")]
        return value.to(self.device, torch.float32).flatten(start_dim=1)

    def _label(self, key: str, kwargs: dict) -> Tensor:
        return torch.as_tensor(kwargs[key]).to(self.device, torch.float32).reshape(-1)

    def forward(self, **kwargs: Any) -> dict[str, Tensor]:
        """Computes risk scores and, with labels, the Cox loss.

        Args:
            **kwargs: A collated batch: one tensor (or processor tuple) per
                feature field, plus optionally the duration and event fields.

        Returns:
            ``risk`` (shape ``[B]``), ``logit`` (``[B, 1]``) and ``y_prob``
            (the risk, ``[B, 1]``); with labels also ``loss`` and ``y_true``
            (``[B, 2]``: duration, event).
        """
        features = torch.cat(
            [self._feature_value(key, kwargs[key]) for key in self.feature_keys],
            dim=1,
        )

        logit = self.head(features)
        risk = logit.squeeze(-1)
        out = {"risk": risk, "logit": logit, "y_prob": logit}
        if self.duration_key not in kwargs or self.event_key not in kwargs:
            return out

        duration = self._label(self.duration_key, kwargs)
        event = self._label(self.event_key, kwargs)
        out["loss"] = cox_partial_likelihood_loss(risk, duration, event)
        out["y_true"] = torch.stack([duration, event], dim=-1)
        return out
