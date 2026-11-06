"""Cox proportional hazards model over dense per-patient features.

Reproduces the survival models of Song et al., "Multimodal Cancer Modeling in
the Age of Foundation Model Embeddings", ML4H 2025
(https://arxiv.org/abs/2505.07683): a linear Cox model on precomputed,
standardized and PCA-reduced foundation-model embeddings, with an optional MLP
head as an extension.
"""

from typing import Literal

from torch import Tensor

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.trajsurv import cox_partial_likelihood_loss

__all__ = ["CoxSurvivalModel"]

Head = Literal["linear", "mlp"]


class CoxSurvivalModel(BaseModel):
    """Cox model with a linear or MLP risk head over concatenated features.

    Each sample's feature fields are flattened and concatenated, and a head
    maps them to one log hazard ratio (``risk``; higher means earlier events).
    Training minimizes the negative Cox partial likelihood (Breslow ties). With
    ``head="linear"`` and ``weight_decay`` in the optimizer this is the ridge
    CoxPH of the paper; ``head="mlp"`` is a DeepSurv-style extension.

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
        head: ``"linear"`` (paper) or ``"mlp"`` (extension).
        hidden_dim: Hidden width of the MLP head. Ignored for ``"linear"``.
        dropout: Dropout before each MLP layer. Ignored for ``"linear"``.

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
        head: Head = "linear",
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(dataset)
        # TODO: validate feature_keys ⊆ input fields and duration/event keys ∈
        # output fields (fail fast with a clear ValueError), infer the input
        # width from the fitted processors, and build the linear or MLP head.
        raise NotImplementedError

    def forward(self, **kwargs: Tensor) -> dict[str, Tensor]:
        """Computes risk scores and, with labels, the Cox loss.

        Args:
            **kwargs: A collated batch: one tensor (or processor tuple) per
                feature field, plus the duration and event fields.

        Returns:
            ``risk`` (shape ``[B]``), ``logit`` (``[B, 1]``) and ``y_prob``
            (the risk, ``[B, 1]``); with labels also ``loss`` and ``y_true``
            (``[B, 2]``: duration, event).
        """
        # TODO: unpack each feature like MLP.forward does (processor schema
        # "value"), concatenate, apply the head, and compute
        # cox_partial_likelihood_loss(risk, duration, event).
        # TODO (optional): Breslow survival curves in y_prob, like
        # TrajSurv.fit_baseline_hazard.
        raise NotImplementedError
