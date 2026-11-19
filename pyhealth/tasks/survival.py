"""Multimodal time-to-event survival task for TCGA.

Contributors: Julien David Jungo (jjungo2), Donovan Scott Below (dbelow2),
    Yaseen Mohamed (yaseenm2)
Paper: Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
    Embeddings", ML4H 2025
Paper link: https://proceedings.mlr.press/v297/song26a.html
Description: Overall survival (time to death, right-censored at last
    follow-up) from precomputed foundation-model embeddings and clinical
    features, one sample per patient.
"""

import logging
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

logger = logging.getLogger(__name__)

AGE_BINS = (0, 20, 40, 60, 80, 100)
EMBEDDING_MODALITIES = ("expr", "hist", "text")
CLINICAL_MODALITIES = ("demo", "canc")
ALL_MODALITIES = EMBEDDING_MODALITIES + CLINICAL_MODALITIES
AGGREGATIONS = ("mean", "first")


class MultimodalSurvivalPrediction(BaseTask):
    """Predicts overall survival from embeddings and clinical features.

    Runs on :class:`~pyhealth.datasets.TCGAMultimodalDataset`. Emits one
    sample per patient with the requested modalities:

    - ``expr``, ``hist``, ``text``: one embedding per patient (RNA-seq,
      histology, pathology report), read from the dataset's events of the
      same name. A patient with several files for a
      modality (e.g. two slides) gets them combined by ``aggregation``. The
      vectors are standardized and PCA-reduced by
      :class:`~pyhealth.processors.EmbeddingPCAProcessor`, fitted on the
      training part when ``set_task`` is given a ``split``.
    - ``demo``: tokens for sex, age bin (as in the paper: ``(0, 20]`` to
      ``(80, 100]``), race and ethnicity, one-hot encoded by
      :class:`~pyhealth.processors.MultiHotIgnoreUnknownProcessor`, which
      gives tokens unseen in the training part no bit.
    - ``canc``: the TCGA project as a single token, encoded the same way.
    - ``duration``: days to death, or to last follow-up when censored.
      Zero-day durations are kept, as in the paper.
    - ``event``: 1.0 if the patient died, else 0.0.

    A patient is kept only if they have one clinical record, a valid outcome,
    and every requested modality. Skipped patients are logged at INFO level
    with the reason, never dropped silently. Requesting fewer modalities keeps
    patients who lack the others; the paper uses patients with all of them.

    Args:
        expr_components: PCA components for ``expr``, or ``None`` to only
            standardize. Defaults to 256, the size the paper reports results
            with (Sec. 3.2, Table 1).
        hist_components: PCA components for ``hist``, or ``None``.
        text_components: PCA components for ``text``, or ``None``.
        modalities: Fields to emit, a non-empty subset of
            ``("expr", "hist", "text", "demo", "canc")``. Defaults to all.
        aggregation: How to combine several files of one patient and
            modality: ``"mean"`` (the paper, Sec. 2.3) or ``"first"`` (the
            file with the smallest file ID).

    Raises:
        ValueError: If ``modalities`` is empty or has unknown names,
            ``aggregation`` is unknown, or a component count is not a
            positive integer or ``None``.

    Examples:
        >>> from pyhealth.datasets import TCGAMultimodalDataset
        >>> from pyhealth.tasks import MultimodalSurvivalPrediction
        >>> dataset = TCGAMultimodalDataset(root="/path/to/data", tables=["text"])
        >>> task = MultimodalSurvivalPrediction(modalities=["text", "demo", "canc"])
        >>> task.input_schema["text"]
        ('embedding_pca', {'n_components': 256})
    """

    task_name: str = "MultimodalSurvivalPrediction"
    output_schema: ClassVar[dict[str, str]] = {"duration": "tensor", "event": "tensor"}

    def __init__(
        self,
        expr_components: int | None = 256,
        hist_components: int | None = 256,
        text_components: int | None = 256,
        modalities: Sequence[str] = ALL_MODALITIES,
        aggregation: str = "mean",
    ) -> None:
        unknown = sorted(set(modalities) - set(ALL_MODALITIES))
        if unknown or not modalities:
            raise ValueError(
                f"modalities must be a non-empty subset of {list(ALL_MODALITIES)}; "
                f"got unknown {unknown}"
            )
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {list(AGGREGATIONS)}")
        components = {
            "expr": expr_components,
            "hist": hist_components,
            "text": text_components,
        }
        for name, n in components.items():
            if n is not None and (
                isinstance(n, bool) or not isinstance(n, int) or n < 1
            ):
                raise ValueError(f"{name}_components must be a positive int or None")
        # Plain str/int/list attributes: set_task serializes vars(task) to JSON
        # for its cache key.
        self.modalities: list[str] = [m for m in ALL_MODALITIES if m in modalities]
        self.aggregation = aggregation
        self.input_schema: dict[str, Any] = {
            m: (
                ("embedding_pca", {"n_components": components[m]})
                if m in EMBEDDING_MODALITIES
                else "multi_hot_ignore_unknown"
            )
            for m in self.modalities
        }
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the patient's sample.

        Args:
            patient: A patient of :class:`~pyhealth.datasets.TCGAMultimodalDataset`.

        Returns:
            One sample, or an empty list when the clinical record, the
            outcome, or a requested modality is missing or invalid.
        """
        clinical = patient.get_events(event_type="clinical")
        if len(clinical) != 1:
            return _skip(patient, f"{len(clinical)} clinical records")
        record = clinical[0]
        label = _survival_label(record)
        if label is None:
            return _skip(patient, "missing or invalid survival outcome")

        sample: dict[str, Any] = {"patient_id": patient.patient_id, **label}
        for modality in self.modalities:
            if modality == "demo":
                sample[modality] = _demographic_tokens(record)
            elif modality == "canc":
                sample[modality] = [_category(getattr(record, "project", None))]
            else:
                vector = self._embedding(patient, modality)
                if vector is None:
                    return _skip(patient, f"no {modality} embedding")
                sample[modality] = vector
        return [sample]

    def _embedding(self, patient: Any, event_type: str) -> list[float] | None:
        """Loads and combines a patient's embedding files of one event type.

        Args:
            patient: The patient.
            event_type: ``"expr"``, ``"hist"`` or ``"text"``.

        Returns:
            One embedding as a list of floats, or None if the patient has no
            file of this type.

        Raises:
            FileNotFoundError: If an indexed ``.npy`` file no longer exists.
            ValueError: If the patient's files are not 1-D or differ in length.
        """
        events = sorted(
            patient.get_events(event_type=event_type), key=lambda e: str(e.file_id)
        )
        if not events:
            return None
        if self.aggregation == "first":
            events = events[:1]
        vectors = [np.load(event.embedding_path) for event in events]
        shapes = {vector.shape for vector in vectors}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(
                f"{event_type} embeddings of patient {patient.patient_id} must be "
                f"1-D with one shape; got {sorted(shapes)}"
            )
        return np.mean(np.stack(vectors), axis=0).astype(np.float32).tolist()


def _survival_label(record: Any) -> dict[str, list[float]] | None:
    """Derives ``duration`` (days) and ``event`` from a clinical record.

    Args:
        record: The patient's clinical event.

    Returns:
        ``{"duration": [days], "event": [1.0 or 0.0]}``, or None if the vital
        status is not "Dead"/"Alive" or the matching day count is missing or
        negative.
    """
    status = str(getattr(record, "vital_status", "")).strip().lower()
    if status == "dead":
        days, event = _to_float(getattr(record, "days_to_death", None)), 1.0
    elif status == "alive":
        days = _to_float(getattr(record, "days_to_last_follow_up", None))
        event = 0.0
    else:
        return None
    if days is None or days < 0:
        return None
    return {"duration": [days], "event": [event]}


def _demographic_tokens(record: Any) -> list[str]:
    """Encodes sex, binned age, race and ethnicity as category tokens.

    Args:
        record: The patient's clinical event.

    Returns:
        Four ``"field:value"`` tokens; missing values become ``"unknown"``.
    """
    return [
        f"sex:{_category(getattr(record, 'sex', None))}",
        f"age:{_age_bin(_to_float(getattr(record, 'age', None)))}",
        f"race:{_category(getattr(record, 'race', None))}",
        f"ethnicity:{_category(getattr(record, 'ethnicity', None))}",
    ]


def _age_bin(age: float | None) -> str:
    """Bins age into ``AGE_BINS`` intervals, right-inclusive like the paper.

    Args:
        age: Age in years, or None.

    Returns:
        A label such as ``"(40, 60]"``, or ``"unknown"`` outside the bins.
    """
    if age is not None:
        for lower, upper in pairwise(AGE_BINS):
            if lower < age <= upper:
                return f"({lower}, {upper}]"
    return "unknown"


def _to_float(value: Any) -> float | None:
    """Converts a raw attribute to float; None if missing or not numeric.

    Args:
        value: Raw value; dataset attributes may arrive as strings.

    Returns:
        The float, or None for None, empty, NaN or non-numeric input.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _category(value: Any) -> str:
    """Normalizes a raw categorical attribute.

    Args:
        value: Raw value.

    Returns:
        The stripped string, or ``"unknown"`` if missing.
    """
    text = "" if value is None else str(value).strip()
    return "unknown" if text in ("", "nan", "None", "<NA>") else text


def _skip(patient: Any, reason: str) -> list[dict[str, Any]]:
    """Logs why a patient is skipped and returns no samples.

    Args:
        patient: The skipped patient.
        reason: Human-readable reason.

    Returns:
        An empty list.
    """
    logger.info(f"Skipping patient {patient.patient_id}: {reason}")
    return []
