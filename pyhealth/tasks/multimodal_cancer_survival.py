"""Multimodal cancer survival task for PyHealth.

Contributors: <Name 1> (<netid1>), <Name 2> (<netid2>), <Name 3> (<netid3>)
Paper: Song, Borjigin-Wang, Madejski, and Grossman. "Multimodal Cancer Modeling
    in the Age of Foundation Model Embeddings." ML4H 2025, PMLR 297.
Paper link: https://proceedings.mlr.press/v297/song26a.html
Description: Builds one right-censored survival sample per patient from any
    subset of five modalities: demographics, cancer type, and foundation-model
    embeddings of gene expression, histology slides, and pathology reports.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base_task import BaseTask

logger = logging.getLogger(__name__)

ALL_MODALITIES: tuple = (
    "demographics",
    "cancer_type",
    "expression",
    "histology",
    "report",
)
EMBEDDING_MODALITIES: tuple = ("expression", "histology", "report")
_SCHEMA: Dict[str, str] = {
    "demographics": "multi_hot",
    "cancer_type": "multi_hot",
    "expression": "tensor",
    "histology": "tensor",
    "report": "tensor",
}
_AGGREGATIONS: tuple = ("mean", "first")
_DURATION_UNITS: Dict[str, float] = {"days": 1.0, "years": 365.25}


class MultimodalCancerSurvival(BaseTask):
    """Predicts time to death from multimodal TCGA data.

    Each sample holds the requested modalities plus a right-censored label:
    ``duration`` is the time to death for patients who died, or to last
    follow-up for patients still alive ("censored"); ``event`` is 1 for an
    observed death and 0 for censoring. This matches how the paper builds its
    labels from ``days_to_death`` and ``days_to_last_follow_up``.

    A patient is kept only if they have one clinical record, a valid label,
    and every requested modality. Excluded patients are logged with the
    reason, never dropped silently. Requesting fewer modalities therefore
    keeps more patients; the paper uses the complete-case cohort.

    Input fields:
        - ``demographics`` (multi_hot): sex, age bin, race, and ethnicity
          tokens. Ages use the paper's bins (0, 20], (20, 40], ..., (80, 100].
        - ``cancer_type`` (multi_hot): TCGA project, e.g. ``"TCGA-BRCA"``.
        - ``expression``, ``histology``, ``report`` (tensor): one embedding
          per patient, combining the patient's files with ``aggregation``.

    Output fields:
        - ``duration`` (tensor of shape (1,)): survival time.
        - ``event`` (tensor of shape (1,)): 1.0 for death, 0.0 for censored.

    Args:
        modalities: Input fields to produce, any subset of ``ALL_MODALITIES``.
        aggregation: How to combine several embedding files of one patient.
            ``"mean"`` averages them (the paper's choice); ``"first"`` keeps
            the file with the smallest ``file_id``.
        duration_unit: ``"days"`` (as in the source data) or ``"years"``.
        min_duration: Durations below this value, in ``duration_unit``, are
            raised to it. The source data contains durations of 0, which some
            models (e.g. ``SurvivalMDN``) reject. The default 0.0 keeps the
            data unchanged, as in the paper.

    Raises:
        ValueError: If an argument is outside its allowed values.

    Examples:
        >>> from pyhealth.datasets import TCGAMultimodalDataset
        >>> from pyhealth.tasks import MultimodalCancerSurvival
        >>> dataset = TCGAMultimodalDataset(root="/path/to/tcga_multimodal")
        >>> task = MultimodalCancerSurvival(
        ...     modalities=["demographics", "cancer_type", "expression"],
        ...     duration_unit="years",
        ... )
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MultimodalCancerSurvival"
    input_schema: Dict[str, str] = dict(_SCHEMA)
    output_schema: Dict[str, str] = {"duration": "tensor", "event": "tensor"}

    def __init__(
        self,
        modalities: Sequence[str] = ALL_MODALITIES,
        aggregation: str = "mean",
        duration_unit: str = "days",
        min_duration: float = 0.0,
    ) -> None:
        super().__init__()
        unknown = sorted(set(modalities) - set(ALL_MODALITIES))
        if unknown or not modalities:
            raise ValueError(
                f"modalities must be a non-empty subset of {list(ALL_MODALITIES)}; "
                f"got unknown {unknown}"
            )
        if aggregation not in _AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {list(_AGGREGATIONS)}")
        if duration_unit not in _DURATION_UNITS:
            raise ValueError(f"duration_unit must be one of {list(_DURATION_UNITS)}")
        if min_duration < 0:
            raise ValueError("min_duration must be >= 0")
        # Plain str/float/list attributes: BaseDataset.set_task serializes
        # vars(task) to JSON to build its cache key.
        self.modalities: List[str] = [m for m in ALL_MODALITIES if m in modalities]
        self.aggregation = aggregation
        self.duration_unit = duration_unit
        self.min_duration = float(min_duration)
        self.input_schema = {m: _SCHEMA[m] for m in self.modalities}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Builds the survival sample for one patient.

        Args:
            patient: A ``Patient`` from ``TCGAMultimodalDataset``.

        Returns:
            A list with one sample dict, or an empty list if the patient is
            excluded (the reason is logged at INFO level).
        """
        clinical = patient.get_events(event_type="clinical")
        if len(clinical) != 1:
            return self._exclude(patient, f"{len(clinical)} clinical records")
        record = clinical[0]
        label = self._survival_label(record)
        if label is None:
            return self._exclude(patient, "missing or invalid survival outcome")

        sample: Dict[str, Any] = {"patient_id": patient.patient_id, **label}
        for modality in self.modalities:
            if modality == "demographics":
                sample[modality] = self._demographic_tokens(record)
            elif modality == "cancer_type":
                sample[modality] = [_category(getattr(record, "project", None))]
            else:
                embedding = self._embedding(patient, modality)
                if embedding is None:
                    return self._exclude(patient, f"no {modality} embedding")
                sample[modality] = embedding
        return [sample]

    def _survival_label(self, record: Any) -> Optional[Dict[str, List[float]]]:
        """Derives ``duration`` and ``event`` from a clinical record.

        Args:
            record: The patient's clinical event.

        Returns:
            ``{"duration": [d], "event": [e]}``, or None if vital status is
            not "Dead"/"Alive" or the relevant day count is missing or negative.
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
        duration = max(days / _DURATION_UNITS[self.duration_unit], self.min_duration)
        return {"duration": [duration], "event": [event]}

    @staticmethod
    def _demographic_tokens(record: Any) -> List[str]:
        """Encodes sex, binned age, race, and ethnicity as category tokens.

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

    def _embedding(self, patient: Any, modality: str) -> Optional[np.ndarray]:
        """Loads and combines a patient's embedding files for one modality.

        Args:
            patient: The patient.
            modality: One of ``EMBEDDING_MODALITIES``.

        Returns:
            A 1-D float32 vector, or None if the patient has no files.

        Raises:
            FileNotFoundError: If an indexed ``.npy`` file no longer exists.
            ValueError: If the patient's files differ in shape or are not 1-D.
        """
        events = sorted(
            patient.get_events(event_type=modality), key=lambda e: str(e.file_id)
        )
        if not events:
            return None
        if self.aggregation == "first":
            events = events[:1]
        vectors = [np.load(event.embedding_path) for event in events]
        shapes = {v.shape for v in vectors}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(
                f"{modality} embeddings of patient {patient.patient_id} must be "
                f"1-D with one shape; got {sorted(shapes)}"
            )
        return np.mean(np.stack(vectors), axis=0).astype(np.float32)

    @staticmethod
    def _exclude(patient: Any, reason: str) -> List[Dict[str, Any]]:
        """Logs why a patient is excluded and returns no samples.

        Args:
            patient: The excluded patient.
            reason: Human-readable reason.

        Returns:
            An empty list.
        """
        logger.info(f"Excluding patient {patient.patient_id}: {reason}")
        return []


def _to_float(value: Any) -> Optional[float]:
    """Converts a raw attribute to float, returning None if missing or invalid.

    Args:
        value: Raw value; CSV attributes may arrive as strings.

    Returns:
        The float value, or None for None, empty, NaN, or non-numeric input.
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


def _age_bin(age: Optional[float]) -> str:
    """Bins age into the paper's 20-year intervals, right-inclusive.

    Args:
        age: Age in years, or None.

    Returns:
        A label such as ``"(40, 60]"``, or ``"unknown"`` outside (0, 100].
    """
    if age is None or not 0 < age <= 100:
        return "unknown"
    upper = 20 * math.ceil(age / 20)
    return f"({upper - 20}, {upper}]"
