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

from typing import Any

from .base_task import BaseTask

AGE_BINS = (0, 20, 40, 60, 80, 100)
EMBEDDING_MODALITIES = ("expr", "hist", "text")


class MultimodalSurvivalPrediction(BaseTask):
    """Predicts overall survival from embeddings and clinical features.

    Emits one sample per patient with all modalities:

    - ``expr``, ``hist``, ``text``: raw precomputed embeddings (RNA-seq,
      histology, pathology report), standardized and PCA-reduced by
      :class:`~pyhealth.processors.EmbeddingPCAProcessor`.
    - ``demo``: tokens for sex, age bin (as in the paper: ``(0, 20]`` to
      ``(80, 100]``), race and ethnicity, one-hot encoded by ``multi_hot``.
    - ``canc``: the TCGA project as a single ``multi_hot`` token.
    - ``duration``: days to death, or to last follow-up when censored.
    - ``event``: 1 if the patient died, else 0.

    Args:
        expr_components: PCA components for ``expr``, or ``None`` for none.
        hist_components: PCA components for ``hist``, or ``None`` for none.
        text_components: PCA components for ``text``, or ``None`` for none.

    Examples:
        >>> from pyhealth.datasets import TCGAMultimodalDataset
        >>> from pyhealth.tasks import MultimodalSurvivalPrediction
        >>> dataset = TCGAMultimodalDataset(root="/path/to/data")
        >>> samples = dataset.set_task(MultimodalSurvivalPrediction())
    """

    task_name: str = "MultimodalSurvivalPrediction"
    output_schema: dict[str, str] = {"duration": "tensor", "event": "tensor"}

    def __init__(
        self,
        expr_components: int | None = 256,
        hist_components: int | None = 256,
        text_components: int | None = 256,
    ) -> None:
        # TODO: replace the placeholder 256s with the paper's PCA dims.
        components = (expr_components, hist_components, text_components)
        self.input_schema: dict[str, Any] = {
            **{
                modality: ("embedding_pca", {"n_components": n})
                for modality, n in zip(EMBEDDING_MODALITIES, components)
            },
            "demo": "multi_hot",
            "canc": "multi_hot",
        }
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the patient's sample.

        Args:
            patient: A patient of :class:`~pyhealth.datasets.TCGAMultimodalDataset`.

        Returns:
            One sample, or an empty list when a modality or the outcome is
            missing or invalid.
        """
        # TODO: read the clinical event and one event per embedding table;
        # derive duration (days_to_death if vital_status == "Dead", else
        # days_to_last_follow_up) and event; skip missing/invalid patients.
        # The authors' data has 0-day durations: keep them (Cox handles them).
        raise NotImplementedError
