"""TCGA multimodal survival dataset for PyHealth.

Precomputed foundation-model embeddings and clinical data for TCGA cases, as
used by Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
Embeddings", ML4H 2025 (https://arxiv.org/abs/2505.07683).
"""

import logging
from pathlib import Path

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TCGAMultimodalDataset(BaseDataset):
    """TCGA cases with clinical data and per-modality embeddings.

    Each patient (TCGA case) has one ``clinical`` event and one event per
    embedding table (``expr``: BulkRNABert RNA-seq, ``hist``: UNI2-h
    histology, ``text``: BioMistral pathology-report summary). Embeddings are
    generated outside PyHealth with the authors' code
    (https://github.com/StevenSong/multimodal-cancer-modeling).

    Expected layout under ``root``::

        clinical.csv              case_id, project, sex, age, race, ethnicity,
                                  vital_status, days_to_death,
                                  days_to_last_follow_up
        embeddings/expr.parquet   case_id, embedding (list<float32>), n_sources
        embeddings/hist.parquet   same
        embeddings/text.parquet   same

    Args:
        root: Directory with the layout above.
        tables: Additional tables to load beyond the defaults.
        dataset_name: Name of the dataset. Defaults to ``"tcga_survival"``.
        config_path: Path to the YAML config. Defaults to
            ``configs/tcga_survival.yaml``.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`.

    Raises:
        FileNotFoundError: If a required file is missing under ``root``.

    Examples:
        >>> from pyhealth.datasets import TCGAMultimodalDataset
        >>> dataset = TCGAMultimodalDataset(root="/path/to/data")
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "tcga_survival.yaml"
        # TODO: check the required files exist (fail fast with the missing
        # path), then call super().__init__ with the default tables
        # ["clinical", "expr", "hist", "text"] plus `tables`, like tcga_prad.py.
        raise NotImplementedError
