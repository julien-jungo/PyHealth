"""TCGA multimodal survival dataset for PyHealth.

Contributors: 
    Julien Jungo (jjungo2@illinois.edu), 
    Donovan Below (dbelow2@illinois.edu), 
    Yaseen Mohamed (yaseenm2@illinois.edu)
Paper: Song, Borjigin-Wang, Madejski, and Grossman. "Multimodal Cancer Modeling
    in the Age of Foundation Model Embeddings." ML4H 2025, PMLR 297.
Paper link: https://proceedings.mlr.press/v297/song26a.html
Description: Loads the pan-cancer TCGA cohort used by the paper. Each patient
    has one clinical record (demographics, cancer type, survival outcome, and
    the paper's cross-validation fold) and zero or more precomputed
    foundation-model embeddings for gene expression, whole-slide histology,
    and pathology reports.
"""


# TODO - regarding all embeddings and data tables, need to ensure format is identical to what
#   Yaseen produces


import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import polars as pl

from ..tasks import MultimodalCancerSurvival
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

EMBEDDING_TABLES: tuple = ("expression", "histology", "report")
REQUIRED_CLINICAL_COLUMNS: tuple = (
    "case_id",
    "project",
    "sex",
    "age",
    "race",
    "ethnicity",
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up",
)
_CLINICAL_CSV = "tcga_multimodal_clinical-pyhealth.csv"
_INDEX_COLUMNS: List[str] = ["patient_id", "file_id", "embedding_path"]
_MISSING_VALUES = ("", "nan", "None", "<NA>")


class TCGAMultimodalDataset(BaseDataset):
    """Pan-cancer TCGA cohort with clinical data and foundation-model embeddings.

    This dataset follows the cohort of Song et al. (ML4H 2025): 7,982 TCGA
    patients across 32 cancer types. It loads one clinical table plus one
    table per requested embedding modality. Embeddings are not computed here;
    they are read from ``.npy`` files produced beforehand (for example, with
    the authors' ``embed/`` scripts and converted from HDF5 to ``.npy``).

    Expected layout of ``root``::

        root/
            clinical.csv          # authors' data/clinical.csv (required)
            split_cases.csv       # authors' results/split_cases.csv (optional)
            embeddings/
                expression/<patient_id>/<file_id>.npy
                histology/<patient_id>/<file_id>.npy
                report/<patient_id>/<file_id>.npy

    A patient may have several ``.npy`` files per modality (for example, two
    slides). Each file becomes its own event, so no sample is dropped or
    averaged at this stage; tasks decide how to combine them. A patient with
    no file for a modality simply has no events of that type.

    On first use, :meth:`prepare_metadata` writes standardized
    ``tcga_multimodal_*-pyhealth.csv`` files into ``root``. Delete those files
    to rebuild them after adding, moving, or replacing input files.

    Args:
        root: Directory containing ``clinical.csv`` and ``embeddings/``.
        tables: Embedding tables to load, any of ``"expression"``,
            ``"histology"``, and ``"report"``. Defaults to all three; pass
            ``[]`` for clinical data only. The ``"clinical"`` table is always
            loaded because it holds the labels. Each requested table must
            have at least one ``.npy`` file.
        dataset_name: Name of the dataset. Defaults to ``"tcga_multimodal"``.
        config_path: Path to the YAML config. Defaults to the bundled
            ``configs/tcga_multimodal.yaml``.
        cache_dir: Directory for PyHealth's processed-data cache.
        num_workers: Number of worker processes for parallel operations.
        dev: If True, load at most 1,000 patients.

    Raises:
        FileNotFoundError: If ``clinical.csv`` or a requested
            ``embeddings/<modality>/`` folder is missing.
        ValueError: If ``tables`` names an unknown table, ``clinical.csv`` or
            ``split_cases.csv`` is malformed, or a requested modality folder
            holds no ``<patient_id>/<file_id>.npy`` files.

    Examples:
        >>> from pyhealth.datasets import TCGAMultimodalDataset
        >>> dataset = TCGAMultimodalDataset(root="/path/to/tcga_multimodal")
        >>> dataset.stats()
        >>> patient = dataset.get_patient("TCGA-OR-A5J1")
        >>> slides = patient.get_events(event_type="histology")
        >>> folds = dataset.get_fold_assignments()  # {"TCGA-OR-A5J1": 2, ...}
        >>> samples = dataset.set_task()  # MultimodalCancerSurvival
    """

    def __init__(
        self,
        root: str,
        tables: Optional[Sequence[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        embedding_tables = list(EMBEDDING_TABLES if tables is None else tables)
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "tcga_multimodal.yaml"
            )
        self.prepare_metadata(root, embedding_tables)
        super().__init__(
            root=root,
            tables=["clinical"] + embedding_tables,
            dataset_name=dataset_name or "tcga_multimodal",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str, tables: Sequence[str] = EMBEDDING_TABLES) -> None:
        """Writes the standardized CSV files that the YAML config points to.

        Files that already exist are left untouched. Called automatically by
        ``__init__``.

        Args:
            root: Dataset root directory (see the class docstring for layout).
            tables: Embedding tables to index. Defaults to all three.

        Raises:
            FileNotFoundError: If ``clinical.csv`` or a requested modality
                folder is missing.
            ValueError: If ``tables`` names an unknown table, an input file is
                malformed, or a requested modality folder has no ``.npy``
                files.
        """
        unknown = sorted(set(tables) - set(EMBEDDING_TABLES))
        if unknown:
            raise ValueError(
                f"Unknown table(s) {unknown}; choose from {list(EMBEDDING_TABLES)}"
            )
        TCGAMultimodalDataset._prepare_clinical(root)
        for modality in tables:
            TCGAMultimodalDataset._prepare_embedding_index(root, modality)

    @staticmethod
    def _prepare_clinical(root: str) -> None:
        """Copies ``clinical.csv`` and adds the paper's fold as a ``fold`` column.

        Every row and column of ``clinical.csv`` is kept. Patients missing
        from ``split_cases.csv`` (or all patients, if that file is absent)
        get an empty ``fold``.

        Args:
            root: Dataset root directory.

        Raises:
            FileNotFoundError: If ``clinical.csv`` is missing.
            ValueError: If required columns are missing or ``case_id`` repeats.
        """
        output_path = os.path.join(root, _CLINICAL_CSV)
        if os.path.exists(output_path):
            return
        clinical_path = os.path.join(root, "clinical.csv")
        if not os.path.exists(clinical_path):
            raise FileNotFoundError(
                f"Expected {clinical_path}. Copy data/clinical.csv from "
                "https://github.com/StevenSong/multimodal-cancer-modeling."
            )
        clinical = pd.read_csv(clinical_path)
        missing = [c for c in REQUIRED_CLINICAL_COLUMNS if c not in clinical.columns]
        if missing:
            raise ValueError(f"{clinical_path} is missing columns: {missing}")
        duplicated = clinical["case_id"][clinical["case_id"].duplicated()]
        if not duplicated.empty:
            raise ValueError(
                f"{clinical_path} repeats case_id values, e.g. "
                f"{duplicated.iloc[:3].tolist()}"
            )

        clinical["fold"] = pd.array([pd.NA] * len(clinical), dtype="Int64")
        split_path = os.path.join(root, "split_cases.csv")
        if os.path.exists(split_path):
            folds = TCGAMultimodalDataset._read_folds(split_path)
            clinical["fold"] = clinical["case_id"].map(folds).astype("Int64")
            unknown = len(set(folds.index) - set(clinical["case_id"]))
            if unknown:
                logger.warning(
                    f"{unknown} case_id(s) in {split_path} are not in "
                    f"{clinical_path}; their folds are ignored."
                )
        else:
            logger.warning(f"No {split_path}; every patient's fold is empty.")
        logger.info(
            f"{int(clinical['fold'].isna().sum())} of {len(clinical)} patients "
            "have no fold assignment."
        )
        clinical.to_csv(output_path, index=False)

    @staticmethod
    def _read_folds(split_path: str) -> pd.Series:
        """Reads the paper's fold file into a ``case_id -> fold`` series.

        Args:
            split_path: Path to the authors' ``split_cases.csv``.

        Returns:
            Fold index (0-4 in the paper) for each ``case_id``.

        Raises:
            ValueError: If columns are missing or a ``case_id`` repeats.
        """
        splits = pd.read_csv(split_path)
        missing = [c for c in ("case_id", "split") if c not in splits.columns]
        if missing:
            raise ValueError(f"{split_path} is missing columns: {missing}")
        if splits["case_id"].duplicated().any():
            raise ValueError(f"{split_path} assigns some case_id to several folds")
        return splits.set_index("case_id")["split"]

    @staticmethod
    def _prepare_embedding_index(root: str, modality: str) -> None:
        """Writes one row per ``.npy`` file under ``embeddings/<modality>/``.

        Files must sit in a per-patient folder named with the 12-character
        patient barcode. Files elsewhere are reported in a warning rather
        than indexed. Nothing is written when an error is raised, so a fixed
        folder is picked up on the next run.

        Args:
            root: Dataset root directory.
            modality: One of ``EMBEDDING_TABLES``.

        Raises:
            FileNotFoundError: If the modality folder is missing.
            ValueError: If the folder holds no ``<patient_id>/<file_id>.npy``.
        """
        output_path = os.path.join(root, f"tcga_multimodal_{modality}-pyhealth.csv")
        if os.path.exists(output_path):
            return
        modality_dir = Path(root) / "embeddings" / modality
        if not modality_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {modality} embeddings in {modality_dir}/<patient_id>/"
                f"<file_id>.npy. To load without them, leave {modality!r} out "
                "of `tables`."
            )
        stray = [p.name for p in modality_dir.iterdir() if p.is_file()]
        if stray:
            logger.warning(
                f"Ignoring {len(stray)} file(s) directly inside {modality_dir}; "
                f"expected <patient_id>/<file_id>.npy, e.g. {stray[:3]}"
            )
        rows = [
            (patient_dir.name, npy_path.stem, str(npy_path.resolve()))
            for patient_dir in sorted(p for p in modality_dir.iterdir() if p.is_dir())
            for npy_path in sorted(patient_dir.glob("*.npy"))
        ]
        if not rows:
            raise ValueError(f"No <patient_id>/<file_id>.npy files in {modality_dir}.")
        index = pd.DataFrame(rows, columns=_INDEX_COLUMNS)
        logger.info(
            f"Indexed {len(index)} {modality} file(s) for "
            f"{index['patient_id'].nunique()} patient(s)."
        )
        index.to_csv(output_path, index=False)

    def get_fold_assignments(self) -> Dict[str, int]:
        """Returns the paper's cross-validation fold of each patient.

        Use this to rebuild the paper's five splits (fold ``k`` is the test
        set of split ``k``) or to nest an inner split inside each training set.

        Returns:
            Mapping from patient ID to fold index. Patients without a fold
            are absent from the mapping.

        Raises:
            ValueError: If no patient has a fold.

        Examples:
            >>> folds = dataset.get_fold_assignments()
            >>> test_ids = [pid for pid, fold in folds.items() if fold == 0]
        """
        folds = {
            patient_id: int(float(fold))
            for patient_id, fold in self._clinical_values("fold").items()
        }
        if not folds:
            raise ValueError(
                f"No patient has a fold. Put the authors' split_cases.csv in "
                f"{self.root}, delete {_CLINICAL_CSV} there, and reload."
            )
        return folds

    def get_cancer_types(self) -> Dict[str, str]:
        """Returns each patient's cancer type (TCGA project, e.g. "TCGA-BRCA").

        Use this to score results per cancer type, as in the paper's Table 17.

        Returns:
            Mapping from patient ID to TCGA project ID.

        Examples:
            >>> cancer_types = dataset.get_cancer_types()
            >>> brca = [p for p, c in cancer_types.items() if c == "TCGA-BRCA"]
        """
        return self._clinical_values("project")

    def _clinical_values(self, column: str) -> Dict[str, str]:
        """Reads one clinical column for every patient that has a value.

        Args:
            column: Attribute name of the clinical table.

        Returns:
            Mapping from patient ID to the raw value as a string; patients
            with a missing value are left out.
        """
        rows = (
            self.global_event_df.filter(pl.col("event_type") == "clinical")
            .select("patient_id", f"clinical/{column}")
            .collect()
            .iter_rows()
        )
        return {
            patient_id: str(value)
            for patient_id, value in rows
            if value is not None and str(value) not in _MISSING_VALUES
        }

    @property
    def default_task(self) -> MultimodalCancerSurvival:
        """Returns the default task: survival with all five modalities.

        Returns:
            MultimodalCancerSurvival: Task using every modality.
        """
        return MultimodalCancerSurvival()
