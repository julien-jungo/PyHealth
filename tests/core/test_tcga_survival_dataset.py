"""Tests for TCGAMultimodalDataset and MultimodalSurvivalPrediction.

Contributors:
    Julien Jungo (jjungo2@illinois.edu),
    Donovan Below (dbelow2@illinois.edu),
    Yaseen Mohamed (yaseenm2@illinois.edu)

All data are synthetic. The dataset tests write five fake patients to a
temporary directory, with 2-4 dimensional embeddings; each covers one edge case:

    P1  dead, all modalities, two histology slides (aggregation)
    P2  alive, all modalities, zero follow-up days (min_duration)
    P3  alive, no histology embedding (missing modality)
    P4  dead, all modalities, age 85, not in the fold file
    P5  vital status "Not Reported" (invalid label)

The task tests build Patient objects by hand, so they need no dataset build
(no Dask start-up) and each runs in milliseconds.
"""

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import TCGAMultimodalDataset
from pyhealth.tasks import MultimodalSurvivalPrediction

P1, P2, P3, P4, P5 = (f"TCGA-AA-000{i}" for i in range(1, 6))
EMBEDDING_FILES = {
    "expr": {pid: {"rna-01A": [1.0, 2.0, 3.0, 4.0]} for pid in (P1, P2, P3, P4)},
    "hist": {
        P1: {"DX1": [0.0, 2.0, 4.0], "DX2": [2.0, 4.0, 6.0]},
        P2: {"DX1": [1.0, 1.0, 1.0]},
        P4: {"DX1": [3.0, 3.0, 3.0]},
    },
    "text": {pid: {"report": [0.5, -0.5]} for pid in (P1, P2, P3, P4)},
}


def write_synthetic_root(root: Path) -> None:
    """Writes clinical.csv, folds.csv, and .npy embeddings into root."""
    pd.DataFrame(
        {
            "case_id": [P1, P2, P3, P4, P5],
            "project": ["TCGA-BRCA", "TCGA-LUAD", "TCGA-THCA", "TCGA-BRCA", "TCGA-ACC"],
            "sex": ["female", "male", "female", "male", "female"],
            "age": [55.0, 67.0, 41.0, 85.0, 30.0],
            "race": ["white", "asian", "white", "not reported", "white"],
            "ethnicity": ["not hispanic or latino"] * 5,
            "vital_status": ["Dead", "Alive", "Alive", "Dead", "Not Reported"],
            "days_to_death": [340.0, None, None, 730.5, None],
            "days_to_last_follow_up": [340.0, 0.0, 2100.0, 730.5, 50.0],
        }
    ).to_csv(root / "clinical.csv", index=False)
    pd.DataFrame({"case_id": [P1, P2, P3, P5], "fold": [0, 1, 0, 1]}).to_csv(
        root / "folds.csv", index=False
    )
    for modality, patients in EMBEDDING_FILES.items():
        for pid, files in patients.items():
            folder = root / "embeddings" / modality / pid
            folder.mkdir(parents=True)
            for file_id, vector in files.items():
                np.save(folder / f"{file_id}.npy", np.array(vector, dtype=np.float32))


class TestTCGAMultimodal(unittest.TestCase):
    """Dataset tests sharing one synthetic dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "data"
        cls.root.mkdir()
        write_synthetic_root(cls.root)
        cls.dataset = TCGAMultimodalDataset(
            root=str(cls.root), cache_dir=str(cls.tmp / "cache")
        )
        # Building BaseDataset's event cache starts Dask once (~2 s); doing it
        # here keeps every individual test in the millisecond range.
        _ = cls.dataset.global_event_df

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    # Dataset: loading and parsing

    def test_dataset_name_and_patients(self) -> None:
        self.assertEqual(self.dataset.dataset_name, "tcga_survival")
        self.assertEqual(sorted(self.dataset.unique_patient_ids), [P1, P2, P3, P4, P5])

    def test_prepare_metadata_writes_index_files(self) -> None:
        for table in ("clinical", "expr", "hist", "text"):
            path = self.root / f"tcga_multimodal_{table}-pyhealth.csv"
            self.assertTrue(path.exists(), path)
        index = pd.read_csv(self.root / "tcga_multimodal_hist-pyhealth.csv")
        self.assertEqual(list(index.columns), ["case_id", "file_id", "embedding_path"])
        self.assertEqual(len(index), 4)  # every slide file is indexed

    def test_events_per_table(self) -> None:
        p1, p3 = self.dataset.get_patient(P1), self.dataset.get_patient(P3)
        self.assertEqual(len(p1.get_events(event_type="clinical")), 1)
        self.assertEqual(len(p1.get_events(event_type="hist")), 2)
        self.assertEqual(len(p3.get_events(event_type="hist")), 0)
        self.assertEqual(len(p3.get_events(event_type="expr")), 1)

    def test_clinical_columns_preserved(self) -> None:
        record = self.dataset.get_patient(P1).get_events(event_type="clinical")[0]
        self.assertEqual(record.project, "TCGA-BRCA")
        self.assertEqual(float(record.days_to_death), 340.0)
        for column in ("sex", "age", "race", "ethnicity", "vital_status", "fold"):
            self.assertIn(column, record)

    def test_embedding_paths_point_to_files(self) -> None:
        event = self.dataset.get_patient(P2).get_events(event_type="text")[0]
        self.assertEqual(event.file_id, "report")
        np.testing.assert_allclose(np.load(event.embedding_path), [0.5, -0.5])

    def test_fold_assignments(self) -> None:
        folds = self.dataset.get_fold_assignments()
        self.assertEqual(folds, {P1: 0, P2: 1, P3: 0, P5: 1})  # P4 has no fold

    def test_no_folds_raises(self) -> None:
        # Same path as a root without folds.csv: no patient has a fold.
        with (
            mock.patch.object(self.dataset, "_clinical_values", return_value={}),
            self.assertRaisesRegex(ValueError, "folds.csv"),
        ):
            self.dataset.get_fold_assignments()

    def test_cancer_types(self) -> None:
        self.assertEqual(
            self.dataset.get_cancer_types(),
            {
                P1: "TCGA-BRCA",
                P2: "TCGA-LUAD",
                P3: "TCGA-THCA",
                P4: "TCGA-BRCA",
                P5: "TCGA-ACC",
            },
        )

    def test_default_task(self) -> None:
        self.assertIsInstance(self.dataset.default_task, MultimodalSurvivalPrediction)

    # Dataset: validation

    def test_missing_clinical_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as empty,
            self.assertRaises(FileNotFoundError),
        ):
            TCGAMultimodalDataset.prepare_metadata(empty)

    def test_malformed_clinical_raises(self) -> None:
        cases = {
            "missing column": pd.DataFrame({"case_id": [P1]}),
            "duplicate id": pd.read_csv(self.root / "clinical.csv").iloc[[0, 0]],
        }
        for name, frame in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                frame.to_csv(Path(tmp) / "clinical.csv", index=False)
                with self.assertRaises(ValueError):
                    TCGAMultimodalDataset.prepare_metadata(tmp)

    def test_unknown_table_raises(self) -> None:
        with self.assertRaises(ValueError):
            TCGAMultimodalDataset(root=str(self.root), tables=["genomics"])

    def test_missing_or_empty_modality_folder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(self.root / "clinical.csv", tmp)
            with self.assertRaisesRegex(FileNotFoundError, "embeddings"):
                TCGAMultimodalDataset.prepare_metadata(tmp, tables=["text"])
            # A file in the wrong place doesn't count: still an error.
            (Path(tmp) / "embeddings" / "text").mkdir(parents=True)
            np.save(Path(tmp) / "embeddings" / "text" / "stray.npy", np.zeros(2))
            with self.assertRaisesRegex(ValueError, "No <case_id>"):
                TCGAMultimodalDataset.prepare_metadata(tmp, tables=["text"])
            # Nothing is written on error, so fixing the folder is enough.
            index = Path(tmp) / "tcga_multimodal_text-pyhealth.csv"
            self.assertFalse(index.exists())

    def test_tables_subset_needs_only_those_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(self.root / "clinical.csv", tmp)
            shutil.copytree(
                self.root / "embeddings" / "expr",
                Path(tmp) / "embeddings" / "expr",
            )
            TCGAMultimodalDataset.prepare_metadata(tmp, tables=["expr"])
            written = sorted(p.name for p in Path(tmp).glob("*-pyhealth.csv"))
            self.assertEqual(
                written,
                [
                    "tcga_multimodal_clinical-pyhealth.csv",
                    "tcga_multimodal_expr-pyhealth.csv",
                ],
            )


def make_patient(
    pid: str,
    clinical: dict[str, Any] | None,
    embeddings: dict[str, dict[str, Path]] | None = None,
) -> Patient:
    """Builds a Patient like TCGAMultimodalDataset does, without a dataset.

    Attribute values are strings, as they come out of the dataset's cache.
    ``embeddings`` maps event type -> file ID -> .npy path.
    """
    rows: list[dict[str, Any]] = []
    if clinical is not None:
        rows.append(
            {"event_type": "clinical"}
            | {
                f"clinical/{k}": None if v is None else str(v)
                for k, v in clinical.items()
            }
        )
    for event_type, files in (embeddings or {}).items():
        for file_id, path in files.items():
            rows.append(
                {
                    "event_type": event_type,
                    f"{event_type}/file_id": file_id,
                    f"{event_type}/embedding_path": str(path),
                }
            )
    frame = pl.DataFrame(rows).with_columns(
        pl.lit(pid).alias("patient_id"),
        pl.lit(None, dtype=pl.Datetime("ms")).alias("timestamp"),
    )
    return Patient(patient_id=pid, data_source=frame)


CLINICAL = {
    "project": "TCGA-BRCA",
    "sex": "male",
    "age": 85.0,
    "race": None,
    "ethnicity": "not hispanic or latino",
    "vital_status": "Dead",
    "days_to_death": 340.0,
    "days_to_last_follow_up": 340.0,
}


class TestMultimodalSurvivalPrediction(unittest.TestCase):
    """Task tests on hand-built patients."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.files: dict[str, Path] = {}
        vectors = {
            "rna": [1.0, 2.0, 3.0, 4.0],
            "DX1": [0.0, 2.0, 4.0],
            "DX2": [2.0, 4.0, 6.0],
            "report": [0.5, -0.5],
        }
        for name, vector in vectors.items():
            cls.files[name] = cls.tmp / f"{name}.npy"
            np.save(cls.files[name], np.array(vector, dtype=np.float32))
        cls.all_embeddings = {
            "expr": {"rna": cls.files["rna"]},
            "hist": {"DX1": cls.files["DX1"], "DX2": cls.files["DX2"]},
            "text": {"report": cls.files["report"]},
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    def run_task(
        self,
        clinical: dict[str, Any] | None = CLINICAL,
        embeddings: dict[str, dict[str, Path]] | None = None,
        **task_args: Any,
    ) -> list[dict[str, Any]]:
        """Runs the task on one hand-built patient."""
        if embeddings is None:
            embeddings = self.all_embeddings
        patient = make_patient("TCGA-AA-0001", clinical, embeddings)
        return MultimodalSurvivalPrediction(**task_args)(patient)

    def test_schema(self) -> None:
        self.assertEqual(
            MultimodalSurvivalPrediction().input_schema,
            {
                "expr": ("embedding_pca", {"n_components": 256}),
                "hist": ("embedding_pca", {"n_components": 256}),
                "text": ("embedding_pca", {"n_components": 256}),
                "demo": "multi_hot_ignore_unknown",
                "canc": "multi_hot_ignore_unknown",
            },
        )
        task = MultimodalSurvivalPrediction(
            text_components=None, modalities=["canc", "text"]
        )
        self.assertEqual(
            task.input_schema,
            {
                "text": ("embedding_pca", {"n_components": None}),
                "canc": "multi_hot_ignore_unknown",
            },
        )
        self.assertEqual(task.output_schema, {"duration": "tensor", "event": "tensor"})

    def test_dead_patient_duration_and_event(self) -> None:
        (sample,) = self.run_task()
        self.assertEqual(sample["duration"], [340.0])
        self.assertEqual(sample["event"], [1.0])

    def test_censored_patient_duration_and_event(self) -> None:
        clinical = CLINICAL | {
            "vital_status": "Alive",
            "days_to_death": None,
            "days_to_last_follow_up": 0.0,
        }
        (sample,) = self.run_task(clinical)
        self.assertEqual(sample["duration"], [0.0])  # kept, as in the paper
        self.assertEqual(sample["event"], [0.0])

    def test_demo_and_canc_tokens(self) -> None:
        (sample,) = self.run_task()
        self.assertEqual(
            sample["demo"],
            [
                "sex:male",
                "age:(80, 100]",
                "race:unknown",
                "ethnicity:not hispanic or latino",
            ],
        )
        self.assertEqual(sample["canc"], ["TCGA-BRCA"])

    def test_embeddings_mean_and_first(self) -> None:
        (mean,) = self.run_task()
        (first,) = self.run_task(aggregation="first")
        np.testing.assert_allclose(mean["hist"], [1.0, 3.0, 5.0])
        np.testing.assert_allclose(first["hist"], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(mean["expr"], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(mean["text"], [0.5, -0.5])

    def test_patient_missing_modality_skipped(self) -> None:
        embeddings = {k: v for k, v in self.all_embeddings.items() if k != "hist"}
        with self.assertLogs("pyhealth.tasks.survival", "INFO") as log:
            self.assertEqual(self.run_task(embeddings=embeddings), [])
        self.assertIn("no hist embedding", log.output[0])

    def test_fewer_modalities_keep_patient(self) -> None:
        (sample,) = self.run_task(
            embeddings={"text": {"report": self.files["report"]}},
            modalities=["text", "demo", "canc"],
        )
        self.assertEqual(
            set(sample), {"patient_id", "text", "demo", "canc", "duration", "event"}
        )

    def test_patient_missing_outcome_skipped(self) -> None:
        cases = {
            "unknown status": CLINICAL | {"vital_status": "Not Reported"},
            "dead without days": CLINICAL | {"days_to_death": None},
            "negative days": CLINICAL | {"days_to_death": -5.0},
        }
        for name, clinical in cases.items():
            with self.subTest(name):
                self.assertEqual(self.run_task(clinical), [])

    def test_patient_without_clinical_record_skipped(self) -> None:
        self.assertEqual(self.run_task(clinical=None), [])

    def test_invalid_arguments_raise(self) -> None:
        for kwargs in (
            {"modalities": ["genomics"]},
            {"modalities": []},
            {"aggregation": "max"},
            {"expr_components": 0},
            {"hist_components": 2.5},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MultimodalSurvivalPrediction(**kwargs)

    def test_arguments_are_json_serializable(self) -> None:
        task = MultimodalSurvivalPrediction(modalities=("text", "demo"))
        json.dumps(vars(task))  # set_task builds its cache key this way
        self.assertEqual(task.modalities, ["text", "demo"])

    @unittest.skip("needs EmbeddingPCAProcessor.fit/process (still a stub)")
    def test_set_task_with_split_fits_on_train_only(self) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
