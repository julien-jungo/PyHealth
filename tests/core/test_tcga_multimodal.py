"""Tests for TCGAMultimodalDataset and MultimodalCancerSurvival.

Contributors: 
    Julien Jungo (jjungo2@illinois.edu), 
    Donovan Below (dbelow2@illinois.edu), 
    Yaseen Mohamed (yaseenm2@illinois.edu)

All data are synthetic: five fake patients written to a temporary directory,
with 2-4 dimensional embeddings. Each patient covers one edge case:

    P1  dead, all modalities, two histology slides (aggregation)
    P2  alive, all modalities, zero follow-up days (min_duration)
    P3  alive, no histology embedding (missing modality)
    P4  dead, all modalities, age 85, not in the fold file
    P5  vital status "Not Reported" (invalid label)
"""

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pyhealth.datasets import TCGAMultimodalDataset
from pyhealth.tasks import MultimodalCancerSurvival

P1, P2, P3, P4, P5 = (f"TCGA-AA-000{i}" for i in range(1, 6))
EMBEDDING_FILES = {
    "expression": {pid: {"rna-01A": [1.0, 2.0, 3.0, 4.0]} for pid in (P1, P2, P3, P4)},
    "histology": {
        P1: {"DX1": [0.0, 2.0, 4.0], "DX2": [2.0, 4.0, 6.0]},
        P2: {"DX1": [1.0, 1.0, 1.0]},
        P4: {"DX1": [3.0, 3.0, 3.0]},
    },
    "report": {pid: {"report": [0.5, -0.5]} for pid in (P1, P2, P3, P4)},
}


def write_synthetic_root(root: Path) -> None:
    """Writes clinical.csv, split_cases.csv, and .npy embeddings into root."""
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
    pd.DataFrame({"case_id": [P1, P2, P3, P5], "split": [0, 1, 0, 1]}).to_csv(
        root / "split_cases.csv", index=False
    )
    for modality, patients in EMBEDDING_FILES.items():
        for pid, files in patients.items():
            folder = root / "embeddings" / modality / pid
            folder.mkdir(parents=True)
            for file_id, vector in files.items():
                np.save(folder / f"{file_id}.npy", np.array(vector, dtype=np.float32))


class TestTCGAMultimodal(unittest.TestCase):
    """Dataset and task tests sharing one synthetic dataset."""

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
        cls.dataset.global_event_df
        cls.task = MultimodalCancerSurvival()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    def samples(self, task: MultimodalCancerSurvival) -> dict:
        """Runs a task on every patient; returns samples keyed by patient ID."""
        return {
            s["patient_id"]: s
            for pid in sorted(self.dataset.unique_patient_ids)
            for s in task(self.dataset.get_patient(pid))
        }

    # Dataset: loading and parsing

    def test_dataset_name_and_patients(self) -> None:
        self.assertEqual(self.dataset.dataset_name, "tcga_multimodal")
        self.assertEqual(sorted(self.dataset.unique_patient_ids), [P1, P2, P3, P4, P5])

    def test_prepare_metadata_writes_index_files(self) -> None:
        for table in ("clinical", "expression", "histology", "report"):
            path = self.root / f"tcga_multimodal_{table}-pyhealth.csv"
            self.assertTrue(path.exists(), path)
        index = pd.read_csv(self.root / "tcga_multimodal_histology-pyhealth.csv")
        self.assertEqual(len(index), 4)  # every slide file is indexed

    def test_events_per_table(self) -> None:
        p1, p3 = self.dataset.get_patient(P1), self.dataset.get_patient(P3)
        self.assertEqual(len(p1.get_events(event_type="clinical")), 1)
        self.assertEqual(len(p1.get_events(event_type="histology")), 2)
        self.assertEqual(len(p3.get_events(event_type="histology")), 0)
        self.assertEqual(len(p3.get_events(event_type="expression")), 1)

    def test_clinical_columns_preserved(self) -> None:
        record = self.dataset.get_patient(P1).get_events(event_type="clinical")[0]
        self.assertEqual(record.project, "TCGA-BRCA")
        self.assertEqual(float(record.days_to_death), 340.0)
        for column in ("sex", "age", "race", "ethnicity", "vital_status", "fold"):
            self.assertIn(column, record)

    def test_embedding_paths_point_to_files(self) -> None:
        event = self.dataset.get_patient(P2).get_events(event_type="report")[0]
        self.assertEqual(event.file_id, "report")
        np.testing.assert_allclose(np.load(event.embedding_path), [0.5, -0.5])

    def test_fold_assignments(self) -> None:
        folds = self.dataset.get_fold_assignments()
        self.assertEqual(folds, {P1: 0, P2: 1, P3: 0, P5: 1})  # P4 has no fold

    def test_no_folds_raises(self) -> None:
        # Same path as a root without split_cases.csv: no patient has a fold.
        with mock.patch.object(self.dataset, "_clinical_values", return_value={}):
            with self.assertRaisesRegex(ValueError, "split_cases.csv"):
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
        self.assertIsInstance(self.dataset.default_task, MultimodalCancerSurvival)

    # Dataset: validation

    def test_missing_clinical_raises(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
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
                TCGAMultimodalDataset.prepare_metadata(tmp, tables=["report"])
            # A file in the wrong place doesn't count: still an error.
            (Path(tmp) / "embeddings" / "report").mkdir(parents=True)
            np.save(Path(tmp) / "embeddings" / "report" / "stray.npy", np.zeros(2))
            with self.assertRaisesRegex(ValueError, "No <patient_id>"):
                TCGAMultimodalDataset.prepare_metadata(tmp, tables=["report"])
            # Nothing is written on error, so fixing the folder is enough.
            index = Path(tmp) / "tcga_multimodal_report-pyhealth.csv"
            self.assertFalse(index.exists())

    def test_tables_subset_needs_only_those_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(self.root / "clinical.csv", tmp)
            shutil.copytree(
                self.root / "embeddings" / "expression",
                Path(tmp) / "embeddings" / "expression",
            )
            TCGAMultimodalDataset.prepare_metadata(tmp, tables=["expression"])
            written = sorted(p.name for p in Path(tmp).glob("*-pyhealth.csv"))
            self.assertEqual(
                written,
                [
                    "tcga_multimodal_clinical-pyhealth.csv",
                    "tcga_multimodal_expression-pyhealth.csv",
                ],
            )

    # Task: labels

    def test_complete_case_cohort(self) -> None:
        self.assertEqual(sorted(self.samples(self.task)), [P1, P2, P4])

    def test_survival_labels(self) -> None:
        samples = self.samples(self.task)
        self.assertEqual(samples[P1]["duration"], [340.0])
        self.assertEqual(samples[P1]["event"], [1.0])
        self.assertEqual(samples[P2]["duration"], [0.0])  # kept as in the paper
        self.assertEqual(samples[P2]["event"], [0.0])

    def test_duration_unit_and_min_duration(self) -> None:
        task = MultimodalCancerSurvival(duration_unit="years", min_duration=0.01)
        samples = self.samples(task)
        self.assertAlmostEqual(samples[P4]["duration"][0], 2.0)
        self.assertEqual(samples[P2]["duration"], [0.01])

    def test_invalid_vital_status_excluded_and_logged(self) -> None:
        task = MultimodalCancerSurvival(modalities=["demographics"])
        logger_name = "pyhealth.tasks.multimodal_cancer_survival"
        with self.assertLogs(logger_name, "INFO") as log:
            self.assertEqual(task(self.dataset.get_patient(P5)), [])
        self.assertIn("survival outcome", log.output[0])

    # Task: features

    def test_demographics_and_cancer_type(self) -> None:
        sample = self.samples(self.task)[P4]
        self.assertEqual(
            sample["demographics"],
            [
                "sex:male",
                "age:(80, 100]",
                "race:not reported",
                "ethnicity:not hispanic or latino",
            ],
        )
        self.assertEqual(sample["cancer_type"], ["TCGA-BRCA"])

    def test_aggregation_mean_and_first(self) -> None:
        mean = self.samples(self.task)[P1]["histology"]
        first = self.samples(MultimodalCancerSurvival(aggregation="first"))[P1]
        np.testing.assert_allclose(mean, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(first["histology"], [0.0, 2.0, 4.0])

    def test_fewer_modalities_keep_more_patients(self) -> None:
        task = MultimodalCancerSurvival(modalities=["cancer_type", "expression"])
        samples = self.samples(task)
        self.assertEqual(sorted(samples), [P1, P2, P3, P4])
        self.assertEqual(
            set(samples[P3]),
            {"patient_id", "cancer_type", "expression", "duration", "event"},
        )

    def test_invalid_arguments_raise(self) -> None:
        for kwargs in (
            {"modalities": ["genomics"]},
            {"modalities": []},
            {"aggregation": "max"},
            {"duration_unit": "months"},
            {"min_duration": -1.0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MultimodalCancerSurvival(**kwargs)

    def test_task_arguments_are_json_serializable(self) -> None:
        task = MultimodalCancerSurvival(modalities=("report", "demographics"))
        json.dumps(vars(task))  # set_task builds its cache key this way
        self.assertEqual(
            task.input_schema, {"demographics": "multi_hot", "report": "tensor"}
        )

    def test_set_task_end_to_end(self) -> None:
        samples = self.dataset.set_task(self.task)
        self.assertEqual(len(samples), 3)
        sample = samples[0]
        self.assertEqual(tuple(sample["histology"].shape), (3,))
        self.assertEqual(tuple(sample["duration"].shape), (1,))
        self.assertEqual(sample["cancer_type"].sum().item(), 1.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
