"""Synthetic tests only: no real TCGA records or gated assets."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_tcga_data.py"
spec = importlib.util.spec_from_file_location("validate_tcga_data", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

IDS = ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003"]


def setup_root(tmp_path):
    root = tmp_path / "tcga_data"
    root.mkdir()
    pd.DataFrame({"case_id": IDS + ["TCGA-BB-0004"], "days_to_death": [100, 200, 300, 400]}).to_csv(root / "clinical.csv", index=False)
    pd.DataFrame({"case_id": IDS, "split": [0, 1, 0]}).to_csv(root / "split_cases.csv", index=False)
    return root


def add_vec(root, modality, pid, name, *, dim=None, value=1.0):
    dim = dim or module.MODALITY_DIMS[modality]
    path = root / "embeddings" / modality / pid / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = np.full(dim, value, dtype=np.float32)
    if modality == "report":
        vec /= np.linalg.norm(vec)
    np.save(path, vec, allow_pickle=False)
    return path


def test_fold_counts_and_clinical_superset(tmp_path):
    root = setup_root(tmp_path)
    clinical, study, folds = module.read_patient_ids(root)
    assert len(clinical) == 4
    assert study == set(IDS)
    assert folds == {"0": 2, "1": 1}


def test_multimodal_coverage_and_intersection(tmp_path):
    root = setup_root(tmp_path)
    for pid in IDS:
        add_vec(root, "report", pid, "report.txt")
    add_vec(root, "histology", IDS[0], "slide_a.h5")
    add_vec(root, "histology", IDS[0], "slide_b.h5")
    add_vec(root, "histology", IDS[1], "slide_c.h5")
    add_vec(root, "expression", IDS[0], "rnaseq.h5")
    result = module.audit_data(root, ["report", "histology", "expression"], required_complete=["report"])
    assert result["passed"]
    assert result["modalities"]["report"]["study_patients_covered"] == 3
    assert result["modalities"]["histology"]["files"] == 3
    assert result["modalities"]["histology"]["study_patients_covered"] == 2
    assert result["modalities"]["expression"]["study_patients_covered"] == 1
    assert result["overlap_study_patients"]["expression+histology+report"] == 1
    assert result["overlap_study_patients"]["histology+report"] == 2


def test_absent_expression_is_reported_not_assumed_complete(tmp_path):
    root = setup_root(tmp_path)
    result = module.audit_data(root, ["expression"])
    assert result["passed"]
    assert result["modalities"]["expression"]["study_patients_missing"] == 3
    assert result["modalities"]["expression"]["folder_exists"] is False


def test_required_full_coverage_detects_missing_patients(tmp_path):
    root = setup_root(tmp_path)
    add_vec(root, "histology", IDS[0], "slide.h5")
    result = module.audit_data(root, ["histology"], required_complete=["histology"])
    assert not result["passed"]
    assert result["modalities"]["histology"]["study_patients_missing"] == 2


def test_report_norm_and_dtype_are_checked(tmp_path):
    root = setup_root(tmp_path)
    p = add_vec(root, "report", IDS[0], "report.txt")
    np.save(p, np.ones(4096, dtype=np.float32), allow_pickle=False)
    result = module.audit_data(root, ["report"])
    assert not result["passed"]
    assert result["modalities"]["report"]["invalid_vectors"] == 1
    assert result["modalities"]["report"]["study_patients_covered"] == 0
    np.save(p, np.ones(4096, dtype=np.float64), allow_pickle=False)
    assert not module.audit_data(root, ["report"])["passed"]


def test_rejects_invalid_histology_vector_and_patient_id(tmp_path):
    root = setup_root(tmp_path)
    add_vec(root, "histology", IDS[0], "slide.h5", dim=128)
    add_vec(root, "histology", "TCGA-ZZ-9999", "slide.h5")
    result = module.audit_data(root, ["histology"])
    assert not result["passed"]
    assert result["modalities"]["histology"]["invalid_vectors"] == 1
    assert result["modalities"]["histology"]["patients_outside_study"] == 1


def test_invalid_vector_in_one_slide_invalidates_patient(tmp_path):
    root = setup_root(tmp_path)
    add_vec(root, "histology", IDS[0], "good.h5")
    bad = add_vec(root, "histology", IDS[0], "bad.h5")
    np.save(bad, np.array([float("nan")] * 1536, dtype=np.float32))
    r = module.audit_data(root, ["histology"])
    assert r["modalities"]["histology"]["study_patients_covered"] == 0
    assert not r["passed"]


def test_report_manifest_checks_exact_paths(tmp_path):
    root = setup_root(tmp_path)
    rows = []
    for pid in IDS:
        p = add_vec(root, "report", pid, "pathology.txt")
        rows.append({"case_id": pid, "report_embedding_relpath": p.relative_to(root).as_posix()})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    result = module.audit_data(root, ["report"], report_manifest=manifest, required_complete=["report"])
    assert result["passed"]
    rows[0]["report_embedding_relpath"] = "embeddings/report/TCGA-AA-0001/wrong.npy"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    result = module.audit_data(root, ["report"], report_manifest=manifest)
    assert not result["passed"]
    assert "manifest differs" in result["problems"][0]


def test_manifest_rejects_traversal(tmp_path):
    root = setup_root(tmp_path)
    path = tmp_path / "m.csv"
    pd.DataFrame({"case_id": IDS, "report_embedding_relpath": [
        "embeddings/report/TCGA-AA-0001/../../bad.npy",
        "embeddings/report/TCGA-AA-0002/r.npy",
        "embeddings/report/TCGA-AA-0003/r.npy",
    ]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unsafe"):
        module.audit_data(root, ["report"], report_manifest=path)


def test_bad_splits_rejected(tmp_path):
    root = setup_root(tmp_path)
    pd.DataFrame({"case_id": [IDS[0], IDS[0]], "split": [0, 1]}).to_csv(root / "split_cases.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        module.audit_data(root, ["report"])


def test_bad_nesting_rejected(tmp_path):
    root = setup_root(tmp_path)
    nested = root / "embeddings" / "expression" / IDS[0] / "wrong" / "vector.npy"
    nested.parent.mkdir(parents=True)
    np.save(nested, np.ones(256, dtype=np.float32))
    r = module.audit_data(root, ["expression"])
    assert not r["passed"]
    assert r["modalities"]["expression"]["invalid_paths"] == 1


def test_cli_writes_aggregate_only_json(tmp_path):
    root = setup_root(tmp_path)
    for pid in IDS:
        add_vec(root, "report", pid, "report.txt")
    report = tmp_path / "summary.json"
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--modalities", "report", "expression",
         "--require-complete", "report", "--json-out", str(report)],
        capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    assert "MULTIMODAL DATA VALIDATION PASSED!" in process.stdout
    result = json.loads(report.read_text())
    assert result["passed"]
    assert result["modalities"]["expression"]["study_patients_covered"] == 0
    assert IDS[0] not in report.read_text()  # Aggregate counts only; no patient identifiers.
