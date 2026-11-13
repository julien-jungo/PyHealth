"""Synthetic, CPU-only tests for the report preparation script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "embed_reports_biomistral.py"
spec = importlib.util.spec_from_file_location("embed_reports_biomistral", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def synthetic(tmp_path: Path):
    reports = tmp_path / "summarized_reports.csv"
    splits = tmp_path / "split_cases.csv"
    pd.DataFrame(
        {
            "patient_filename": ["TCGA-AA-0001.report.txt", "TCGA-AA-0002.report.txt"],
            "text": ["A short pathology report", "Another short pathology report"],
        }
    ).to_csv(reports, index=False)
    pd.DataFrame({"case_id": ["TCGA-AA-0001", "TCGA-AA-0002"], "split": [0, 1]}).to_csv(
        splits, index=False
    )
    return reports, splits


def test_exact_paths_and_full_audit(tmp_path):
    reports, splits = synthetic(tmp_path)
    study = module.load_study(reports, splits)
    out = tmp_path / "out"
    for row in study.itertuples(index=False):
        vec = np.ones(4096, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        module.write_atomic(module.output_path(out, row.case_id, row.patient_filename), vec)
    assert module.audit(study, out) == (2, 0, 0)


def test_missing_and_unexpected_detected(tmp_path):
    reports, splits = synthetic(tmp_path)
    study = module.load_study(reports, splits)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Embedding audit failed"):
        module.audit(study, out)
    p = out / "TCGA-AA-0001" / "bad.npy"
    p.parent.mkdir(parents=True)
    np.save(p, np.ones(4096, dtype=np.float32))
    with pytest.raises(ValueError, match="Embedding audit failed"):
        module.audit(study, out)


def test_bad_vectors_rejected(tmp_path):
    vec = np.zeros(4096, dtype=np.float32)
    assert not module.valid_vector(vec)
    assert not module.valid_vector(np.ones(4096, dtype=np.float64))
    vec[0] = 1
    assert module.valid_vector(vec)
    vec[1] = np.nan
    assert not module.valid_vector(vec)


def test_missing_patient_and_duplicate_rejected(tmp_path):
    reports, splits = synthetic(tmp_path)
    pd.DataFrame({"case_id": ["TCGA-AA-0001", "TCGA-AA-0003"]}).to_csv(
        splits, index=False
    )
    with pytest.raises(ValueError, match="no pathology report"):
        module.load_study(reports, splits)
    pd.DataFrame({"case_id": ["TCGA-AA-0001", "TCGA-AA-0002"]}).to_csv(
        splits, index=False
    )
    original = pd.read_csv(reports)
    pd.concat([original, original.iloc[[0]]]).to_csv(reports, index=False)
    with pytest.raises(ValueError, match="Duplicate report filenames"):
        module.load_study(reports, splits)


def test_existing_config_must_match(tmp_path):
    reports, splits = synthetic(tmp_path)
    out = tmp_path / "out"
    config = module.config_payload(reports, splits, 2048)
    module.check_or_write_config(out, config)
    module.check_or_write_config(out, config)
    assert json.loads((out / "embedding_config.json").read_text())["pooling"] == module.POOLING
    with pytest.raises(ValueError, match="Configuration mismatch"):
        module.check_or_write_config(out, {**config, "max_allowed_tokens": 1024})


def test_unsafe_names_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsafe report filename"):
        module.output_path(tmp_path, "TCGA-AA-0001", "../escape")
    with pytest.raises(ValueError, match="Invalid case ID"):
        module.output_path(tmp_path, "../../escape", "good.txt")
