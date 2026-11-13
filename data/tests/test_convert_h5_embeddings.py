"""HDF5 -> NPY converter tests using fake TCGA IDs and synthetic numeric vectors."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_h5_embeddings.py"
spec = importlib.util.spec_from_file_location("convert_h5_embeddings", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def setup(tmp_path):
    root = tmp_path / "tcga_data"
    root.mkdir()
    (root / "clinical.csv").write_text("case_id\nTCGA-AA-0001\nTCGA-BB-0002\nTCGA-CC-0003\n")
    (root / "split_cases.csv").write_text("case_id,split\nTCGA-AA-0001,0\nTCGA-BB-0002,1\n")
    source = tmp_path / "expr.h5"
    with h5py.File(source, "w") as h5:
        a = h5.create_group("TCGA-AA-0001")
        a.create_dataset("sample_1.txt", data=np.array([1., 2., 3.], dtype=np.float64))
        a.create_dataset("sample_2.txt", data=np.array([4., 5., 6.], dtype=np.float64))
        b = h5.create_group("TCGA-BB-0002")
        b.create_dataset("sample_3.txt", data=np.array([7., 8., 9.], dtype=np.float32))
        c = h5.create_group("TCGA-CC-0003")
        c.create_dataset("excluded.txt", data=np.array([1., 2., 3.], dtype=np.float32))
    return root, source


def call(source, root, **kw):
    return mod.convert(source, "expression", root, expected_dim=3, **kw)


def test_conversion_preserves_patient_file_ids_and_float32(tmp_path):
    root, source = setup(tmp_path)
    result = call(source, root, study_only=True)
    assert result["written"] == 3 and result["matched_patients"] == 2
    assert result["outside_study"] == 1
    first = root / "embeddings/expression/TCGA-AA-0001/sample_1.txt.npy"
    assert first.is_file() and np.load(first).dtype == np.float32
    np.testing.assert_array_equal(np.load(first), [1, 2, 3])
    assert not (root / "embeddings/expression/TCGA-CC-0003").exists()


def test_resume_verifies_actual_values(tmp_path):
    root, source = setup(tmp_path)
    call(source, root, study_only=True)
    result = call(source, root, study_only=True)
    assert result["written"] == 0 and result["already_present"] == 3
    out = root / "embeddings/expression/TCGA-AA-0001/sample_1.txt.npy"
    np.save(out, np.array([9, 9, 9], dtype=np.float32))
    with pytest.raises(ValueError, match="Existing vector differs"):
        call(source, root, study_only=True)
    # Deliberate overwrite repairs it.
    repaired = call(source, root, study_only=True, overwrite=True)
    assert repaired["written"] == 1 and repaired["already_present"] == 2
    np.testing.assert_array_equal(np.load(out), [1, 2, 3])


def test_check_only_reports_missing_and_mismatch_without_writing(tmp_path):
    root, source = setup(tmp_path)
    result = call(source, root, study_only=True, check_only=True)
    assert result["missing"] == 3 and result["written"] == 0 and not result["passed"]
    call(source, root, study_only=True)
    path = root / "embeddings/expression/TCGA-AA-0001/sample_1.txt.npy"
    np.save(path, np.array([0, 0, 0], dtype=np.float32))
    result = call(source, root, study_only=True, check_only=True)
    assert result["mismatched"] == 1 and not result["passed"]
    np.testing.assert_array_equal(np.load(path), [0, 0, 0])


def test_check_only_succeeds_after_conversion(tmp_path):
    root, source = setup(tmp_path)
    call(source, root, study_only=True)
    assert call(source, root, study_only=True, check_only=True)["passed"]


def test_rejects_patch_matrix_and_does_not_write(tmp_path):
    root, source = setup(tmp_path)
    with h5py.File(source, "a") as h5:
        del h5["TCGA-AA-0001/sample_1.txt"]
        h5["TCGA-AA-0001"].create_dataset("sample_1.txt", data=np.ones((2, 3)))
    with pytest.raises(ValueError, match="1-D pooled"):
        call(source, root, study_only=True)


def test_rejects_nonfinite_and_wrong_dimension(tmp_path):
    root, source = setup(tmp_path)
    with h5py.File(source, "a") as h5:
        del h5["TCGA-AA-0001/sample_1.txt"]
        h5["TCGA-AA-0001"].create_dataset("sample_1.txt", data=[1., float("nan"), 2.])
    with pytest.raises(ValueError, match="NaN"):
        call(source, root, study_only=True)
    with pytest.raises(ValueError, match="expected 256"):
        mod.convert(source, "expression", root, study_only=True)


def test_rejects_missing_or_unmatched_clinical(tmp_path):
    root, source = setup(tmp_path)
    (root / "clinical.csv").unlink()
    with pytest.raises(FileNotFoundError, match="clinical.csv"):
        call(source, root, study_only=True)
    root.joinpath("clinical.csv").write_text("case_id\nTCGA-AA-0001\n")
    with pytest.raises(ValueError, match="absent from clinical.csv"):
        call(source, root, study_only=True)


def test_rejects_bad_filename_or_case_id(tmp_path):
    root, source = setup(tmp_path)
    with h5py.File(source, "a") as h5:
        h5["TCGA-AA-0001"].create_dataset("bad name", data=[1., 2., 3.])
    with pytest.raises(ValueError, match="unsafe source filename"):
        call(source, root, study_only=True)
    with h5py.File(source, "a") as h5:
        del h5["TCGA-AA-0001/bad name"]
        h5.create_group("not-a-case").create_dataset("a.txt", data=[1., 2., 3.])
    with pytest.raises(ValueError, match="Unsafe/invalid"):
        call(source, root, study_only=True)


def test_rejects_nested_group(tmp_path):
    root, source = setup(tmp_path)
    with h5py.File(source, "a") as h5:
        h5["TCGA-AA-0001"].create_group("nested")
    with pytest.raises(ValueError, match="nested group"):
        call(source, root, study_only=True)


def test_cli_synthetic_smoke(tmp_path):
    root, source = setup(tmp_path)
    command = [sys.executable, str(SCRIPT), "--h5", str(source), "--modality", "expression",
               "--out-root", str(root), "--expected-dim", "3", "--study-only"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "HDF5 CONVERSION COMPLETE!" in result.stdout
    check = subprocess.run(command + ["--check-only"], check=True, capture_output=True, text=True)
    assert "HDF5 CONVERSION AUDIT PASSED!" in check.stdout


def test_check_only_and_overwrite_mutually_exclusive(tmp_path):
    root, source = setup(tmp_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        call(source, root, check_only=True, overwrite=True)
