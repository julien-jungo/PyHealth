"""Synthetic-only tests: no downloads, gates, UNI2 weights or TCGA data needed."""
from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_uni2_histology.py"
spec = importlib.util.spec_from_file_location("aggregate_uni2_histology", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_h5(path: Path, value: float = 2., *, rank: int = 3, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Nonconstant patch values to ensure pooling is actually tested
    patches = np.stack([np.full(1536, value + i, dtype=np.float32) for i in range(n)])
    with h5py.File(path, "w") as h5:
        h5.create_dataset("features", data=patches[None, ...] if rank == 3 else patches)
        h5.create_dataset("coords", data=np.zeros((1, n, 2), dtype=np.int32))


def make_tar(path: Path, files: list[Path]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f"subfolder/{f.name}")


def test_mean_pooling_both_shapes(tmp_path):
    for rank in (2, 3):
        source = tmp_path / f"rank{rank}.h5"
        make_h5(source, value=2, rank=rank, n=3)
        vector = module.aggregate_h5(source, chunk_patches=2)
        assert vector.shape == (1536,)
        assert vector.dtype == np.float32
        np.testing.assert_allclose(vector, np.full(1536, 3, dtype=np.float32))
        assert not np.isclose(np.linalg.norm(vector), 1.0)  # no L2 norm


def test_archive_ignores_nonstudy_and_resumes(tmp_path):
    study_slide = tmp_path / "TCGA-AA-0001-01Z-00-DX1.aaa.h5"
    other_slide = tmp_path / "TCGA-BB-0002-01Z-00-DX1.bbb.h5"
    make_h5(study_slide, value=5)
    make_h5(other_slide, value=7)
    archive = tmp_path / "features.tar.gz"
    make_tar(archive, [study_slide, other_slide])
    ids = {"TCGA-AA-0001"}
    root = tmp_path / "out"
    first = module.process_archive(archive, ids, root)
    assert first == {"matched": 1, "new": 1, "skipped": 0, "missing_or_invalid": 0, "outside_study": 1}
    target = module.output_path(root, "TCGA-AA-0001", study_slide.name)
    assert module.valid_vector(target)
    np.testing.assert_allclose(np.load(target), np.full(1536, 6, dtype=np.float32))
    second = module.process_archive(archive, ids, root, check_only=True)
    assert second["skipped"] == 1 and second["missing_or_invalid"] == 0
    third = module.process_archive(archive, ids, root)
    assert third["new"] == 0 and third["skipped"] == 1
    target.unlink()
    with pytest.raises(ValueError, match="Histology audit found 1"):
        module.process_archive(archive, ids, root, check_only=True)


def test_directory_aggregation(tmp_path):
    raw = tmp_path / "raw"
    make_h5(raw / "x" / "TCGA-AA-0001-01Z-00-DX1.a.h5", value=1)
    make_h5(raw / "x" / "TCGA-BB-0002-01Z-00-DX1.b.h5", value=9)
    counts = module.process_directory(raw, {"TCGA-AA-0001"}, tmp_path / "out", max_new=1)
    assert counts["matched"] == 1 and counts["new"] == 1
    assert counts["outside_study"] == 1


def test_invalid_feature_data_rejected(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as h5:
        h5.create_dataset("features", data=np.zeros((1, 2, 500), dtype=np.float32))
    with pytest.raises(ValueError, match="Unexpected 3D feature shape"):
        module.aggregate_h5(bad)
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as h5:
        h5.create_dataset("features", shape=(1, 0, 1536), dtype="float32")
    with pytest.raises(ValueError, match="zero patches"):
        module.aggregate_h5(empty)
    nan = tmp_path / "nan.h5"
    make_h5(nan)
    with h5py.File(nan, "r+") as h5:
        h5["features"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module.aggregate_h5(nan)


def test_paths_and_study_ids_rejected(tmp_path):
    assert module.slide_name_and_patient("dir/TCGA-AA-0001-01Z-00-DX1.x.h5") == (
        "TCGA-AA-0001-01Z-00-DX1.x.h5", "TCGA-AA-0001"
    )
    with pytest.raises(ValueError, match="Unsafe slide path"):
        module.slide_name_and_patient("../TCGA-AA-0001-foo.h5")
    with pytest.raises(ValueError, match="does not match"):
        module.output_path(tmp_path, "TCGA-AA-0001", "TCGA-BB-0002-foo.h5")
    csv = tmp_path / "folds.csv"
    pd.DataFrame({"case_id": ["TCGA-AA-0001"]}).to_csv(csv, index=False)
    assert module.load_study_ids(csv) == {"TCGA-AA-0001"}
    pd.DataFrame({"case_id": ["TCGA-AA-0001", "TCGA-AA-0001"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        module.load_study_ids(csv)


def test_atomic_replace_and_invalid_vector(tmp_path):
    p = tmp_path / "test.npy"
    a = np.full(1536, 2.5, dtype=np.float32)
    module.save_atomic(p, a)
    assert module.valid_vector(p)
    module.save_atomic(p, np.full(1536, 3.0, dtype=np.float32))
    assert np.all(np.load(p) == 3.0)
    with pytest.raises(ValueError, match="Invalid vector"):
        module.save_atomic(p, np.zeros(4, dtype=np.float32))


def test_max_new_resumes_without_overwriting(tmp_path):
    slides = [tmp_path / f"TCGA-AA-000{i}-01Z-00-DX1.x.h5" for i in (1, 2, 3)]
    for i, p in enumerate(slides):
        make_h5(p, value=i + 1)
    archive = tmp_path / "pilot.tar.gz"
    make_tar(archive, slides)
    ids = {f"TCGA-AA-000{i}" for i in (1, 2, 3)}
    out = tmp_path / "out"
    one = module.process_archive(archive, ids, out, max_new=1)
    assert one["new"] == 1 and one["missing_or_invalid"] == 2
    another = module.process_archive(archive, ids, out, max_new=2)
    assert another["new"] == 2 and another["skipped"] == 1
    assert module.process_archive(archive, ids, out, check_only=True)["skipped"] == 3


def test_archive_rejects_unsafe_member(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        content = b"not real h5"
        info = tarfile.TarInfo(name="../TCGA-AA-0001-01Z-00-DX1.a.h5")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        module.process_archive(archive, {"TCGA-AA-0001"}, tmp_path / "out")
