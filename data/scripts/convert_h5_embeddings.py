#!/usr/bin/env python3
"""Convert pre-pooled TCGA HDF5 vectors into patient-level PyHealth .npy files.

Supported input hierarchy (the authors' post-embedding format):

    <TCGA-case-id>/<source-filename>  -> 1-D numeric HDF5 dataset

Output (same source filename plus '.npy'):

    <out-root>/embeddings/<modality>/<TCGA-case-id>/<source-filename>.npy

This tool does NOT pool histology patches, run foundation models, infer case IDs
from slide barcodes, or normalize vectors. Use aggregate_uni2_histology.py for
raw UNI2-h tile-feature archives before using PyHealth.

Examples:
  python data/scripts/convert_h5_embeddings.py --h5 /private/expr.h5 \\
      --modality expression --out-root /private/tcga_data --study-only

  python data/scripts/convert_h5_embeddings.py --h5 /private/expr.h5 \\
      --modality expression --out-root /private/tcga_data --study-only --check-only
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np

DIMS = {"report": 4096, "histology": 1536, "expression": 256}
CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")
FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def read_ids(path: Path, *, required: bool) -> set[str] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required patient table not found: {path}")
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "case_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing case_id column: {path}")
        ids = [row["case_id"].strip() for row in reader]
    if not ids or len(ids) != len(set(ids)) or any(not CASE_RE.fullmatch(x) for x in ids):
        raise ValueError(f"Empty, invalid or duplicate case IDs: {path}")
    return set(ids)


def _vector(ds: h5py.Dataset, *, expected_dim: int | None, source: str) -> np.ndarray:
    if ds.ndim != 1 or ds.size == 0:
        raise ValueError(f"{source}: expected nonempty 1-D pooled embedding, got {ds.shape}")
    if ds.dtype.kind not in "fiu":
        raise ValueError(f"{source}: expected numeric dataset, got {ds.dtype}")
    if expected_dim is not None and ds.size != expected_dim:
        raise ValueError(f"{source}: expected {expected_dim} values, found {ds.size}")
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.asarray(ds[()], dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{source}: NaN, infinity or float32 overflow")
    return arr


def _existing_matches(path: Path, expected: np.ndarray) -> bool:
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError, TypeError):
        return False
    return arr.shape == expected.shape and arr.dtype == np.float32 and bool(np.array_equal(arr, expected))


def _atomic_save(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".partial_", suffix=".npy",
                                         dir=str(path.parent), delete=False) as out:
            temp = out.name
            np.save(out, vector, allow_pickle=False)
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)


def convert(
    h5_path: Path,
    modality: str,
    out_root: Path,
    *,
    expected_dim: int | None = None,
    study_only: bool = False,
    strict_clinical: bool = True,
    check_only: bool = False,
    overwrite: bool = False,
) -> dict:
    """Convert or audit grouped vectors; never silently accept different existing values."""
    if modality not in DIMS:
        raise ValueError(f"modality must be one of {sorted(DIMS)}")
    h5_path, out_root = Path(h5_path), Path(out_root)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if expected_dim is None:
        expected_dim = DIMS[modality]
    if expected_dim <= 0:
        raise ValueError("expected_dim must be positive")
    if check_only and overwrite:
        raise ValueError("--overwrite cannot be combined with --check-only")

    clinical_ids = read_ids(out_root / "clinical.csv", required=strict_clinical)
    study_ids = read_ids(out_root / "split_cases.csv", required=study_only)
    if study_ids is not None and clinical_ids is not None and not study_ids <= clinical_ids:
        raise ValueError("split_cases.csv includes case IDs absent from clinical.csv")

    counts = {"source_patients": 0, "source_vectors": 0, "outside_study": 0,
              "matched_patients": 0, "matched_vectors": 0, "written": 0,
              "already_present": 0, "missing": 0, "mismatched": 0}
    output_root = out_root / "embeddings" / modality

    # In check-only mode we aggregate all output problems rather than exit on first.
    with h5py.File(h5_path, "r") as h5:
        if not h5.keys():
            raise ValueError(f"Empty input HDF5 file: {h5_path}")
        for case_id in sorted(h5.keys()):
            if not CASE_RE.fullmatch(case_id):
                raise ValueError(f"Unsafe/invalid TCGA case ID in HDF5: {case_id!r}")
            group = h5[case_id]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"{case_id}: expected an HDF5 group")
            counts["source_patients"] += 1
            if clinical_ids is not None and case_id not in clinical_ids:
                raise ValueError(f"{case_id}: absent from clinical.csv")
            if study_only and case_id not in study_ids:
                counts["outside_study"] += 1
                continue
            if not group.keys():
                raise ValueError(f"{case_id}: empty HDF5 group")
            counts["matched_patients"] += 1
            for file_id in sorted(group.keys()):
                if not FILE_RE.fullmatch(file_id) or file_id in (".", ".."):
                    raise ValueError(f"{case_id}: unsafe source filename {file_id!r}")
                ds = group[file_id]
                if not isinstance(ds, h5py.Dataset):
                    raise ValueError(f"{case_id}/{file_id}: nested group is not a vector")
                counts["source_vectors"] += 1
                vector = _vector(ds, expected_dim=expected_dim, source=f"{case_id}/{file_id}")
                counts["matched_vectors"] += 1
                target = output_root / case_id / (file_id + ".npy")
                if target.parent.is_symlink() or target.is_symlink():
                    raise ValueError(f"Symlink encountered at {target}; refusing to write")
                if target.is_file():
                    if _existing_matches(target, vector):
                        counts["already_present"] += 1
                        continue
                    if check_only:
                        counts["mismatched"] += 1
                        continue
                    if not overwrite:
                        raise ValueError(
                            f"Existing vector differs: {target}. Refusing to mix embedding "
                            "provenance; inspect the source or use --overwrite deliberately."
                        )
                elif check_only:
                    counts["missing"] += 1
                    continue
                _atomic_save(target, vector)
                if not _existing_matches(target, vector):
                    raise OSError(f"Write verification failed: {target}")
                counts["written"] += 1

    counts["embedding_dim"] = expected_dim
    counts["modality"] = modality
    counts["passed"] = counts["missing"] == 0 and counts["mismatched"] == 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="Pre-pooled grouped HDF5 file")
    parser.add_argument("--modality", choices=sorted(DIMS), required=True)
    parser.add_argument("--out-root", type=Path, required=True, help="PyHealth tcga_data root")
    parser.add_argument("--expected-dim", type=int, help="Override the modality's expected dimension")
    parser.add_argument("--study-only", action="store_true",
                        help="Keep only patients present in split_cases.csv")
    parser.add_argument("--allow-missing-clinical", action="store_true",
                        help="Disable clinical.csv ID check (not recommended for study use)")
    parser.add_argument("--check-only", action="store_true", help="Audit existing .npy outputs, read-only")
    parser.add_argument("--overwrite", action="store_true",
                        help="Intentionally replace differing existing vectors")
    args = parser.parse_args()
    result = convert(args.h5, args.modality, args.out_root,
                     expected_dim=args.expected_dim, study_only=args.study_only,
                     strict_clinical=not args.allow_missing_clinical,
                     check_only=args.check_only, overwrite=args.overwrite)
    for key, val in result.items():
        print(f"{key}: {val}")
    if not result["passed"]:
        raise SystemExit("HDF5 CONVERSION AUDIT FAILED!")
    print("HDF5 CONVERSION AUDIT PASSED!" if args.check_only else "HDF5 CONVERSION COMPLETE!")
    print("After adding vectors, refresh PyHealth generated modality indexes and task caches.")


if __name__ == "__main__":
    main()
