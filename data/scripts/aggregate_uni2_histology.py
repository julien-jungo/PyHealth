#!/usr/bin/env python3
"""Aggregate precomputed UNI2-h patch features into PyHealth slide embeddings.

CPU-only; does not download gated files or load the UNI2-h model. Supports
MahmoodLab TCGA .tar.gz archives without extracting every slide to disk and
unpacked directories of slide .h5/.hdf5 files. Each matching slide produces:
    OUT_ROOT/embeddings/histology/<TCGA-case-id>/<slide-filename>.npy

Uses the *mean* of the patch embeddings, matching the authors' aggregation
choice (but float64 accumulation can cause tiny roundoff differences).

Examples:
  python data/scripts/aggregate_uni2_histology.py \\
    --archive /private/TCGA-DLBC.tar.gz --splits /private/split_cases.csv \\
    --out-root /private/tcga_data --max-new 5

  python data/scripts/aggregate_uni2_histology.py \\
    --archive /private/TCGA-DLBC.tar.gz --splits /private/split_cases.csv \\
    --out-root /private/tcga_data --check-only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

import h5py
import numpy as np
import pandas as pd

DIM = 1536
CASE_PATTERN = re.compile(r"^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}$")
SLIDE_PATTERN = re.compile(r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})[-.]")
EXTS = (".h5", ".hdf5")


def load_study_ids(splits_csv: Path) -> set[str]:
    splits = pd.read_csv(splits_csv)
    if "case_id" not in splits:
        raise ValueError("Split CSV must have a case_id column")
    ids = splits["case_id"].astype(str)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Null/duplicate study patient IDs")
    study_ids = set(ids)
    if not study_ids or not all(CASE_PATTERN.fullmatch(s) for s in study_ids):
        raise ValueError("Invalid or empty TCGA study patient IDs")
    return study_ids


def slide_name_and_patient(member_name: str) -> tuple[str, str]:
    """Parse a real TCGA slide filename, rejecting traversal/ambiguous names."""
    p = PurePosixPath(member_name)
    if p.is_absolute() or ".." in p.parts or "\\" in member_name:
        raise ValueError(f"Unsafe slide path: {member_name}")
    name = p.name
    if not name.lower().endswith(EXTS):
        raise ValueError(f"Not an HDF5 slide: {member_name}")
    match = SLIDE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Slide filename has no TCGA patient prefix: {name}")
    return name, match.group(1)


def output_path(out_root: Path, patient_id: str, slide_name: str) -> Path:
    if not CASE_PATTERN.fullmatch(patient_id):
        raise ValueError(f"Invalid patient ID: {patient_id}")
    parsed_name, parsed_id = slide_name_and_patient(slide_name)
    if parsed_name != slide_name or parsed_id != patient_id:
        raise ValueError("Slide name does not match patient ID")
    return out_root / "embeddings" / "histology" / patient_id / (slide_name + ".npy")


def valid_vector(path: Path) -> bool:
    try:
        array = np.load(path, allow_pickle=False)
        return (
            array.shape == (DIM,)
            and array.dtype == np.float32
            and bool(np.isfinite(array).all())
        )
    except (OSError, ValueError, EOFError, TypeError):
        return False


def aggregate_h5(h5_file: Path, chunk_patches: int = 4096) -> np.ndarray:
    """Chunked float64 accumulation of mean UNI2-h patch features."""
    if chunk_patches <= 0:
        raise ValueError("chunk_patches must be positive")
    with h5py.File(h5_file, "r") as h5:
        if "features" not in h5:
            raise ValueError("Slide HDF5 is missing the 'features' dataset")
        ds = h5["features"]
        if ds.ndim == 3:
            if ds.shape[0] != 1 or ds.shape[2] != DIM:
                raise ValueError(f"Unexpected 3D feature shape {ds.shape}")
            count = ds.shape[1]
            select = lambda start, end: ds[0, start:end, :]
        elif ds.ndim == 2:
            if ds.shape[1] != DIM:
                raise ValueError(f"Unexpected 2D feature shape {ds.shape}")
            count = ds.shape[0]
            select = lambda start, end: ds[start:end, :]
        else:
            raise ValueError(f"Unexpected feature dimensions: {ds.shape}")
        if count == 0:
            raise ValueError("Cannot aggregate a slide with zero patches")
        total = np.zeros(DIM, dtype=np.float64)
        for start in range(0, count, chunk_patches):
            block = np.asarray(select(start, min(count, start + chunk_patches)))
            if not np.isfinite(block).all():
                raise ValueError("Slide has non-finite patch features")
            total += block.sum(axis=0, dtype=np.float64)
    vector = np.asarray(total / count, dtype=np.float32)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise ValueError("Invalid aggregated slide embedding")
    # No L2-normalization: authors' UNI2-h script simply mean-pools patches.
    return vector


def save_atomic(target: Path, vector: np.ndarray) -> None:
    if vector.shape != (DIM,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError("Invalid vector; refusing to save")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".part-", suffix=".npy", dir=target.parent, delete=False
        ) as tmp:
            temp_name = Path(tmp.name)
            np.save(tmp, vector, allow_pickle=False)
        if not valid_vector(temp_name):
            raise ValueError("Temporary vector validation failed")
        os.replace(temp_name, target)
    finally:
        if temp_name is not None:
            temp_name.unlink(missing_ok=True)


def process_h5_file(h5_path: Path, target: Path, chunk_patches: int) -> None:
    vector = aggregate_h5(h5_path, chunk_patches)
    save_atomic(target, vector)


def _safe_member(member: tarfile.TarInfo) -> bool:
    """Reject unsafe archive paths even though we never call extractall."""
    p = PurePosixPath(member.name)
    if p.is_absolute() or ".." in p.parts or "\\" in member.name:
        raise ValueError(f"Unsafe archive member path: {member.name}")
    if member.issym() or member.islnk():
        raise ValueError(f"Archive contains a link: {member.name}")
    return member.isfile() and member.name.lower().endswith(EXTS)


def process_archive(
    archive: Path,
    study_ids: set[str],
    out_root: Path,
    *,
    max_new: int | None = None,
    check_only: bool = False,
    chunk_patches: int = 4096,
) -> dict[str, int]:
    """Stream one compressed TCGA archive; safely save only split-assigned slides."""
    if max_new is not None and max_new < 0:
        raise ValueError("max_new must be non-negative")
    counts = {"matched": 0, "new": 0, "skipped": 0, "missing_or_invalid": 0, "outside_study": 0}
    seen: set[Path] = set()
    # Streaming gzip avoids repeatedly decompressing a huge tar archive.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not _safe_member(member):
                continue
            name, patient_id = slide_name_and_patient(member.name)
            if patient_id not in study_ids:
                counts["outside_study"] += 1
                continue
            target = output_path(out_root, patient_id, name)
            if target in seen:
                raise ValueError(f"Duplicate target slide filename in archive: {target}")
            seen.add(target)
            counts["matched"] += 1
            if valid_vector(target):
                counts["skipped"] += 1
                continue
            if check_only:
                counts["missing_or_invalid"] += 1
                continue
            if max_new is not None and counts["new"] >= max_new:
                # Count other missing files but don't open more HDF5 data.
                counts["missing_or_invalid"] += 1
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive slide: {member.name}")
            # HDF5 needs random access. Extract only this *one* slide to a
            # temporary file; clean it up before advancing to the next.
            with tempfile.TemporaryDirectory(prefix="uni2-slide-") as workdir:
                tmp_h5 = Path(workdir) / "slide.h5"
                with tmp_h5.open("wb") as f:
                    shutil.copyfileobj(stream, f, length=1024 * 1024)
                process_h5_file(tmp_h5, target, chunk_patches)
            counts["new"] += 1
            if counts["new"] <= 5 or counts["new"] % 10 == 0:
                print(f"Saved {counts['new']} new slide embeddings (patient {patient_id})", flush=True)
    if check_only and counts["missing_or_invalid"]:
        raise ValueError(f"Histology audit found {counts['missing_or_invalid']} missing or invalid files")
    return counts


def process_directory(
    source_dir: Path,
    study_ids: set[str],
    out_root: Path,
    *,
    max_new: int | None = None,
    check_only: bool = False,
    chunk_patches: int = 4096,
) -> dict[str, int]:
    if max_new is not None and max_new < 0:
        raise ValueError("max_new must be non-negative")
    counts = {"matched": 0, "new": 0, "skipped": 0, "missing_or_invalid": 0, "outside_study": 0}
    seen: set[Path] = set()
    slides = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in EXTS and p.is_file())
    for path in slides:
        name, patient_id = slide_name_and_patient(path.name)
        if patient_id not in study_ids:
            counts["outside_study"] += 1
            continue
        target = output_path(out_root, patient_id, name)
        if target in seen:
            raise ValueError(f"Duplicate slide filename in source dir: {name}")
        seen.add(target)
        counts["matched"] += 1
        if valid_vector(target):
            counts["skipped"] += 1
        elif check_only or (max_new is not None and counts["new"] >= max_new):
            counts["missing_or_invalid"] += 1
        else:
            process_h5_file(path, target, chunk_patches)
            counts["new"] += 1
            if counts["new"] <= 5 or counts["new"] % 10 == 0:
                print(f"Saved {counts['new']} new slide embeddings (patient {patient_id})", flush=True)
    if check_only and counts["missing_or_invalid"]:
        raise ValueError(f"Histology audit found {counts['missing_or_invalid']} missing or invalid files")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Gated MahmoodLab TCGA-<project>.tar.gz archive")
    source.add_argument("--source-dir", type=Path, help="Directory of extracted slide .h5 files")
    parser.add_argument("--splits", type=Path, required=True, help="Study split_cases.csv")
    parser.add_argument("--out-root", type=Path, required=True, help="PyHealth TCGA data root")
    parser.add_argument("--max-new", type=int, default=None, help="Generate at most N new slide vectors")
    parser.add_argument("--check-only", action="store_true", help="Validate saved vectors for this archive/dir")
    parser.add_argument("--chunk-patches", type=int, default=4096)
    args = parser.parse_args()
    ids = load_study_ids(args.splits)
    if args.archive:
        result = process_archive(args.archive, ids, args.out_root,
                                 max_new=args.max_new, check_only=args.check_only,
                                 chunk_patches=args.chunk_patches)
    else:
        result = process_directory(args.source_dir, ids, args.out_root,
                                   max_new=args.max_new, check_only=args.check_only,
                                   chunk_patches=args.chunk_patches)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("HISTOLOGY ARCHIVE AUDIT PASSED!" if args.check_only else "HISTOLOGY AGGREGATION COMPLETE!")


if __name__ == "__main__":
    main()
