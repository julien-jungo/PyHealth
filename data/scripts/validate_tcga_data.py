#!/usr/bin/env python3
"""Audit patient-aligned TCGA clinical, fold and PyHealth embedding inputs.

CPU-only and READ-ONLY: never modifies data or loads a foundation model.
Summarizes report, histology and/or expression modality coverage, numerical
validity and study-patient overlap. Missing modalities are allowed unless
explicitly listed via --require-complete. A requested modality may have no
folder yet; this is reported as zero coverage, not silently treated as complete.

Examples:
  python data/scripts/validate_tcga_data.py --root /private/tcga_data \\
    --modalities report histology expression --require-complete report

  python data/scripts/validate_tcga_data.py --root /private/tcga_data \\
    --modalities report --require-complete report \\
    --report-manifest /private/report_embedding_manifest.csv \\
    --json-out /private/multimodal_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

MODALITY_DIMS = {"report": 4096, "histology": 1536, "expression": 256}
CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")


def read_patient_ids(root: Path) -> tuple[set[str], set[str], dict[str, int]]:
    """Validate tables and return (clinical IDs, study IDs, fold sizes)."""
    clinical = pd.read_csv(root / "clinical.csv")
    splits = pd.read_csv(root / "split_cases.csv")
    if "case_id" not in clinical or not {"case_id", "split"}.issubset(splits.columns):
        raise ValueError("clinical.csv needs case_id; split_cases.csv needs case_id,split")
    for label, df in (("clinical", clinical), ("split", splits)):
        col = df["case_id"]
        if col.isna().any() or col.duplicated().any():
            raise ValueError(f"{label}: missing or duplicate case_id values")
        if not col.astype(str).map(lambda x: bool(CASE_RE.fullmatch(x))).all():
            raise ValueError(f"{label}: invalid TCGA case_id values")
    if splits["split"].isna().any():
        raise ValueError("split_cases.csv has missing fold assignments")
    # Check that folds are nonnegative integers; do not assume an arbitrary fold count.
    fold_numeric = pd.to_numeric(splits["split"], errors="coerce")
    if fold_numeric.isna().any() or (fold_numeric < 0).any() or (fold_numeric % 1 != 0).any():
        raise ValueError("split_cases.csv must contain nonnegative integer folds")
    study_ids = set(splits["case_id"].astype(str))
    clinical_ids = set(clinical["case_id"].astype(str))
    if not study_ids:
        raise ValueError("Study cohort is empty")
    if not study_ids.issubset(clinical_ids):
        raise ValueError("Study has patients not present in clinical.csv")
    counts = fold_numeric.astype(int).value_counts().sort_index()
    return clinical_ids, study_ids, {str(k): int(v) for k, v in counts.items()}


def check_vector(path: Path, dim: int, *, unit_norm: bool = False) -> bool:
    """Check shape/dtype/finiteness and (for BioMistral reports) unit norm."""
    try:
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        if arr.shape != (dim,) or arr.dtype != np.float32:
            return False
        if not bool(np.isfinite(arr).all()):
            return False
        if unit_norm and not bool(np.isclose(np.linalg.norm(arr), 1.0, atol=1e-4)):
            return False
        return True
    except (OSError, EOFError, ValueError, TypeError):
        return False


def read_report_manifest(path: Path, root: Path, study_ids: set[str]) -> set[str]:
    """Validate an optional private manifest; return expected relative paths."""
    df = pd.read_csv(path)
    required = {"case_id", "report_embedding_relpath"}
    if not required.issubset(df.columns):
        raise ValueError("Manifest needs case_id,report_embedding_relpath")
    if df[list(required)].isna().any().any() or df["case_id"].duplicated().any():
        raise ValueError("Manifest has missing or duplicate patient IDs/paths")
    if set(df["case_id"].astype(str)) != study_ids:
        raise ValueError("Manifest patients do not match study cohort")
    expected: set[str] = set()
    for patient_id, rel in zip(df["case_id"].astype(str), df["report_embedding_relpath"].astype(str)):
        p = PurePosixPath(rel)
        if (p.is_absolute() or ".." in p.parts or "\\" in rel
                or len(p.parts) != 4 or p.parts[:2] != ("embeddings", "report")
                or p.parts[2] != patient_id or not p.name.endswith(".npy")):
            raise ValueError("Manifest contains an unsafe or misaligned report path")
        expected.add(p.as_posix())
    if len(expected) != len(df):
        raise ValueError("Manifest has duplicate embedding paths")
    return expected


def audit_data(
    root: Path,
    modalities: list[str],
    *,
    required_complete: list[str] | None = None,
    report_manifest: Path | None = None,
    expression_dim: int = 256,
) -> dict:
    """Return aggregate statistics and checks; no patient-level data is emitted."""
    root = Path(root)
    modalities = list(dict.fromkeys(modalities))
    required_complete = list(dict.fromkeys(required_complete or []))
    if not modalities or set(modalities) - set(MODALITY_DIMS):
        raise ValueError("Unknown or missing modality")
    if set(required_complete) - set(modalities):
        raise ValueError("--require-complete must be a subset of --modalities")
    if expression_dim <= 0:
        raise ValueError("Expression dimension must be positive")
    clinical_ids, study_ids, fold_counts = read_patient_ids(root)
    expected_report_paths = (
        read_report_manifest(report_manifest, root, study_ids)
        if report_manifest is not None else None
    )
    if expected_report_paths is not None and "report" not in modalities:
        raise ValueError("--report-manifest requires modality 'report'")

    problems: list[str] = []
    coverage: dict[str, set[str]] = {}
    summaries: dict[str, dict] = {}

    for modality in modalities:
        folder = root / "embeddings" / modality
        dim = expression_dim if modality == "expression" else MODALITY_DIMS[modality]
        all_files = sorted(folder.rglob("*.npy")) if folder.is_dir() else []
        valid_ids: set[str] = set()
        seen_ids: set[str] = set()
        invalid_files = 0
        invalid_paths = 0
        unexpected_clinical_ids: set[str] = set()
        valid_file_count = 0
        valid_relpaths: set[str] = set()
        invalid_patient_ids: set[str] = set()
        for path in all_files:
            rel = path.relative_to(folder)
            if (len(rel.parts) != 2 or not CASE_RE.fullmatch(rel.parts[0])
                    or path.is_symlink() or path.parent.is_symlink()):
                invalid_paths += 1
                continue
            patient_id = rel.parts[0]
            seen_ids.add(patient_id)
            if patient_id not in clinical_ids:
                unexpected_clinical_ids.add(patient_id)
            if not check_vector(path, dim, unit_norm=(modality == "report")):
                invalid_files += 1
                invalid_patient_ids.add(patient_id)
                continue
            valid_file_count += 1
            valid_relpaths.add(path.relative_to(root).as_posix())
            if patient_id in study_ids:
                valid_ids.add(patient_id)
        # Patients with ANY invalid file are not counted as fully covered.
        valid_ids -= invalid_patient_ids
        if expected_report_paths is not None and modality == "report":
            missing_paths = expected_report_paths - valid_relpaths
            extra_paths = valid_relpaths - expected_report_paths
            if missing_paths or extra_paths:
                problems.append(
                    f"report: manifest differs: {len(missing_paths)} missing and {len(extra_paths)} extra paths"
                )
        if invalid_files or invalid_paths or unexpected_clinical_ids:
            problems.append(
                f"{modality}: {invalid_files} invalid vectors, {invalid_paths} bad paths, "
                f"{len(unexpected_clinical_ids)} IDs absent from clinical.csv"
            )
        if modality in required_complete and valid_ids != study_ids:
            problems.append(f"{modality}: {len(study_ids - valid_ids)} study patients lack valid vectors")
        coverage[modality] = valid_ids
        summaries[modality] = {
            "folder_exists": folder.is_dir(),
            "files": len(all_files),
            "valid_files": valid_file_count,
            "patients_with_files": len(seen_ids),
            "study_patients_covered": len(valid_ids),
            "study_patients_missing": len(study_ids - valid_ids),
            "patients_outside_study": len(seen_ids - study_ids),
            "invalid_vectors": invalid_files,
            "invalid_paths": invalid_paths,
            "embedding_dim": dim,
        }

    overlaps: dict[str, int] = {}
    if len(modalities) >= 2:
        for k in range(2, len(modalities) + 1):
            for combo in combinations(sorted(modalities), k):
                overlaps["+".join(combo)] = len(set.intersection(*(coverage[name] for name in combo)))
    return {
        "clinical_patients": len(clinical_ids),
        "study_patients": len(study_ids),
        "fold_counts": fold_counts,
        "modalities": summaries,
        "overlap_study_patients": overlaps,
        "passed": not problems,
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="PyHealth tcga_data root")
    parser.add_argument("--modalities", nargs="+", choices=sorted(MODALITY_DIMS),
                        default=["report", "histology", "expression"])
    parser.add_argument("--require-complete", nargs="*", choices=sorted(MODALITY_DIMS),
                        default=[], help="Require 100%% study-patient coverage")
    parser.add_argument("--report-manifest", type=Path,
                        help="Optional private report_embedding_manifest.csv for exact paths")
    parser.add_argument("--expression-dim", type=int, default=256,
                        help="Expected BulkRNABert embedding dimension (default: 256)")
    parser.add_argument("--json-out", type=Path, help="Save aggregate-only audit JSON")
    args = parser.parse_args()
    result = audit_data(args.root, args.modalities,
                        required_complete=args.require_complete,
                        report_manifest=args.report_manifest,
                        expression_dim=args.expression_dim)
    print(f"Clinical patients: {result['clinical_patients']}")
    print(f"Study patients: {result['study_patients']}")
    print("Fold counts:", result["fold_counts"])
    for name, stats in result["modalities"].items():
        print(f"{name}: {stats['study_patients_covered']}/{result['study_patients']} patients, "
              f"{stats['files']} files, {stats['invalid_vectors']} invalid vectors, "
              f"{stats['invalid_paths']} invalid paths")
    print("Overlaps:", result["overlap_study_patients"])
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Aggregate audit saved:", args.json_out)
    if not result["passed"]:
        for issue in result["problems"]:
            print("AUDIT ISSUE:", issue)
        raise SystemExit(1)
    print("MULTIMODAL DATA VALIDATION PASSED!")


if __name__ == "__main__":
    main()
