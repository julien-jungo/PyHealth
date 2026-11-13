#!/usr/bin/env python3
"""Embed TCGA summarized reports with 4-bit BioMistral-7B.

Writes one L2-normalized float32 vector per study report at
    OUTPUT_DIR/<case_id>/<patient_filename>.npy

Uses a quantized Transformers backbone with final-token mean-free pooling;
this is NOT guaranteed numerically identical to the paper's BF16/vLLM run.

Examples:
  python data/scripts/embed_reports_biomistral.py \
    --reports data/summarized_reports.csv --splits results/split_cases.csv \
    --output-dir /private/embeddings/report --max-new 5

  python data/scripts/embed_reports_biomistral.py \
    --reports data/summarized_reports.csv --splits results/split_cases.csv \
    --output-dir /private/embeddings/report --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIM = 4096
MODEL_ID = "BioMistral/BioMistral-7B"
ID_PATTERN = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")
POOLING = "final-token hidden state, L2 normalized"
QUANT = "4-bit NF4, double quantization"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_study(reports_csv: Path, splits_csv: Path) -> pd.DataFrame:
    """Exactly one safe, nonempty filename/text row per split-case patient."""
    reports = pd.read_csv(reports_csv)
    splits = pd.read_csv(splits_csv)
    if not {"patient_filename", "text"}.issubset(reports.columns):
        raise ValueError("Reports CSV needs patient_filename,text columns")
    if "case_id" not in splits.columns:
        raise ValueError("Split CSV needs case_id column")
    if reports[["patient_filename", "text"]].isna().any().any():
        raise ValueError("Report filenames/text cannot be missing")
    if reports["patient_filename"].duplicated().any():
        raise ValueError("Duplicate report filenames")
    reports = reports.copy()
    reports["patient_filename"] = reports["patient_filename"].astype(str)
    reports["case_id"] = reports["patient_filename"].str.split(
        ".", n=1, regex=False
    ).str[0]
    split_ids = splits["case_id"].astype(str)
    if split_ids.duplicated().any():
        raise ValueError("Duplicate study case IDs in split CSV")
    study_ids = set(split_ids)
    if not all(ID_PATTERN.fullmatch(pid) for pid in study_ids):
        raise ValueError("Invalid TCGA case ID in split CSV")
    study = reports[reports["case_id"].isin(study_ids)].copy()
    if set(study["case_id"]) != study_ids:
        raise ValueError("Some study patients have no pathology report")
    if study["case_id"].duplicated().any():
        raise ValueError("Expected exactly one summarized report per study patient")
    for row in study.itertuples(index=False):
        filename = row.patient_filename
        if not filename or filename in {".", ".."} or Path(filename).name != filename:
            raise ValueError(f"Unsafe report filename: {filename!r}")
        if not str(row.text).strip():
            raise ValueError(f"Empty report text: {filename}")
    return study.sort_values("case_id").reset_index(drop=True)


def output_path(output_dir: Path, case_id: str, filename: str) -> Path:
    if not ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"Invalid case ID: {case_id}")
    if not filename or filename in {".", ".."} or Path(filename).name != filename:
        raise ValueError(f"Unsafe report filename: {filename}")
    return output_dir / case_id / (filename + ".npy")


def valid_vector(array: np.ndarray, *, unit_norm: bool = True) -> bool:
    if array.shape != (DIM,) or array.dtype != np.float32:
        return False
    if not np.isfinite(array).all():
        return False
    return not unit_norm or bool(np.isclose(np.linalg.norm(array), 1, atol=1e-4))


def existing_valid(path: Path) -> bool:
    try:
        return valid_vector(np.load(path, allow_pickle=False))
    except (OSError, ValueError, EOFError, TypeError):
        return False


def write_atomic(path: Path, array: np.ndarray) -> None:
    """Write a complete .npy before replacing destination; supports resume."""
    if not valid_vector(array):
        raise ValueError(f"Invalid embedding for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".part-", suffix=".npy", dir=path.parent, delete=False
        ) as handle:
            temp_name = handle.name
            np.save(handle, array, allow_pickle=False)
        if not existing_valid(Path(temp_name)):
            raise ValueError("Temporary embedding failed validation")
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def audit(study: pd.DataFrame, output_dir: Path) -> tuple[int, int, int]:
    """Check exact filenames, number of patients, and stored vector validity."""
    expected = {
        output_path(output_dir, row.case_id, row.patient_filename).relative_to(
            output_dir
        ).as_posix()
        for row in study.itertuples(index=False)
    }
    actual_paths = list(output_dir.glob("*/*.npy")) if output_dir.exists() else []
    actual = {p.relative_to(output_dir).as_posix() for p in actual_paths}
    missing = expected - actual
    unexpected = actual - expected
    invalid = [p for p in actual_paths if not existing_valid(p)]
    print(f"Expected study reports: {len(expected)}")
    print(f"Saved embeddings:      {len(actual_paths)}")
    print(f"Missing expected:      {len(missing)}")
    print(f"Unexpected files:      {len(unexpected)}")
    print(f"Invalid vectors:       {len(invalid)}")
    if missing or unexpected or invalid:
        raise ValueError("Embedding audit failed; see counts above")
    print("REPORT EMBEDDING AUDIT PASSED!")
    return len(actual_paths), len(missing), len(invalid)


def config_payload(reports: Path, splits: Path, max_tokens: int) -> dict:
    return {
        "model": MODEL_ID,
        "quantization": QUANT,
        "compute_dtype": "float16",
        "pooling": POOLING,
        "embedding_dtype": "float32",
        "embedding_dimension": DIM,
        "max_allowed_tokens": max_tokens,
        "reports_sha256": sha256(reports),
        "splits_sha256": sha256(splits),
    }


def check_or_write_config(output_dir: Path, config: dict) -> None:
    """Refuse mixing different known configs in the same embedding folder."""
    config_file = output_dir / "embedding_config.json"
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding="utf-8"))
        for key, current in config.items():
            if previous.get(key) != current:
                raise ValueError(f"Configuration mismatch for {key}: {previous.get(key)!r} vs {current!r}")
        return
    if any(output_dir.glob("*/*.npy")):
        raise ValueError(
            "Existing embeddings but no embedding_config.json. "
            "Audit and establish provenance before generating into this folder."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def embed_one(model, tokenizer, text: str, max_tokens: int) -> tuple[np.ndarray, int]:
    """Compute last-token embedding using the successful Colab inference path."""
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    token_count = inputs["input_ids"].shape[1]
    if token_count > max_tokens:
        raise ValueError(f"Report has {token_count} tokens; limit is {max_tokens}")
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        outputs = model.model(**inputs, use_cache=False, return_dict=True)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        normalized = F.normalize(last_hidden.float(), p=2, dim=-1)
        array = normalized[0].cpu().numpy().astype(np.float32)
    if not valid_vector(array):
        raise ValueError("Model generated an invalid embedding")
    return array, token_count


def run(args: argparse.Namespace) -> None:
    reports = Path(args.reports)
    splits = Path(args.splits)
    output_dir = Path(args.output_dir)
    study = load_study(reports, splits)
    print(f"Matched study patients: {len(study)}")

    if args.check_only:
        audit(study, output_dir)
        return

    if args.max_new is not None and args.max_new < 0:
        raise ValueError("--max-new must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    check_or_write_config(output_dir, config_payload(reports, splits, args.max_tokens))
    if args.max_new == 0:
        print("No new embeddings requested")
        return
    pending = [
        row for row in study.itertuples(index=False)
        if not existing_valid(output_path(output_dir, row.case_id, row.patient_filename))
    ]
    print(f"Valid existing embeddings: {len(study) - len(pending)}")
    print(f"Pending generation: {len(pending)}")
    if not pending:
        print("Nothing to generate; use --check-only for full audit")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit BioMistral generation")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading BioMistral on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if model.config.hidden_size != DIM:
        raise ValueError(f"Unexpected model hidden_size: {model.config.hidden_size}")
    print(f"Model ready on {next(model.parameters()).device}")

    limit = len(pending) if args.max_new is None else min(len(pending), args.max_new)
    started = time.monotonic()
    for i, row in enumerate(pending[:limit], start=1):
        array, tokens = embed_one(model, tokenizer, str(row.text), args.max_tokens)
        target = output_path(output_dir, row.case_id, row.patient_filename)
        write_atomic(target, array)
        if i <= 5 or i % 50 == 0 or i == limit:
            print(f"Saved {i}/{limit} new embeddings | tokens={tokens} | {time.monotonic()-started:.1f}s")
    print(f"BATCH COMPLETE: {limit} new embeddings | {len(pending)-limit} still pending")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="Authors' summarized_reports.csv")
    parser.add_argument("--splits", required=True, help="Authors' split_cases.csv")
    parser.add_argument("--output-dir", required=True, help="Destination embeddings/report folder")
    parser.add_argument("--max-new", type=int, default=None, help="Max new files this invocation (default all)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Fail instead of truncating long reports")
    parser.add_argument("--check-only", action="store_true", help="Verify existing .npy files; do not load model")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
