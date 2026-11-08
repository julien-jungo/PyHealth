"""Multimodal cancer survival on TCGA with CoxSurvivalModel: pipeline + ablation.

Reproduces Song et al., "Multimodal Cancer Modeling in the Age of Foundation
Model Embeddings", ML4H 2025 (https://arxiv.org/abs/2505.07683): per-modality
Cox models on standardized, PCA-reduced foundation-model embeddings, combined
by two-stage late fusion.

Pipeline (per cross-validation fold):
    TCGAMultimodalDataset -> MultimodalSurvivalPrediction
    -> set_task(split=FoldSplit) (processors fitted on the training fold only)
    -> one CoxSurvivalModel per modality (stage 1)
    -> CoxSurvivalModel on the z-scored stage-1 risk scores (stage 2)
    -> concordance_index, pooled and per cancer type.

Experimental setup:
    TODO: data, folds, PCA dims, optimizer, epochs.
    Train full-batch (batch_size = training fold size), so the Cox risk sets
    match the paper's full partial likelihood. The paper's ridge penalty
    (scikit-survival ``CoxPHSurvivalAnalysis(alpha=0.1)``) becomes
    ``weight_decay = ridge_weight_decay(0.1, n_events)`` on the linear head,
    with ``n_events`` counted on the training fold.

Ablations (novel relative to the paper):
    TODO: two-stage (paper) vs. joint fusion; linear vs. MLP head;
    hidden_dim / dropout / learning-rate sweep. Build options per head:
    hidden_dim and dropout raise with head="linear".

Results:
    TODO: table of mean ± std C-index over folds, and per-cohort C-index next
    to the paper's Table 17.

Findings:
    TODO

Usage:
    python examples/tcga_survival_cox.py --synthetic
    python examples/tcga_survival_cox.py --root /path/to/data
"""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyhealth.datasets import Split
from pyhealth.tasks.survival import EMBEDDING_MODALITIES

PAPER_RIDGE_ALPHA = 0.1
N_FOLDS = 5

SYNTHETIC_COHORTS = ("TCGA-OV", "TCGA-UCS", "TCGA-UVM")
SYNTHETIC_COHORT_LOG_HAZARD = (0.5, 0.0, -0.5)
SYNTHETIC_SEXES = ("female", "male")
SYNTHETIC_RACES = ("white", "black or african american", "asian", "not reported")
SYNTHETIC_ETHNICITIES = ("not hispanic or latino", "hispanic or latino")
SYNTHETIC_AGE_RANGE = (30.0, 85.0)
SYNTHETIC_LOG_HAZARD_PER_AGE_YEAR = 0.02
SYNTHETIC_EMBEDDING_DIM = 32
SYNTHETIC_SIGNAL_DIMS = 2
# Latent std relative to unit noise: the signal directions dominate the
# variance, so PCA on each modality keeps them.
SYNTHETIC_SIGNAL_SCALE = 3.0
SYNTHETIC_LOG_HAZARD_PER_SIGNAL = 0.8
SYNTHETIC_BASELINE_SURVIVAL_DAYS = 1000.0
SYNTHETIC_CENSORING_DAYS = 1000.0
SYNTHETIC_MAX_SOURCES = 2


def ridge_weight_decay(alpha: float, n_events: int) -> float:
    """Converts scikit-survival's ridge ``alpha`` to an Adam ``weight_decay``.

    ``CoxPHSurvivalAnalysis(alpha)`` minimizes the Breslow negative partial
    log-likelihood plus ``alpha / 2 * ||w||^2``, both divided by the number of
    samples. ``cox_partial_likelihood_loss`` divides the likelihood term by the
    number of events instead, and ``weight_decay`` adds the gradient of
    ``weight_decay / 2 * ||w||^2``. Rescaling the objective doesn't move the
    minimizer, so ``alpha / n_events`` gives the same coefficients. The
    mapping is exact for the linear head trained full-batch to convergence.

    Args:
        alpha: Ridge penalty of ``CoxPHSurvivalAnalysis`` (the paper uses 0.1).
        n_events: Observed events in the training set.

    Returns:
        The ``weight_decay`` to pass to ``Trainer.train``.

    Raises:
        ValueError: If ``alpha`` is negative or ``n_events`` is below 1.

    Examples:
        >>> ridge_weight_decay(0.1, n_events=200)
        0.0005
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if n_events < 1:
        raise ValueError(f"n_events must be positive, got {n_events}")
    return alpha / n_events


class FoldSplit(Split):
    """Train/test split for one of the paper's published CV folds.

    Examples:
        >>> FoldSplit(test_patients=["p2"]).split_indices({"p1": [0], "p2": [1]})
        [array([0]), array([1])]
    """

    # TODO: implement like the HoldoutSplit example in splitter.py; read
    # the test patients of a fold from folds.csv (case_id, fold).
    def __init__(self, test_patients: Sequence[str]) -> None:
        raise NotImplementedError

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns the training indices, then the fold's test indices."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Returns the fold's test patients, for the processed-cache key."""
        raise NotImplementedError


def _embed(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hides a latent signal in a few random directions of noisy embeddings."""
    directions, _ = np.linalg.qr(
        rng.normal(size=(SYNTHETIC_EMBEDDING_DIM, signal.shape[1]))
    )
    noise = rng.normal(size=(len(signal), SYNTHETIC_EMBEDDING_DIM))
    return (SYNTHETIC_SIGNAL_SCALE * signal @ directions.T + noise).astype(np.float32)


def _write_embeddings(
    path: Path, case_ids: list[str], embeddings: np.ndarray, n_sources: np.ndarray
) -> None:
    """Writes one modality's table with a ``list<float32>`` embedding column."""
    table = pa.table(
        {
            "case_id": case_ids,
            "embedding": pa.array(list(embeddings), type=pa.list_(pa.float32())),
            "n_sources": n_sources,
        }
    )
    pq.write_table(table, path)


def _assign_folds(cohort: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deals patients round-robin in cohort order, shuffled within each cohort.

    Fold sizes differ by at most one, and each cohort is spread as evenly as
    its size allows (a cohort smaller than ``N_FOLDS`` misses some folds).
    """
    order = np.lexsort((rng.random(len(cohort)), cohort))
    fold = np.empty(len(cohort), dtype=np.int64)
    fold[order] = np.arange(len(cohort)) % N_FOLDS
    return fold


def write_synthetic_data(root: str, n_patients: int = 300, seed: int = 0) -> None:
    """Writes a synthetic dataset with a known survival signal to ``root``.

    Uses the layout of :class:`~pyhealth.datasets.TCGAMultimodalDataset`, plus
    ``folds.csv`` (``case_id``, ``fold``). Each embedding modality hides its
    own latent signal in a few high-variance directions, and the log hazard
    sums those signals with a cohort and an age effect. Survival and
    censoring times are exponential. Every modality should therefore beat
    C-index 0.5, and fusing them should beat each one alone.

    Args:
        root: Output directory; created if missing, files overwritten.
        n_patients: Number of synthetic TCGA cases.
        seed: Seed of the random generator.

    Raises:
        ValueError: If ``n_patients`` is smaller than the number of folds.
    """
    if n_patients < N_FOLDS:
        raise ValueError(f"n_patients must be at least {N_FOLDS}, got {n_patients}")

    rng = np.random.default_rng(seed)
    root_path = Path(root)
    (root_path / "embeddings").mkdir(parents=True, exist_ok=True)
    case_ids = [f"TCGA-SY-{i:04d}" for i in range(n_patients)]

    cohort = rng.integers(len(SYNTHETIC_COHORTS), size=n_patients)
    age = rng.uniform(*SYNTHETIC_AGE_RANGE, size=n_patients)
    log_hazard = np.asarray(SYNTHETIC_COHORT_LOG_HAZARD)[cohort]
    log_hazard += SYNTHETIC_LOG_HAZARD_PER_AGE_YEAR * (
        age - np.mean(SYNTHETIC_AGE_RANGE)
    )

    for modality in EMBEDDING_MODALITIES:
        signal = rng.normal(size=(n_patients, SYNTHETIC_SIGNAL_DIMS))
        log_hazard += SYNTHETIC_LOG_HAZARD_PER_SIGNAL * signal.sum(axis=1)
        _write_embeddings(
            root_path / "embeddings" / f"{modality}.parquet",
            case_ids,
            _embed(signal, rng),
            rng.integers(1, SYNTHETIC_MAX_SOURCES + 1, size=n_patients),
        )

    death = rng.exponential(SYNTHETIC_BASELINE_SURVIVAL_DAYS * np.exp(-log_hazard))
    follow_up = rng.exponential(SYNTHETIC_CENSORING_DAYS, size=n_patients)
    dead = death <= follow_up
    pd.DataFrame(
        {
            "case_id": case_ids,
            "project": np.asarray(SYNTHETIC_COHORTS)[cohort],
            "sex": rng.choice(SYNTHETIC_SEXES, size=n_patients),
            "age": age,
            "race": rng.choice(SYNTHETIC_RACES, size=n_patients),
            "ethnicity": rng.choice(SYNTHETIC_ETHNICITIES, size=n_patients),
            "vital_status": np.where(dead, "Dead", "Alive"),
            "days_to_death": np.where(dead, np.round(death), np.nan),
            "days_to_last_follow_up": np.round(np.minimum(death, follow_up)),
        }
    ).to_csv(root_path / "clinical.csv", index=False)
    pd.DataFrame({"case_id": case_ids, "fold": _assign_folds(cohort, rng)}).to_csv(
        root_path / "folds.csv", index=False
    )


def run_fold(root: str, fold: int) -> dict[str, float]:
    """Trains and evaluates every configuration on one fold."""
    # TODO: stage 1, stage 2, ablations; return C-index per
    # configuration (and per cohort). On synthetic data, PCA components must
    # not exceed SYNTHETIC_EMBEDDING_DIM (the task's 256 default would raise).
    raise NotImplementedError


def main() -> None:
    """Runs the pipeline on --root or on synthetic data and prints results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--root", help="dataset directory that also holds folds.csv (case_id, fold)"
    )
    source.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    # TODO: write synthetic data to a temp dir if requested, run all
    # folds, print the results table.
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
