"""Multimodal cancer survival on TCGA with CoxSurvivalModel: pipeline + ablation.

Contributors: Julien David Jungo (jjungo2), Donovan Scott Below (dbelow2),
    Yaseen Mohamed (yaseenm2)
Paper: Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
    Embeddings", ML4H 2025
Paper link: https://proceedings.mlr.press/v297/song26a.html
    (preprint: https://arxiv.org/abs/2505.07683)
Description: Reproduces the paper's per-modality Cox models on
    standardized, PCA-reduced foundation-model embeddings, combined by
    two-stage late fusion, and ablates the fusion and the risk head.

Pipeline (per cross-validation fold, the paper's 5 folds from ``folds.csv``):
    TCGAMultimodalDataset -> MultimodalSurvivalPrediction
    -> set_task(split=FoldSplit) (processors fitted on the training fold only)
    -> one CoxSurvivalModel per modality (stage 1)
    -> CoxSurvivalModel on the z-scored stage-1 risk scores (stage 2)
    -> concordance_index, pooled and per cancer type.

Ablations (novel relative to the paper):
    - Two-stage vs. joint fusion ("joint": one linear model, all modalities).
    - Linear vs. MLP head ("two-stage mlp", "joint mlp").
    - Hyperparameter sweep on "joint mlp": hidden_dim, dropout, learning rate.

Results (real, the 10 smallest TCGA cohorts, 873 patients, PCA 256): two-stage
fusion reaches C-index 0.711 ± 0.030, vs. 0.793 in the paper on all 32
cohorts. It beats each embedding modality (0.660-0.664) but not cancer type
alone (0.709). Setup, convergence checks, full synthetic and real tables,
per-cohort results vs. the paper's Table 17, and caveats are in
``examples/tcga_survival_cox_survival.md``.

Usage (synthetic data in a temp dir by default):
    python examples/tcga_survival_cox_survival.py
    python examples/tcga_survival_cox_survival.py --root /path/to/data \\
        --cache-dir /path/to/cache
"""

import os

# Before any import of tqdm: one progress bar per fit would flood the output.
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import logging
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import (
    SampleDataset,
    Split,
    TCGAMultimodalDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.metrics.survival import concordance_index
from pyhealth.models import CoxSurvivalModel
from pyhealth.models.cox_survival import Head
from pyhealth.processors import TensorProcessor
from pyhealth.tasks import MultimodalSurvivalPrediction
from pyhealth.tasks.survival import EMBEDDING_MODALITIES
from pyhealth.trainer import Trainer

PAPER_RIDGE_ALPHA = 0.1
# The paper's reported setting (results/analyze-results.ipynb, Table 1).
PAPER_PCA_COMPONENTS = {"expr": 256, "hist": 256, "text": 256}
SYNTHETIC_PCA_COMPONENTS = 8
N_FOLDS = 5
SEED = 0

LEARNING_RATE = 1e-2
# Full-batch Adam steps; see "Optimizer" in the module docstring.
LINEAR_STEPS = 3000
JOINT_LINEAR_STEPS = 10000
MLP_STEPS = 300
MLP_HIDDEN_DIM = 64

FUSION_KEY = "risk"
PAPER_FUSION = "two-stage"
PAPER_CONFIGS = ("demo", "canc", "expr", "hist", "text", PAPER_FUSION)

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
SYNTHETIC_SOURCE_NOISE = 0.1


@dataclass(frozen=True)
class HeadConfig:
    """Risk head and optimizer options of one ``CoxSurvivalModel`` fit."""

    head: Head = "linear"
    hidden_dim: int | None = None
    dropout: float | None = None
    learning_rate: float = LEARNING_RATE
    steps: int = LINEAR_STEPS


LINEAR = HeadConfig()
MLP = HeadConfig(head="mlp", hidden_dim=MLP_HIDDEN_DIM, dropout=0.0, steps=MLP_STEPS)
JOINT_CONFIGS = {
    "joint": HeadConfig(steps=JOINT_LINEAR_STEPS),
    "joint mlp": MLP,
    "joint mlp h=16": replace(MLP, hidden_dim=16),
    "joint mlp p=0.3": replace(MLP, dropout=0.3),
    "joint mlp lr=1e-3": replace(MLP, learning_rate=1e-3),
}


@dataclass(frozen=True)
class Predictions:
    """Risk scores and labels of one model on one dataset part."""

    patient_ids: tuple[str, ...]
    risk: np.ndarray
    duration: np.ndarray
    event: np.ndarray


@dataclass(frozen=True)
class FoldResult:
    """Test C-index of each configuration on one fold.

    Attributes:
        c_index: Configuration name to C-index over the fold's test patients.
        cohort_c_index: Configuration name to cancer type to C-index; NaN
            where a cohort has no comparable pair in the fold.
    """

    c_index: dict[str, float]
    cohort_c_index: dict[str, dict[str, float]]


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

    The fold's patients form the test part; every other patient trains, as
    in the paper's ``run_experiment``. Take the test patients of fold ``k``
    from :meth:`~pyhealth.datasets.TCGAMultimodalDataset.get_fold_assignments`.

    Args:
        test_patients: Patient IDs of the fold.

    Raises:
        ValueError: If ``test_patients`` is empty.

    Examples:
        >>> FoldSplit(test_patients=["p2"]).split_indices({"p1": [0], "p2": [1]})
        [array([0]), array([1])]
    """

    def __init__(self, test_patients: Sequence[str]) -> None:
        if not test_patients:
            raise ValueError("test_patients must not be empty")
        self.test_patients = sorted(set(test_patients))

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns the training indices, then the fold's test indices.

        Args:
            patient_to_index: Patient ID to the indices of its samples.

        Returns:
            Two sorted ``int64`` index arrays: training part, test part.

        Raises:
            ValueError: If either part has no sample.
        """
        test = set(self.test_patients)
        parts: tuple[list[int], list[int]] = ([], [])
        for patient_id, indices in patient_to_index.items():
            parts[patient_id in test].extend(indices)

        if not parts[0] or not parts[1]:
            raise ValueError(
                f"Fold split needs training and test samples; got {len(parts[0])} "
                f"training and {len(parts[1])} test samples"
            )
        return [np.array(sorted(part), dtype=np.int64) for part in parts]

    def to_dict(self) -> dict:
        """Returns the fold's test patients, for the processed-cache key."""
        return {"kind": "fold", "test_patients": self.test_patients}


def _embed(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hides a latent signal in a few random directions of noisy embeddings.

    Args:
        signal: Latent signal, shape ``[n_patients, n_signal_dims]``.
        rng: Random generator for the directions and the noise.

    Returns:
        Embeddings of shape ``[n_patients, SYNTHETIC_EMBEDDING_DIM]``, float32.
    """
    directions, _ = np.linalg.qr(
        rng.normal(size=(SYNTHETIC_EMBEDDING_DIM, signal.shape[1]))
    )
    noise = rng.normal(size=(len(signal), SYNTHETIC_EMBEDDING_DIM))
    return (SYNTHETIC_SIGNAL_SCALE * signal @ directions.T + noise).astype(np.float32)


def _write_embeddings(
    folder: Path,
    case_ids: list[str],
    embeddings: np.ndarray,
    n_sources: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Writes one modality as ``<case_id>/<file_id>.npy`` files under ``folder``.

    A case with several sources (e.g. slides) gets one file per source. The
    files scatter around the case's embedding and average back to it, so the
    task's ``mean`` aggregation recovers the embedding.

    Args:
        folder: Modality folder, e.g. ``root/embeddings/expr``.
        case_ids: TCGA case ID of each row.
        embeddings: One embedding per case, shape ``[n_patients, dim]``.
        n_sources: Number of source files to write for each case.
        rng: Random generator for the scatter between sources.

    Returns:
        None. Existing files of the same name are overwritten.
    """
    for case_id, embedding, n in zip(case_ids, embeddings, n_sources):
        case_dir = folder / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        scatter = rng.normal(scale=SYNTHETIC_SOURCE_NOISE, size=(n, len(embedding)))
        scatter -= scatter.mean(axis=0)

        for source, vector in enumerate(embedding + scatter):
            np.save(case_dir / f"source-{source}.npy", vector.astype(np.float32))


def _assign_folds(cohort: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deals patients round-robin in cohort order, shuffled within each cohort.

    Fold sizes differ by at most one, and each cohort is spread as evenly as
    its size allows (a cohort smaller than ``N_FOLDS`` misses some folds).

    Args:
        cohort: Cohort index of each patient.
        rng: Random generator for the shuffle within each cohort.

    Returns:
        The fold of each patient, in ``[0, N_FOLDS)``.
    """
    order = np.lexsort((rng.random(len(cohort)), cohort))
    fold = np.empty(len(cohort), dtype=np.int64)
    fold[order] = np.arange(len(cohort)) % N_FOLDS
    return fold


def write_synthetic_data(root: str, n_patients: int = 300, seed: int = 0) -> None:
    """Writes a synthetic dataset with a known survival signal to ``root``.

    Uses the layout of :class:`~pyhealth.datasets.TCGAMultimodalDataset`,
    including ``folds.csv`` (``case_id``, ``fold``). Each embedding modality
    hides its own latent signal in a few high-variance directions, and the log
    hazard sums those signals with a cohort and an age effect. Survival and
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
    root_path.mkdir(parents=True, exist_ok=True)
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
            root_path / "embeddings" / modality,
            case_ids,
            _embed(signal, rng),
            rng.integers(1, SYNTHETIC_MAX_SOURCES + 1, size=n_patients),
            rng,
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


def _n_events(dataset: SampleDataset) -> int:
    """Counts the observed events of a dataset."""
    return int(sum(float(sample["event"].sum()) for sample in dataset))


def full_batch_loader(dataset: SampleDataset) -> DataLoader:
    """Collates ``dataset`` once and returns a loader that yields that batch.

    A ``set_task`` part is a litdata subset that is re-read on every pass,
    which made full-batch training ~20x slower than one cached batch.

    Args:
        dataset: Samples to collate into one batch.

    Returns:
        A loader with one batch of all samples, in the dataset's order.
    """
    batch = next(iter(get_dataloader(dataset, batch_size=len(dataset))))
    return DataLoader([batch], batch_size=None)


def _predict(trainer: Trainer, dataset: SampleDataset) -> Predictions:
    """Scores every sample of ``dataset`` with the trainer's model.

    Args:
        trainer: Trainer that holds a fitted ``CoxSurvivalModel``.
        dataset: Samples to score, with ``duration`` and ``event``.

    Returns:
        The risks and labels, in the dataset's order.
    """
    y_true, y_prob, _, patient_ids = trainer.inference(
        full_batch_loader(dataset), return_patient_ids=True
    )
    return Predictions(
        patient_ids=tuple(patient_ids),
        risk=y_prob.reshape(-1),
        duration=y_true[:, 0],
        event=y_true[:, 1],
    )


def fit_predict(
    train: SampleDataset,
    test: SampleDataset,
    feature_keys: Sequence[str],
    config: HeadConfig,
    n_events: int,
) -> tuple[Predictions, Predictions]:
    """Trains one ``CoxSurvivalModel`` full-batch and scores both parts.

    The ridge penalty is the paper's, rescaled by :func:`ridge_weight_decay`.
    It applies to every head; only for the linear head does it equal the
    paper's ridge CoxPH.

    Args:
        train: Training part; its fitted processors define the features.
        test: Test part, scored with the trained model.
        feature_keys: Input fields of the model.
        config: Head and optimizer options.
        n_events: Observed events in ``train``, for the ridge rescaling.

    Returns:
        Predictions on ``train`` (in-sample, as the paper feeds stage 2),
        then on ``test``.
    """
    torch.manual_seed(SEED)
    model = CoxSurvivalModel(
        train,
        feature_keys=list(feature_keys),
        head=config.head,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )

    trainer = Trainer(model=model, enable_logging=False)
    trainer.train(
        train_dataloader=full_batch_loader(train),
        epochs=1,
        steps_per_epoch=config.steps,
        optimizer_params={"lr": config.learning_rate},
        weight_decay=ridge_weight_decay(PAPER_RIDGE_ALPHA, n_events),
    )

    return _predict(trainer, train), _predict(trainer, test)


def _risk_samples(predictions: Mapping[str, Predictions]) -> list[dict]:
    """Turns aligned stage-1 risks into stage-2 samples with one risk vector.

    Args:
        predictions: Stage-1 predictions per modality, on the same patients
            in the same order.

    Returns:
        One sample per patient: ``risk`` (one value per modality, in the
        mapping's order), ``duration`` and ``event``.

    Raises:
        ValueError: If the modalities' patients or their order differ.
    """
    first, *others = predictions.values()
    if any(other.patient_ids != first.patient_ids for other in others):
        raise ValueError("stage-1 predictions must cover the same patients in order")

    risks = np.stack([p.risk for p in predictions.values()], axis=1)
    return [
        {
            "patient_id": patient_id,
            FUSION_KEY: risk.tolist(),
            "duration": [float(duration)],
            "event": [float(event)],
        }
        for patient_id, risk, duration, event in zip(
            first.patient_ids, risks, first.duration, first.event
        )
    ]


def fusion_datasets(
    train: Mapping[str, Predictions], test: Mapping[str, Predictions]
) -> tuple[SampleDataset, SampleDataset]:
    """Builds the stage-2 datasets: risks z-scored with training statistics.

    The paper z-scores each modality's training risks and applies the same
    scaler to the test risks. Here a ``TensorProcessor`` with ``"zscore"``
    is fitted on the training risks only and shared by both parts.

    Args:
        train: Stage-1 predictions on the training part, per modality.
        test: Stage-1 predictions on the test part, same modalities.

    Returns:
        The stage-2 training and test datasets.
    """
    train_samples = _risk_samples(train)

    processor = TensorProcessor(normalization="zscore")
    processor.fit(train_samples, FUSION_KEY)

    def build(samples: list[dict]) -> SampleDataset:
        return create_sample_dataset(
            samples,
            input_schema={FUSION_KEY: "tensor"},
            output_schema=MultimodalSurvivalPrediction.output_schema,
            input_processors={FUSION_KEY: processor},
        )

    return build(train_samples), build(_risk_samples(test))


def two_stage(
    train: SampleDataset,
    test: SampleDataset,
    modalities: Sequence[str],
    config: HeadConfig,
    n_events: int,
) -> tuple[dict[str, Predictions], Predictions]:
    """Runs the paper's late fusion: unimodal Cox models, then Cox on risks.

    Args:
        train: Training part with every modality.
        test: Test part.
        modalities: Stage-1 modalities, one model each.
        config: Head of the stage-1 models. Stage 2 is always the linear
            Cox model, as in the paper.
        n_events: Observed events in ``train``.

    Returns:
        Test predictions per stage-1 modality, and of the fused model.
    """
    stage1 = {
        modality: fit_predict(train, test, [modality], config, n_events)
        for modality in modalities
    }

    fusion_train, fusion_test = fusion_datasets(
        {modality: parts[0] for modality, parts in stage1.items()},
        {modality: parts[1] for modality, parts in stage1.items()},
    )
    _, fused = fit_predict(fusion_train, fusion_test, [FUSION_KEY], LINEAR, n_events)

    return {modality: parts[1] for modality, parts in stage1.items()}, fused


def _cohort_c_index(
    predictions: Predictions, cancer_types: Mapping[str, str]
) -> dict[str, float]:
    """Scores predictions within each cancer type; NaN without a usable pair."""
    cohorts = np.array([cancer_types[p] for p in predictions.patient_ids])
    return {
        cohort: concordance_index(
            predictions.duration[mask], predictions.event[mask], predictions.risk[mask]
        )
        for cohort in np.unique(cohorts)
        for mask in [cohorts == cohort]
    }


def score(
    predictions: Mapping[str, Predictions], cancer_types: Mapping[str, str]
) -> FoldResult:
    """Computes the pooled and per-cohort C-index of each configuration.

    Args:
        predictions: Test predictions per configuration name.
        cancer_types: Patient ID to TCGA project.

    Returns:
        The fold's C-index per configuration, pooled and per cohort.
    """
    return FoldResult(
        c_index={
            name: concordance_index(p.duration, p.event, p.risk)
            for name, p in predictions.items()
        },
        cohort_c_index={
            name: _cohort_c_index(p, cancer_types) for name, p in predictions.items()
        },
    )


def run_fold(
    dataset: TCGAMultimodalDataset,
    fold: int,
    pca_components: Mapping[str, int | None],
) -> FoldResult:
    """Trains and evaluates every configuration on one fold.

    Processors (z-score + PCA, one-hot) are fitted on the training patients
    only, through ``set_task(task, split=FoldSplit(...))``.

    Args:
        dataset: Dataset whose patients all have a fold.
        fold: Fold whose patients form the test set.
        pca_components: PCA components per embedding modality.

    Returns:
        C-index per configuration (unimodal, fused, ablations), pooled over
        the fold's test patients and per cancer type.
    """
    task = MultimodalSurvivalPrediction(
        **{f"{m}_components": n for m, n in pca_components.items()}
    )
    folds = dataset.get_fold_assignments()
    split = FoldSplit([patient for patient, f in folds.items() if f == fold])
    train, test = dataset.set_task(task, split=split)
    n_events = _n_events(train)

    unimodal, fused = two_stage(train, test, task.modalities, LINEAR, n_events)
    _, fused_mlp = two_stage(train, test, task.modalities, MLP, n_events)
    predictions = {**unimodal, PAPER_FUSION: fused, "two-stage mlp": fused_mlp}

    for name, config in JOINT_CONFIGS.items():
        _, predictions[name] = fit_predict(
            train, test, task.modalities, config, n_events
        )

    return score(predictions, dataset.get_cancer_types())


def summarize(results: Sequence[FoldResult]) -> pd.DataFrame:
    """Returns mean and std (ddof=1) of the C-index over folds, per config."""
    table = pd.DataFrame([r.c_index for r in results])
    return pd.DataFrame({"mean": table.mean(), "std": table.std()})


def summarize_cohorts(
    results: Sequence[FoldResult], configs: Sequence[str] = PAPER_CONFIGS
) -> pd.DataFrame:
    """Returns ``mean ± std (folds)`` of the per-cohort C-index.

    Folds where a cohort has no comparable pair (NaN) are left out; the
    count in parentheses says how many folds remain.

    Args:
        results: One result per fold.
        configs: Configurations to show, one column each.

    Returns:
        One row per cohort, one column per configuration.
    """
    cells: dict[str, dict[str, str]] = {}
    for config in configs:
        table = pd.DataFrame([r.cohort_c_index[config] for r in results])
        cells[config] = {
            cohort: f"{values.mean():.3f} ± {values.std():.3f} ({values.count()})"
            for cohort, values in table.items()
        }
    return pd.DataFrame(cells).sort_index()


def run(
    root: str,
    cache_dir: str | None,
    pca_components: Mapping[str, int | None],
    num_workers: int,
) -> list[FoldResult]:
    """Loads the dataset and runs every fold.

    Args:
        root: Dataset directory in the loader layout, with ``folds.csv``.
        cache_dir: PyHealth cache directory; None for PyHealth's default.
        pca_components: PCA components per embedding modality.
        num_workers: Worker processes for the dataset and task.

    Returns:
        One result per fold, in fold order.

    Raises:
        ValueError: If a patient has no fold.
    """
    dataset = TCGAMultimodalDataset(
        root=root, cache_dir=cache_dir, num_workers=num_workers
    )

    folds = dataset.get_fold_assignments()
    unassigned = sorted(set(dataset.unique_patient_ids) - set(folds))
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} patient(s) have no fold in folds.csv, e.g. "
            f"{unassigned[:3]}; they would train in every fold"
        )

    return [
        run_fold(dataset, fold, pca_components) for fold in sorted(set(folds.values()))
    ]


def _parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        help="dataset directory in the TCGAMultimodalDataset layout, with "
        "folds.csv (case_id, fold); default: synthetic data in a temp dir",
    )
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument(
        "--pca-components",
        type=int,
        help="PCA components of every embedding modality; default: the paper's "
        f"{PAPER_PCA_COMPONENTS['expr']}, or {SYNTHETIC_PCA_COMPONENTS} on "
        "synthetic data",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Runs the pipeline on --root or on synthetic data and prints results."""
    args = _parse_args()
    logging.getLogger("pyhealth").setLevel(logging.WARNING)
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        if args.root is None:
            root = str(Path(tmp) / "data")
            write_synthetic_data(root)
            cache_dir = args.cache_dir or str(Path(tmp) / "cache")
            components = dict.fromkeys(EMBEDDING_MODALITIES, SYNTHETIC_PCA_COMPONENTS)
        else:
            root, cache_dir = args.root, args.cache_dir
            components = dict(PAPER_PCA_COMPONENTS)

        if args.pca_components is not None:
            components = dict.fromkeys(EMBEDDING_MODALITIES, args.pca_components)

        results = run(root, cache_dir, components, args.num_workers)

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(f"\nC-index over {len(results)} folds:")
        print(summarize(results).round(3).to_string())
        print("\nPer-cohort C-index, mean ± std (folds with a comparable pair):")
        print(summarize_cohorts(results).to_string())
    print(f"\nRuntime: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
