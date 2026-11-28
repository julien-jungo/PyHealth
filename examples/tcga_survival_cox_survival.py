"""Multimodal cancer survival on TCGA with CoxSurvivalModel: pipeline + ablation.

Contributors: Julien David Jungo (jjungo2), Donovan Scott Below (dbelow2),
    Yaseen Mohamed (yaseenm2)
Paper: Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
    Embeddings", ML4H 2025
Paper link: https://proceedings.mlr.press/v297/song26a.html
    (preprint: https://arxiv.org/abs/2505.07683)
Description: Reproduces the paper's per-modality Cox models on
    standardized, PCA-reduced foundation-model embeddings, combined by
    two-stage late fusion, and ablates the fusion, the risk head and the
    embedding reducer.

Pipeline, per fold of the paper's 5 CV folds (``split_cases.csv``):
    TCGAMultimodalDataset -> MultimodalSurvivalPrediction
    -> set_task(split=FoldSplit) (processors fitted on the training fold only)
    -> one CoxSurvivalModel per modality (stage 1)
    -> CoxSurvivalModel on the z-scored stage-1 risk scores (stage 2)
    -> concordance_index, pooled and per cancer type.

Ablations (novel relative to the paper):
    - Two-stage vs. joint fusion ("joint": one linear model, all modalities).
    - In-sample vs. out-of-fold stage-1 risks ("two-stage oof", ``--oof``).
    - Linear vs. MLP head ("two-stage mlp", "joint mlp").
    - Hyperparameter sweep on "joint mlp": hidden_dim, dropout, learning rate.
    - Embedding reducer: PCA (the paper), kernel PCA, FastICA or a Gaussian
      random projection (``--reduction-ablation``, run alone).

Results (real data, 10 TCGA cohorts chosen for small UNI2-h feature downloads
(at most 10 GB each), 873 patients, PCA 256):
two-stage fusion reaches C-index 0.711 ± 0.030, vs. 0.793 in the paper on
all 32 cohorts; it and cancer type alone (0.709 ± 0.016) are within the
noise. Out-of-fold stage-1 risks reach 0.739 ± 0.023. With kernel PCA
instead of PCA, two-stage fusion reaches 0.733 ± 0.023. Setup, all tables,
per-cohort results vs. the paper's Table 17, caveats and the guide to get
the real data are in ``examples/tcga_survival_cox_survival.md``.

Usage (synthetic data in a temp dir by default):
    python examples/tcga_survival_cox_survival.py
    python examples/tcga_survival_cox_survival.py --root /path/to/data \\
        --cache-dir /path/to/cache
    python examples/tcga_survival_cox_survival.py --oof  # adds "two-stage oof"
    python examples/tcga_survival_cox_survival.py --reduction fast_ica
    python examples/tcga_survival_cox_survival.py --reduction-ablation
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
from pyhealth.datasets.tcga_survival import NUM_FOLDS
from pyhealth.metrics.survival import concordance_index
from pyhealth.models import CoxSurvivalModel
from pyhealth.models.cox_survival import Head
from pyhealth.processors import TensorProcessor
from pyhealth.processors.embedding_reduction_processor import (
    METHOD_PCA,
    REDUCTION_METHODS,
)
from pyhealth.tasks import MultimodalSurvivalPrediction
from pyhealth.tasks.tcga_multimodal_survival import EMBEDDING_MODALITIES
from pyhealth.trainer import Trainer

logger = logging.getLogger(__name__)

PAPER_RIDGE_ALPHA = 0.1
# The paper's reported setting (results/analyze-results.ipynb, Table 1).
PAPER_PCA_COMPONENTS = {"expr": 256, "hist": 256, "text": 256}
SYNTHETIC_COMPONENTS = 8
INNER_FOLDS = 5
SEED = 0

LEARNING_RATE = 1e-2
# Full-batch Adam steps; see "Optimizer" in the module docstring.
LINEAR_STEPS = 3000
JOINT_LINEAR_STEPS = 10000
MLP_STEPS = 300
MLP_HIDDEN_DIM = 64

FUSION_KEY = "risk"
PAPER_FUSION = "two-stage"
OOF_FUSION = "two-stage oof"
PAPER_CONFIGS = ("demo", "canc", "expr", "hist", "text", PAPER_FUSION)
REDUCTION_CONFIGS = (*EMBEDDING_MODALITIES, PAPER_FUSION)
OOF_COHORT_CONFIGS = (*PAPER_CONFIGS, OOF_FUSION)
HAZARD_RATIO_FUSIONS = (PAPER_FUSION, OOF_FUSION)

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
# variance, so a reducer on each modality can keep them.
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
class Fusion:
    """Test predictions of a stage-2 model and its hazard ratios.

    Attributes:
        predictions: Fused risks on the test part.
        hazard_ratios: Modality to ``exp(coef)`` of its z-scored stage-1
            risk, as in the paper's Table 8.
    """

    predictions: Predictions
    hazard_ratios: dict[str, float]


@dataclass(frozen=True)
class FoldResult:
    """Test C-index of each configuration on one fold.

    Attributes:
        c_index: Configuration name to C-index over the fold's test patients.
        cohort_c_index: Configuration name to cancer type to C-index; NaN
            where a cohort has no comparable pair in the fold.
        hazard_ratios: Fusion configuration name to modality to stage-2
            hazard ratio.
    """

    c_index: dict[str, float]
    cohort_c_index: dict[str, dict[str, float]]
    hazard_ratios: dict[str, dict[str, float]]


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

    The fold's patients form the test part; the patients of every other fold
    train, as in the paper's ``run_experiment``. Patients without a fold are
    in neither part. Take the folds from
    :meth:`~pyhealth.datasets.TCGAMultimodalDataset.get_fold_assignments`.

    Args:
        folds: Patient ID to fold index.
        fold: Fold whose patients form the test part.

    Raises:
        ValueError: If no patient is in ``fold``, or no patient is in
            another fold.

    Examples:
        >>> split = FoldSplit(folds={"p1": 0, "p2": 1}, fold=1)
        >>> split.split_indices({"p1": [0], "p2": [1], "p3": [2]})
        [array([0]), array([1])]
    """

    def __init__(self, folds: Mapping[str, int], fold: int) -> None:
        self.train_patients = sorted(p for p, f in folds.items() if f != fold)
        self.test_patients = sorted(p for p, f in folds.items() if f == fold)

        if not self.train_patients or not self.test_patients:
            raise ValueError(
                f"Fold {fold} needs training and test patients; got "
                f"{len(self.train_patients)} training and "
                f"{len(self.test_patients)} test patients"
            )

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns the training indices, then the fold's test indices.

        Args:
            patient_to_index: Patient ID to the indices of its samples.

        Returns:
            Two sorted ``int64`` index arrays: training part, test part.
            Samples of patients without a fold are in neither part.

        Raises:
            ValueError: If either part has no sample.
        """
        train, test = set(self.train_patients), set(self.test_patients)
        parts: tuple[list[int], list[int]] = ([], [])
        for patient_id, indices in patient_to_index.items():
            if patient_id in train:
                parts[0].extend(indices)
            elif patient_id in test:
                parts[1].extend(indices)

        if not parts[0] or not parts[1]:
            raise ValueError(
                f"Fold split needs training and test samples; got {len(parts[0])} "
                f"training and {len(parts[1])} test samples"
            )
        return [np.array(sorted(part), dtype=np.int64) for part in parts]

    def to_dict(self) -> dict:
        """Returns both parts' patients, for the processed-cache key."""
        return {
            "kind": "fold",
            "train_patients": self.train_patients,
            "test_patients": self.test_patients,
        }


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


def _assign_folds(
    group: np.ndarray, rng: np.random.Generator, n_folds: int = NUM_FOLDS
) -> np.ndarray:
    """Deals patients round-robin in group order, shuffled within each group.

    Fold sizes differ by at most one, and each group (e.g. cohort or event
    status) is spread as evenly as its size allows (a group smaller than
    ``n_folds`` misses some folds).

    Args:
        group: Group of each patient.
        rng: Random generator for the shuffle within each group.
        n_folds: Number of folds.

    Returns:
        The fold of each patient, in ``[0, n_folds)``.
    """
    order = np.lexsort((rng.random(len(group)), group))
    fold = np.empty(len(group), dtype=np.int64)
    fold[order] = np.arange(len(group)) % n_folds
    return fold


def write_synthetic_data(root: str, n_patients: int = 300, seed: int = 0) -> None:
    """Writes a synthetic dataset with a known survival signal to ``root``.

    Uses the layout of :class:`~pyhealth.datasets.TCGAMultimodalDataset`,
    including ``split_cases.csv`` (``case_id``, ``split``). Each embedding
    modality hides its own latent signal in a few high-variance directions, and
    the log hazard sums those signals with a cohort and an age effect. Survival and
    censoring times are exponential. Every modality should therefore beat
    C-index 0.5, and fusing them should beat each one alone.

    Args:
        root: Output directory; created if missing, files overwritten.
        n_patients: Number of synthetic TCGA cases.
        seed: Seed of the random generator.

    Raises:
        ValueError: If ``n_patients`` is smaller than the number of folds.
    """
    if n_patients < NUM_FOLDS:
        raise ValueError(f"n_patients must be at least {NUM_FOLDS}, got {n_patients}")

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
    pd.DataFrame({"case_id": case_ids, "split": _assign_folds(cohort, rng)}).to_csv(
        root_path / "split_cases.csv", index=False
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


def fit(
    train: SampleDataset,
    feature_keys: Sequence[str],
    config: HeadConfig,
    n_events: int,
) -> Trainer:
    """Trains one ``CoxSurvivalModel`` full-batch.

    The ridge penalty is the paper's, rescaled by :func:`ridge_weight_decay`.
    It applies to every head; only for the linear head does it equal the
    paper's ridge CoxPH.

    Args:
        train: Training part; its fitted processors define the features.
        feature_keys: Input fields of the model.
        config: Head and optimizer options.
        n_events: Observed events in ``train``, for the ridge rescaling.

    Returns:
        The trainer that holds the fitted model.
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
    return trainer


def fit_predict(
    train: SampleDataset,
    test: SampleDataset,
    feature_keys: Sequence[str],
    config: HeadConfig,
    n_events: int,
) -> tuple[Predictions, Predictions]:
    """Trains one model with :func:`fit` and scores both parts.

    Args:
        train: Training part.
        test: Test part, scored with the trained model.
        feature_keys: Input fields of the model.
        config: Head and optimizer options.
        n_events: Observed events in ``train``.

    Returns:
        Predictions on ``train`` (in-sample, as the paper feeds stage 2),
        then on ``test``.
    """
    trainer = fit(train, feature_keys, config, n_events)
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


def stage1(
    train: SampleDataset,
    test: SampleDataset,
    modalities: Sequence[str],
    config: HeadConfig,
) -> tuple[dict[str, Predictions], dict[str, Predictions]]:
    """Fits one unimodal Cox model per modality, the paper's stage 1.

    Args:
        train: Training part with every modality.
        test: Test part.
        modalities: Stage-1 modalities, one model each.
        config: Head of the stage-1 models.

    Returns:
        Predictions per modality on ``train`` (in-sample), then on ``test``.
    """
    n_events = _n_events(train)
    parts = {
        modality: fit_predict(train, test, [modality], config, n_events)
        for modality in modalities
    }
    return (
        {modality: train_test[0] for modality, train_test in parts.items()},
        {modality: train_test[1] for modality, train_test in parts.items()},
    )


def stage2(
    train_risks: Mapping[str, Predictions],
    test_risks: Mapping[str, Predictions],
    n_events: int,
) -> Fusion:
    """Fits the linear Cox model on z-scored stage-1 risks, as in the paper.

    Args:
        train_risks: Stage-1 risks on the training part, per modality.
        test_risks: Stage-1 risks on the test part, same modalities.
        n_events: Observed events in the training part.

    Returns:
        The fused test predictions and one hazard ratio per modality.

    Raises:
        ValueError: If ``train_risks`` and ``test_risks`` have different
            modalities or a different modality order.
    """
    if list(train_risks) != list(test_risks):
        raise ValueError(
            f"stage-2 modalities differ: train {list(train_risks)}, "
            f"test {list(test_risks)}"
        )

    fusion_train, fusion_test = fusion_datasets(train_risks, test_risks)
    trainer = fit(fusion_train, [FUSION_KEY], LINEAR, n_events)

    coef = trainer.model.head.weight.detach().cpu().numpy().reshape(-1)
    return Fusion(
        predictions=_predict(trainer, fusion_test),
        hazard_ratios=dict(zip(train_risks, np.exp(coef).tolist())),
    )


def out_of_fold_risks(
    dataset: TCGAMultimodalDataset,
    task: MultimodalSurvivalPrediction,
    train_risks: Mapping[str, Predictions],
    seed: int,
) -> dict[str, Predictions]:
    """Scores each training patient with stage-1 models that never saw it.

    The training patients are dealt into ``INNER_FOLDS`` folds, stratified
    by event. Each inner fold is scored by stage-1 models trained on the
    other inner folds. Every inner ``set_task`` call fits the processors on
    its inner training patients only; test patients take no part.

    Args:
        dataset: Dataset of the outer fold.
        task: Task of the outer fold.
        train_risks: In-sample stage-1 risks on the outer training part, per
            modality; they give the patients, their order and their labels.
        seed: Seed of the inner fold assignment.

    Returns:
        Out-of-fold stage-1 risks, in the order of ``train_risks``.
    """
    reference = next(iter(train_risks.values()))
    events = dict(zip(reference.patient_ids, reference.event))
    patient_ids = sorted(events)
    rng = np.random.default_rng(seed)
    inner_folds = dict(
        zip(
            patient_ids,
            _assign_folds(np.array([events[p] for p in patient_ids]), rng, INNER_FOLDS),
        )
    )

    risks: dict[str, dict[str, float]] = {modality: {} for modality in train_risks}
    for inner_fold in range(INNER_FOLDS):
        inner_train, held_out = dataset.set_task(
            task, split=FoldSplit(inner_folds, inner_fold)
        )
        _, held_out_risks = stage1(inner_train, held_out, list(train_risks), LINEAR)

        for modality, predictions in held_out_risks.items():
            risks[modality].update(zip(predictions.patient_ids, predictions.risk))

    return {
        modality: replace(
            predictions,
            risk=np.array([risks[modality][p] for p in predictions.patient_ids]),
        )
        for modality, predictions in train_risks.items()
    }


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
    predictions: Mapping[str, Predictions],
    cancer_types: Mapping[str, str],
    hazard_ratios: Mapping[str, dict[str, float]],
) -> FoldResult:
    """Computes the pooled and per-cohort C-index of each configuration.

    Args:
        predictions: Test predictions per configuration name.
        cancer_types: Patient ID to TCGA project.
        hazard_ratios: Stage-2 hazard ratios per fusion configuration.

    Returns:
        The fold's C-index per configuration, pooled and per cohort, and
        the hazard ratios.
    """
    return FoldResult(
        hazard_ratios=dict(hazard_ratios),
        c_index={
            name: concordance_index(p.duration, p.event, p.risk)
            for name, p in predictions.items()
        },
        cohort_c_index={
            name: _cohort_c_index(p, cancer_types) for name, p in predictions.items()
        },
    )


def make_task(
    components: Mapping[str, int | None], method: str = METHOD_PCA
) -> MultimodalSurvivalPrediction:
    """Builds the task with one reducer for every embedding modality.

    Args:
        components: Reduced size per embedding modality.
        method: Reduction method of every embedding modality.

    Returns:
        The task.
    """
    return MultimodalSurvivalPrediction(
        **{f"{m}_components": n for m, n in components.items()},
        **{f"{m}_method": method for m in components},
    )


def run_fold(
    dataset: TCGAMultimodalDataset,
    fold: int,
    components: Mapping[str, int | None],
    oof: bool = False,
    method: str = METHOD_PCA,
) -> FoldResult:
    """Trains and evaluates every configuration on one fold.

    Processors (z-score + reducer, one-hot) are fitted on the training
    patients only, through ``set_task(task, split=FoldSplit(...))``; so are
    those of the inner folds of "two-stage oof".

    Args:
        dataset: Dataset with a fold for at least one patient. Patients
            without a fold are left out.
        fold: Fold whose patients form the test set.
        components: Reduced size per embedding modality.
        oof: Whether to add "two-stage oof" (25 more stage-1 fits).
        method: Reduction method of every embedding modality.

    Returns:
        C-index per configuration (unimodal, fused, ablations), pooled over
        the fold's test patients and per cancer type, and the stage-2 hazard
        ratios of "two-stage" and "two-stage oof".
    """
    task = make_task(components, method)
    split = FoldSplit(dataset.get_fold_assignments(), fold)
    train, test = dataset.set_task(task, split=split)
    print(f"fold {fold} ({fold + 1}/{NUM_FOLDS}): samples built, training", flush=True)
    n_events = _n_events(train)

    train_risks, test_risks = stage1(train, test, task.modalities, LINEAR)
    fusions = {PAPER_FUSION: stage2(train_risks, test_risks, n_events)}
    if oof:
        fusions[OOF_FUSION] = stage2(
            out_of_fold_risks(dataset, task, train_risks, seed=fold),
            test_risks,
            n_events,
        )
    fusions["two-stage mlp"] = stage2(
        *stage1(train, test, task.modalities, MLP), n_events=n_events
    )

    predictions = {
        **test_risks,
        **{name: fusion.predictions for name, fusion in fusions.items()},
    }
    for name, config in JOINT_CONFIGS.items():
        _, predictions[name] = fit_predict(
            train, test, task.modalities, config, n_events
        )

    hazard_ratios = {
        name: fusions[name].hazard_ratios
        for name in HAZARD_RATIO_FUSIONS
        if name in fusions
    }
    return score(predictions, dataset.get_cancer_types(), hazard_ratios)


def run_reduction_fold(
    dataset: TCGAMultimodalDataset,
    fold: int,
    components: Mapping[str, int | None],
    method: str,
) -> dict[str, float]:
    """Scores the paper's pipeline on one fold with one embedding reducer.

    Args:
        dataset: Dataset with a fold for at least one patient.
        fold: Fold whose patients form the test set.
        components: Reduced size per embedding modality.
        method: Reduction method of every embedding modality.

    Returns:
        Test C-index of each embedding modality and of "two-stage".
    """
    task = make_task(components, method)
    train, test = dataset.set_task(
        task, split=FoldSplit(dataset.get_fold_assignments(), fold)
    )
    print(
        f"{method} fold {fold} ({fold + 1}/{NUM_FOLDS}): samples built, training",
        flush=True,
    )
    n_events = _n_events(train)

    train_risks, test_risks = stage1(train, test, task.modalities, LINEAR)
    fusion = stage2(train_risks, test_risks, n_events)
    predictions = {**test_risks, PAPER_FUSION: fusion.predictions}
    return {
        name: concordance_index(p.duration, p.event, p.risk)
        for name, p in predictions.items()
        if name in REDUCTION_CONFIGS
    }


def reduction_ablation(
    dataset: TCGAMultimodalDataset, components: Mapping[str, int | None]
) -> pd.DataFrame:
    """Runs :func:`run_reduction_fold` for every reducer on every fold.

    Args:
        dataset: Dataset with a fold for at least one patient.
        components: Reduced size per embedding modality, the same for every
            reducer.

    Returns:
        ``mean ± std`` (ddof=1, over folds) of the test C-index, one row per
        reducer, one column per configuration of ``REDUCTION_CONFIGS``.
    """
    folds = sorted(set(dataset.get_fold_assignments().values()))
    rows = {}
    for method in REDUCTION_METHODS:
        table = pd.DataFrame(
            [run_reduction_fold(dataset, f, components, method) for f in folds]
        )
        rows[method] = {
            config: f"{table[config].mean():.3f} ± {table[config].std():.3f}"
            for config in REDUCTION_CONFIGS
        }
    return pd.DataFrame(rows).T


def summarize(results: Sequence[FoldResult]) -> pd.DataFrame:
    """Returns mean and std (ddof=1) of the C-index over folds, per config."""
    table = pd.DataFrame([r.c_index for r in results])
    return pd.DataFrame({"mean": table.mean(), "std": table.std()})


def summarize_hazard_ratios(results: Sequence[FoldResult]) -> pd.DataFrame:
    """Returns the mean stage-2 hazard ratio over folds, per modality.

    Args:
        results: One result per fold.

    Returns:
        One row per modality, one column per fusion configuration.
    """
    names = results[0].hazard_ratios
    return pd.DataFrame(
        {
            name: pd.DataFrame([r.hazard_ratios[name] for r in results]).mean()
            for name in names
        }
    )


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


def load_dataset(
    root: str, cache_dir: str | None, num_workers: int
) -> TCGAMultimodalDataset:
    """Loads the dataset and warns once about patients without a fold.

    Patients without a fold in ``split_cases.csv`` are left out of every
    fold.

    Args:
        root: Dataset directory in the loader layout, with ``split_cases.csv``.
        cache_dir: PyHealth cache directory; None for PyHealth's default.
        num_workers: Worker processes for the dataset and task.

    Returns:
        The dataset.

    Raises:
        ValueError: If no patient has a fold.
    """
    dataset = TCGAMultimodalDataset(
        root=root, cache_dir=cache_dir, num_workers=num_workers
    )

    folds = dataset.get_fold_assignments()
    patients = set(dataset.unique_patient_ids)
    unassigned = len(patients - set(folds))
    if unassigned:
        logger.warning(
            "Skipping %d of %d patients without a split in split_cases.csv",
            unassigned,
            len(patients),
        )
    return dataset


def run(
    dataset: TCGAMultimodalDataset,
    components: Mapping[str, int | None],
    oof: bool = False,
    method: str = METHOD_PCA,
) -> list[FoldResult]:
    """Runs every fold.

    Args:
        dataset: Dataset from :func:`load_dataset`.
        components: Reduced size per embedding modality.
        oof: Whether to add "two-stage oof" to every fold.
        method: Reduction method of every embedding modality.

    Returns:
        One result per fold, in fold order.

    Raises:
        ValueError: If a fold has no training or test sample.
    """
    folds = sorted(set(dataset.get_fold_assignments().values()))
    return [run_fold(dataset, fold, components, oof, method) for fold in folds]


def _parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        help="dataset directory in the TCGAMultimodalDataset layout, with "
        "the authors' split_cases.csv (case_id, split); default: synthetic data "
        "in a temp dir",
    )
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument(
        "--n-components",
        type=int,
        help="reduced size of every embedding modality; default: the paper's "
        f"{PAPER_PCA_COMPONENTS['expr']}, or {SYNTHETIC_COMPONENTS} on "
        "synthetic data",
    )
    parser.add_argument(
        "--reduction",
        choices=REDUCTION_METHODS,
        default=METHOD_PCA,
        help="embedding reducer of every embedding modality; default: the paper's pca",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--oof",
        action="store_true",
        help='add the "two-stage oof" ablation: stage 2 on out-of-fold stage-1 '
        "risks (25 more stage-1 fits per fold)",
    )
    parser.add_argument(
        "--reduction-ablation",
        action="store_true",
        help="run only the reducer ablation: the paper's pipeline with each of "
        f"{', '.join(REDUCTION_METHODS)} at the same size (ignores --reduction "
        "and --oof)",
    )
    return parser.parse_args()


def _print_results(results: Sequence[FoldResult], oof: bool) -> None:
    """Prints the pooled and per-cohort C-index and the hazard ratios."""
    cohort_configs = OOF_COHORT_CONFIGS if oof else PAPER_CONFIGS

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(f"\nC-index over {len(results)} folds:")
        print(summarize(results).round(3).to_string())
        print("\nPer-cohort C-index, mean ± std (folds with a comparable pair):")
        print(summarize_cohorts(results, cohort_configs).to_string())
        print("\nStage-2 hazard ratios exp(coef), mean over folds:")
        print(summarize_hazard_ratios(results).round(3).to_string())


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
            components = dict.fromkeys(EMBEDDING_MODALITIES, SYNTHETIC_COMPONENTS)
        else:
            root, cache_dir = args.root, args.cache_dir
            components = dict(PAPER_PCA_COMPONENTS)

        if args.n_components is not None:
            components = dict.fromkeys(EMBEDDING_MODALITIES, args.n_components)

        dataset = load_dataset(root, cache_dir, args.num_workers)
        if args.reduction_ablation:
            table = reduction_ablation(dataset, components)
            print(f"\nC-index per reducer, mean ± std over folds, {components}:")
            print(table.to_string())
        else:
            _print_results(run(dataset, components, args.oof, args.reduction), args.oof)
    print(f"\nRuntime: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
