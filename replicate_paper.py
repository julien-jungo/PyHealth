"""Local replication harness for Song et al. (ML4H 2025). NOT part of the PR.

Runs the paper's train/test pipeline on top of our PyHealth code:

    TCGAMultimodalDataset -> MultimodalCancerSurvival -> per-fold features
    -> Step 1: one Cox model per modality -> risk scores
    -> Step 2: Cox "fusion" model over the z-scored Step 1 risk scores
    -> test C-index per fold, averaged over the paper's 5 folds.

Step 1 follows the authors' notebook (experiments/1-survival-experiments.ipynb,
cells 19-20): embeddings are z-scored and PCA-reduced (random_state=42) with
statistics from the training fold only; one-hot modalities are used as is;
Cox ridge alpha = 0.1. Step 2 z-scores each modality's risk scores with
training statistics and fits another Cox model (alpha = 0.1).

Step 2 is run two ways:
    in-sample    the paper: training risk scores come from Step 1 models that
                 were trained on those same patients.
    out-of-fold  our planned fix: training risk scores come from Step 1
                 models that never saw the patient (inner K-fold split drawn
                 only from the training fold). Test risk scores are identical
                 in both variants, so only the fusion weights differ.

Cox backends:
    sksurv    scikit-survival CoxPHSurvivalAnalysis(alpha=0.1), the paper's
              exact model. Needs `pip install scikit-survival`.
    pyhealth  the team's CoxSurvivalModel (linear head), trained full-batch
              with weight_decay = alpha / n_events.

Usage:
    python replicate_paper.py --synthetic                       # mechanics only
    python replicate_paper.py --root DATA --modalities demographics cancer_type
    python replicate_paper.py --root DATA                       # all five
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pyhealth.datasets import TCGAMultimodalDataset
from pyhealth.metrics import concordance_index
from pyhealth.tasks import MultimodalCancerSurvival
from pyhealth.tasks.multimodal_cancer_survival import (
    ALL_MODALITIES,
    EMBEDDING_MODALITIES,
)

ALPHA = 0.1  # paper Sec. 2.4: CoxPH ridge penalty
PCA_SEED = 42  # authors' notebook
SHORT = {
    "demographics": "Demo",
    "cancer_type": "Canc",
    "expression": "Expr",
    "histology": "Hist",
    "report": "Text",
}
# Paper Table 1 (PCA 256 for embeddings) and Table 2.
PAPER = {
    ("demographics",): 0.630,
    ("cancer_type",): 0.737,
    ("expression",): 0.750,
    ("histology",): 0.754,
    ("report",): 0.752,
    ("cancer_type", "demographics"): 0.747,
    ("histology", "report"): 0.780,
    ("expression", "histology", "report"): 0.788,
    ("demographics", "expression", "histology", "report"): 0.795,
    ("cancer_type", "demographics", "expression", "histology", "report"): 0.793,
}

Arrays = Tuple[List[str], Dict[str, np.ndarray], np.ndarray, np.ndarray]
FitFn = Callable[[np.ndarray, np.ndarray, np.ndarray], "CoxFit"]


class CoxFit:
    """A fitted Cox model: risk scores for new rows and coefficients."""

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray], coef: np.ndarray):
        self.predict = predict
        self.coef = coef


def sksurv_fit(x: np.ndarray, t: np.ndarray, e: np.ndarray) -> CoxFit:
    """Fits the paper's model: scikit-survival CoxPH with ridge alpha 0.1."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv

    model = CoxPHSurvivalAnalysis(alpha=ALPHA).fit(x, Surv.from_arrays(e > 0, t))
    return CoxFit(model.predict, model.coef_)


def make_pyhealth_fit(epochs: int, lr: float, device: str) -> FitFn:
    """Returns a fit function that trains the team's CoxSurvivalModel."""
    import torch

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import CoxSurvivalModel
    from pyhealth.trainer import Trainer

    def fit(x: np.ndarray, t: np.ndarray, e: np.ndarray) -> CoxFit:
        dataset = create_sample_dataset(
            samples=[
                {"patient_id": str(i), "x": row.tolist(), "duration": [d], "event": [v]}
                for i, (row, d, v) in enumerate(zip(x, t, e))
            ],
            input_schema={"x": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
        )
        torch.manual_seed(0)
        model = CoxSurvivalModel(dataset, feature_keys=["x"])
        Trainer(model=model, enable_logging=False, device=device).train(
            get_dataloader(dataset, batch_size=len(dataset)),  # full batch
            epochs=epochs,
            optimizer_params={"lr": lr},
            weight_decay=ALPHA / max(int(e.sum()), 1),
        )
        model.eval()

        def predict(rows: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                features = torch.tensor(rows, dtype=torch.float32, device=device)
                return model(x=features)["risk"].cpu().numpy()

        return CoxFit(predict, model.head.weight.detach().cpu().numpy().ravel())

    return fit


def load_arrays(root: str, modalities: Sequence[str], cache_dir: str) -> Arrays:
    """Builds per-modality feature matrices with our dataset and task.

    Prints how many patients are kept at each step, so nothing disappears
    silently. Patients are kept if the task produces a sample (one clinical
    record, valid label, every requested modality) and the fold file assigns
    them a fold.
    """
    tables = [m for m in modalities if m in EMBEDDING_MODALITIES]
    dataset = TCGAMultimodalDataset(root=root, tables=tables, cache_dir=cache_dir)
    folds = dataset.get_fold_assignments()
    samples = dataset.set_task(MultimodalCancerSurvival(modalities=modalities))
    print(
        f"patients in clinical.csv: {len(dataset.unique_patient_ids)} | "
        f"task samples: {len(samples)} | with a fold: {len(folds)}"
    )
    ids, features, durations, events = [], {m: [] for m in modalities}, [], []
    for sample in samples:
        if sample["patient_id"] not in folds:
            continue
        ids.append(sample["patient_id"])
        for m in modalities:
            features[m].append(sample[m].numpy().reshape(-1))
        durations.append(float(sample["duration"][0]))
        events.append(float(sample["event"][0]))
    print(f"kept for cross-validation (sample AND fold): {len(ids)}")
    matrices = {m: np.stack(rows).astype(np.float64) for m, rows in features.items()}
    return ids, matrices, np.array(durations), np.array(events)


def step1(
    modality: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    pca_dim: int,
    fit: FitFn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits one unimodal Cox model; returns training and test risk scores.

    Embeddings are z-scored and PCA-reduced with training statistics only;
    one-hot modalities are not transformed (paper Sec. 2.4).
    """
    if modality in EMBEDDING_MODALITIES:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        n_components = min(pca_dim, *x_train.shape)
        pca = PCA(n_components=n_components, random_state=PCA_SEED).fit(x_train)
        x_train, x_test = pca.transform(x_train), pca.transform(x_test)
    model = fit(x_train, t, e)
    return model.predict(x_train), model.predict(x_test)


def out_of_fold_scores(
    modality: str,
    x_train: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    pca_dim: int,
    fit: FitFn,
    inner_folds: int,
) -> np.ndarray:
    """Scores every training patient with a Step 1 model that never saw them.

    The inner split uses only training-fold patients, stratified by event so
    each inner part keeps a similar death rate. Test patients are never used.
    """
    scores = np.empty(len(t))
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=0)
    for inner_train, inner_held in splitter.split(x_train, e):
        _, scores[inner_held] = step1(
            modality,
            x_train[inner_train],
            x_train[inner_held],
            t[inner_train],
            e[inner_train],
            pca_dim,
            fit,
        )
    return scores


def step2(
    combo: Sequence[str],
    train_risks: Dict[str, np.ndarray],
    test_risks: Dict[str, np.ndarray],
    t: np.ndarray,
    e: np.ndarray,
    fit: FitFn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits the fusion Cox model on z-scored risk scores; returns test risk
    and the fusion hazard ratios exp(coef), one per modality."""
    scaler = StandardScaler().fit(np.column_stack([train_risks[m] for m in combo]))
    x_train = scaler.transform(np.column_stack([train_risks[m] for m in combo]))
    x_test = scaler.transform(np.column_stack([test_risks[m] for m in combo]))
    model = fit(x_train, t, e)
    return model.predict(x_test), np.exp(model.coef)


def fusion_combos(modalities: Sequence[str]) -> List[Tuple[str, ...]]:
    """The paper's Table 2 combinations available here, plus all modalities."""
    available = set(modalities)
    combos = [c for c in PAPER if len(c) > 1 and set(c) <= available]
    full = tuple(sorted(modalities))
    if len(full) > 1 and full not in combos:
        combos.append(full)
    return combos


def run(
    ids: List[str],
    x: Dict[str, np.ndarray],
    t: np.ndarray,
    e: np.ndarray,
    folds: Dict[str, int],
    fit: FitFn,
    pca_dim: int,
    inner_folds: int,
    only_folds: Sequence[int] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs Step 1 and both Step 2 variants on each outer fold (or only the
    folds in ``only_folds``, for quick checks)."""
    fold = np.array([folds[p] for p in ids])
    modalities = list(x)
    combos = fusion_combos(modalities)
    rows, ratios = [], []
    for k in sorted(set(fold) & set(only_folds or fold)):
        tr, te = fold != k, fold == k
        start = time.time()
        in_sample, oof, test = {}, {}, {}
        for m in modalities:
            in_sample[m], test[m] = step1(
                m, x[m][tr], x[m][te], t[tr], e[tr], pca_dim, fit
            )
            rows.append(
                {
                    "fold": k,
                    "model": SHORT[m],
                    "variant": "unimodal",
                    "c_index": concordance_index(t[te], e[te], test[m]),
                }
            )
            if combos and inner_folds > 1:
                oof[m] = out_of_fold_scores(
                    m, x[m][tr], t[tr], e[tr], pca_dim, fit, inner_folds
                )
        for combo in combos:
            name = "-".join(SHORT[m] for m in combo)
            variants = {"in-sample": in_sample, "out-of-fold": oof}
            for variant, train_risks in variants.items():
                if not train_risks:
                    continue
                risk, hr = step2(combo, train_risks, test, t[tr], e[tr], fit)
                rows.append(
                    {
                        "fold": k,
                        "model": name,
                        "variant": variant,
                        "c_index": concordance_index(t[te], e[te], risk),
                    }
                )
                ratios += [
                    {
                        "fold": k,
                        "model": name,
                        "variant": variant,
                        "modality": SHORT[m],
                        "hazard_ratio": h,
                    }
                    for m, h in zip(combo, hr)
                ]
        print(f"fold {k}: {int(te.sum())} test patients, {time.time() - start:.1f}s")
    return pd.DataFrame(rows), pd.DataFrame(ratios)


def summarize(results: pd.DataFrame, ratios: pd.DataFrame, real: bool) -> None:
    """Prints mean ± std C-index (next to the paper's, for real data) and the
    fusion hazard ratios."""
    paper = {"-".join(SHORT[m] for m in k): v for k, v in PAPER.items()} if real else {}
    table = (
        results.groupby(["model", "variant"], sort=False)["c_index"]
        .agg(["mean", "std"])
        .reset_index()
    )
    table["paper"] = [
        paper.get(m, np.nan) if v in ("unimodal", "in-sample") else np.nan
        for m, v in zip(table["model"], table["variant"])
    ]
    print("\nTest C-index over folds (paper: Tables 1-2):")
    print(table.to_string(index=False, float_format="%.3f"))
    if not ratios.empty:
        hr = ratios.pivot_table(
            index=["model", "modality"], columns="variant", values="hazard_ratio"
        )
        print("\nFusion hazard ratios, mean over folds (paper: Table 8):")
        print(hr.to_string(float_format="%.3f"))


def write_synthetic_root(root: Path, n_patients: int = 400, seed: int = 0) -> None:
    """Writes fake data in the dataset's layout, with a survival signal in
    cancer type, expression, and histology (none in reports)."""
    rng = np.random.default_rng(seed)
    ids = [f"TCGA-SY-{i:04d}" for i in range(n_patients)]
    project = rng.choice(["TCGA-BRCA", "TCGA-LUAD", "TCGA-GBM"], n_patients)
    signal = {m: rng.normal(size=n_patients) for m in EMBEDDING_MODALITIES}
    log_hazard = 1.0 * (project == "TCGA-GBM") + 0.7 * signal["expression"]
    log_hazard += 0.7 * signal["histology"]
    death = rng.exponential(1500 * np.exp(-log_hazard))
    censor = rng.uniform(0, 4000, n_patients)
    dead, days = death <= censor, np.round(np.minimum(death, censor))
    pd.DataFrame(
        {
            "case_id": ids,
            "project": project,
            "sex": rng.choice(["female", "male"], n_patients),
            "age": rng.uniform(25, 90, n_patients),
            "race": rng.choice(["white", "asian"], n_patients),
            "ethnicity": "not hispanic or latino",
            "vital_status": np.where(dead, "Dead", "Alive"),
            "days_to_death": np.where(dead, days, np.nan),
            "days_to_last_follow_up": days,
        }
    ).to_csv(root / "clinical.csv", index=False)
    pd.DataFrame({"case_id": ids, "split": np.arange(n_patients) % 5}).to_csv(
        root / "split_cases.csv", index=False
    )
    for m in EMBEDDING_MODALITIES:
        direction = rng.normal(size=16)
        for i, pid in enumerate(ids):
            folder = root / "embeddings" / m / pid
            folder.mkdir(parents=True)
            vector = 3 * signal[m][i] * direction / np.linalg.norm(direction)
            np.save(folder / "file.npy", vector + rng.normal(size=16))


def main() -> None:
    """Parses arguments and runs the replication."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="folder with clinical.csv, split_cases.csv")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--modalities", nargs="+", default=list(ALL_MODALITIES))
    parser.add_argument("--backend", choices=["sksurv", "pyhealth"], default="sksurv")
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--inner-folds", type=int, default=5, help="0 = skip OOF")
    parser.add_argument("--folds", type=int, nargs="*", default=[], help="subset")
    parser.add_argument("--epochs", type=int, default=300, help="pyhealth backend")
    parser.add_argument("--lr", type=float, default=0.05, help="pyhealth backend")
    parser.add_argument("--device", default="cpu", help="pyhealth backend")
    parser.add_argument("--out", default="replication_results.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    fit = (
        sksurv_fit
        if args.backend == "sksurv"
        else make_pyhealth_fit(args.epochs, args.lr, args.device)
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = args.root
        if args.synthetic:
            root = str(Path(tmp) / "data")
            Path(root).mkdir()
            write_synthetic_root(Path(root))
        cache = str(Path(tmp) / "cache")
        ids, x, t, e = load_arrays(root, args.modalities, cache)
        folds = TCGAMultimodalDataset(
            root=root, tables=[], cache_dir=cache
        ).get_fold_assignments()
        results, ratios = run(
            ids, x, t, e, folds, fit, args.pca_dim, args.inner_folds, args.folds
        )
    results.to_csv(args.out, index=False)
    ratios.to_csv(Path(args.out).with_suffix(".hazard_ratios.csv"), index=False)
    summarize(results, ratios, real=not args.synthetic)
    print(f"\nPer-fold results: {args.out}")


if __name__ == "__main__":
    main()
