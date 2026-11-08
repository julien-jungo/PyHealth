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
    TODO: data, folds, PCA dims, optimizer, epochs, ridge weight_decay mapping.
    Train full-batch (batch_size = training fold size), so the Cox risk sets
    match the paper's full partial likelihood.

Ablations (novel relative to the paper):
    TODO: two-stage (paper) vs. joint fusion; linear vs. MLP head;
    hidden_dim / dropout / learning-rate sweep.

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

import numpy as np

from pyhealth.datasets import Split


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


def write_synthetic_data(root: str, n_patients: int = 60, seed: int = 0) -> None:
    """Writes a small synthetic dataset with a known survival signal to root."""
    # TODO: same layout as TCGAMultimodalDataset expects; risk is a linear
    # function of a few embedding dims, exponential survival times, random
    # censoring, so a working pipeline beats C-index 0.5.
    raise NotImplementedError


def run_fold(root: str, fold: int) -> dict[str, float]:
    """Trains and evaluates every configuration on one fold."""
    # TODO: stage 1, stage 2, ablations; return C-index per
    # configuration (and per cohort).
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
