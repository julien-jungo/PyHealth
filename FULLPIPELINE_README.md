# Local testing and replication tools (not part of the PR)

Two scripts for checking our PyHealth code yourself:

- `replicate_paper.py` runs the paper's full train/test pipeline (Step 1 Cox per modality, Step 2 fusion Cox, 5-fold C-index) on top of `TCGAMultimodalDataset` and `MultimodalCancerSurvival`, with the paper's in-sample fusion and our out-of-fold fusion side by side.
- `convert_h5_embeddings.py` turns the authors' embedding files (`expr.h5`, `hist.h5`, `summ.h5`) into the dataset's `.npy` layout.

They use `scikit-survival` and `h5py`, which PyHealth does not depend on, so they stay out of the PR.

## Part 1: set up and run the unit tests

Needs Python 3.12 or 3.13 (PyHealth requires `>=3.12,<3.14`). Everything below was tested on Linux. On Windows, WSL2 (Ubuntu) or Colab is the safest route; native Windows is untested.

    git clone https://github.com/sunlabuiuc/PyHealth.git
    cd PyHealth
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -e .
    git apply /path/to/dataset_pr_draft/tcga_multimodal_draft.patch

If `git apply` fails, upstream moved: copy the 7 new files from the draft bundle and redo the 4 one-line edits listed in its README.

Add the teammate's model (get the files from them):

    pyhealth/models/cox_survival.py        # delete the line `self.mode = None`
    tests/core/test_cox_survival.py
    # and add to pyhealth/models/__init__.py:
    from .cox_survival import CoxSurvivalModel

Run the tests and style checks:

    pip install pytest flake8 black
    python -m pytest tests/core/test_tcga_multimodal.py tests/core/test_cox_survival.py -v
    python -m flake8 --max-line-length 88 pyhealth/datasets/tcga_multimodal.py pyhealth/tasks/multimodal_cancer_survival.py
    python -m black --check -l 88 pyhealth/datasets/tcga_multimodal.py pyhealth/tasks/multimodal_cancer_survival.py

Expected: 24 dataset/task tests and 14 model tests pass (one-time setup about 3 s, each test < 0.1 s).

## Part 2: see every pipeline step on synthetic data (about 30 s)

    pip install scikit-survival
    python replicate_paper.py --synthetic

400 fake patients, all five modalities, 5 folds, in-sample vs. out-of-fold fusion. The numbers mean nothing about real patients; they show the pipeline runs. In the fake data, demographics and reports carry no survival signal, so a working out-of-fold step should give them lower fusion weights than the in-sample step does.

## Part 3: replicate the paper's clinical baselines on real data (about 2 min, CPU)

Make a data folder from the authors' repo (https://github.com/StevenSong/multimodal-cancer-modeling):

    mkdir tcga_data
    cp multimodal-cancer-modeling/data/clinical.csv tcga_data/
    cp multimodal-cancer-modeling/results/split_cases.csv tcga_data/
    python replicate_paper.py --root tcga_data --modalities demographics cancer_type

Expected output (verified Nov 12, 2026; scikit-survival backend):

    patients in clinical.csv: 8061 | task samples: 8061 | with a fold: 7982
    kept for cross-validation (sample AND fold): 7982
        model     variant  mean   std  paper
         Demo    unimodal 0.630 0.012  0.630
         Canc    unimodal 0.737 0.011  0.737
    Canc-Demo   in-sample 0.747 0.012  0.747
    Canc-Demo out-of-fold 0.747 0.012    NaN

The first three rows match the paper's Tables 1-2 exactly. The 79 patients without a fold are in clinical.csv but not in the paper's cohort.

### Same thing with the team's CoxSurvivalModel

    python replicate_paper.py --root tcga_data --modalities demographics cancer_type \
        --backend pyhealth --device cuda --folds 0 --inner-folds 0

On fold 0 it matched scikit-survival to within 0.0001 C-index (Demo 0.6360 vs 0.6361, Canc 0.7408, Canc-Demo 0.7556) and 0.003 in fusion hazard ratio. It is slow on CPU: the Cox loss compares all ~6,400 training patients with each other, about 0.7 s per epoch, so 300 epochs is ~3.5 min per model. A GPU should be far faster (not tested here). Use `--folds 0` for quick checks.

## Part 4: the full five-modality replication (needs embeddings)

1. The data teammate runs the authors' `embed/` scripts to get `expr.h5` (BulkRNABert), `hist.h5` (UNI2-h), and `summ.h5` (BioMistral on the Llama summaries; this is the paper's main text model).
2. Convert each file into the data folder:

        pip install h5py
        python convert_h5_embeddings.py --h5 expr.h5 --modality expression --out-root tcga_data
        python convert_h5_embeddings.py --h5 hist.h5 --modality histology --out-root tcga_data
        python convert_h5_embeddings.py --h5 summ.h5 --modality report --out-root tcga_data

   Every vector becomes one `.npy`; nothing is averaged or skipped, and the script stops on anything unexpected.
3. Run everything:

        python replicate_paper.py --root tcga_data

   Compare against the `paper` column: Expr 0.750, Hist 0.754, Text 0.752 (Table 1, PCA 256); Hist-Text 0.780, Expr-Hist-Text 0.788, Demo-Expr-Hist-Text 0.795, all five 0.793 (Table 2). Check Text alone first: if it is far from 0.752, the embeddings differ from the authors' (see the vLLM pooling note in `claude/proposal-verified-findings.md`), not the pipeline.

Runtime for Part 4 is untested here. The default run fits 150 Step 1 models (5 folds x 5 modalities x [1 + 5 inner]) plus the fusion models; the paper reports 1,085 models in 17 minutes (Appendix E).

## Options

| Flag | Meaning |
|---|---|
| `--modalities ...` | any of demographics, cancer_type, expression, histology, report |
| `--backend sksurv\|pyhealth` | paper's exact Cox model, or the team's CoxSurvivalModel |
| `--inner-folds K` | inner split for out-of-fold fusion; 0 skips it |
| `--folds 0 1` | run only these outer folds |
| `--pca-dim N` | PCA size for embeddings (paper: 256) |
| `--epochs`, `--lr`, `--device` | CoxSurvivalModel training (defaults 300, 0.05, cpu) |
| `--out file.csv` | per-fold C-index; hazard ratios go to `file.hazard_ratios.csv` |
