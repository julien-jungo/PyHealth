# CoxSurvivalModel on TCGA multimodal survival: results

Results of `examples/tcga_survival_cox_survival.py` for the two-stage late
fusion of Song et al. ([Multimodal Cancer Modeling in the Age of Foundation
Model Embeddings](https://proceedings.mlr.press/v297/song26a.html), ML4H 2025;
preprint: [arXiv:2505.07683](https://arxiv.org/abs/2505.07683)) and its
ablations. For real TCGA data, see [Run on real data](#run-on-real-data).

## Experimental setup

- **Data:** `--root` in the TCGAMultimodalDataset layout, or by default 300
  synthetic patients in 3 cohorts (`write_synthetic_data`, temp dir).
- **Folds:** the paper's 5 published CV folds (the `split` column of the
  authors' `results/split_cases.csv`); fold `k` is the test set, the other
  folds train, with no validation set, as in the paper. Patients without a
  fold (79 of the 8,061 in the authors' full `clinical.csv`, so 7,982 remain)
  are left out with one warning that gives the count.
- **Features:** expr/hist/text z-scored and PCA-reduced
  (EmbeddingReductionProcessor) to 256 components each, the paper's setting; 8
  on synthetic data (32-dim embeddings). Set them with `--n-components` and
  `--reduction`. demo and canc are one-hot (multi_hot), unscaled.
- **Optimizer:** full-batch Adam, so the Cox risk sets match the paper's full
  partial likelihood; lr 1e-2, seed 0. Steps: 3,000 for every linear fit
  (stage 1 and stage 2), 10,000 for "joint", 300 for MLP heads.
- **Ridge:** the paper's `CoxPHSurvivalAnalysis(alpha=0.1)` becomes
  `weight_decay = ridge_weight_decay(0.1, n_events)` (`n_events` of the
  training fold). MLP heads use the same decay, where it is not a ridge
  penalty.
- **Fusion (paper, "two-stage"):** one linear model per modality; their
  in-sample training risks are z-scored with training statistics (a `"zscore"`
  TensorProcessor) and fed to a second linear model.
- **Metric:** Harrell's C-index on the fold's test patients, pooled and per
  cancer type, then mean ± std (ddof=1) over the 5 folds.

### Convergence

On real data (folds 0 and 1), against the exact Newton optimum of the same
objective, the unimodal and stage-2 fits at 3,000 steps are at most 5.2e-4
above the optimal loss and 3.4e-4 off its test C-index. At 300 steps, the expr
fit is 0.03 above the optimal loss and up to 0.017 off its C-index. "joint"
(about 800 inputs) is 0.008 above at 3,000 steps and 2e-4 at 10,000. MLP heads
are not trained to convergence: 300 steps act as early stopping.

## Ablations

All are novel relative to the paper.

- **Two-stage vs. joint fusion:** "joint" is one linear model over all five
  modalities, trained end to end.
- **In-sample vs. out-of-fold stage-1 risks:** "two-stage oof" trains stage 2
  on out-of-fold stage-1 risks from 5 inner folds (stratified by event,
  processors refitted per inner fold); test risks come from the full-fold
  models. The paper uses in-sample risks. Add it with `--oof`.
- **Linear vs. MLP head:** "two-stage mlp" uses MLP heads (64 hidden units) in
  stage 1, with the linear stage 2; "joint mlp" is one MLP over all
  modalities.
- **Hyperparameter sweep** on "joint mlp", one change at a time: hidden_dim 16
  ("h=16"), dropout 0.3 ("p=0.3"), learning rate 1e-3 ("lr=1e-3").
- **Embedding reducer:** after the z-score, one of `pca` (the paper),
  `kernel_pca` (RBF kernel), `fast_ica` (unit-variance whitening) or
  `random_projection` (Gaussian), seeded with 42, at the same size. It is
  scored on the three embedding models and "two-stage". `--reduction-ablation`
  runs only this ablation.

## Synthetic data

300 patients, 5 folds; about 100 s on an 8-core CPU by default, and about
265 s with `--oof` (25 inner `set_task` calls and 125 more stage-1 fits).

| Configuration     | C-index       |
| ----------------- | ------------- |
| demo              | 0.529 ± 0.031 |
| canc              | 0.589 ± 0.038 |
| expr              | 0.669 ± 0.037 |
| hist              | 0.584 ± 0.056 |
| text              | 0.681 ± 0.036 |
| two-stage (paper) | 0.793 ± 0.033 |
| two-stage oof     | 0.782 ± 0.015 |
| two-stage mlp     | 0.649 ± 0.057 |
| joint             | 0.777 ± 0.032 |
| joint mlp         | 0.773 ± 0.020 |
| joint mlp h=16    | 0.751 ± 0.027 |
| joint mlp p=0.3   | 0.783 ± 0.030 |
| joint mlp lr=1e-3 | 0.780 ± 0.016 |

Findings:

- Two-stage fusion (0.793) beats the best single modality (text, 0.681) by
  0.11. Joint fusion (0.777) is within one std; on a second draw (seed 1, 200
  patients, 4 components) joint wins, 0.797 vs. 0.776.
- Out-of-fold risks do not help (0.782, within one std): with 8 components the
  stage-1 models overfit little.
- MLP heads in stage 1 hurt fusion (0.649): their in-sample risks overfit. On
  joint fusion the MLP head (0.773) is no better than the linear head, as the
  synthetic hazard is linear; the sweep rows are within about one std.

### Embedding reducer

`--reduction-ablation` on the same synthetic data and seed (8 components per
modality), about 200 s on an 8-core CPU; the `pca` row equals the main table.

| Reducer           | expr          | hist          | text          | two-stage     |
| ----------------- | ------------- | ------------- | ------------- | ------------- |
| pca (paper)       | 0.669 ± 0.037 | 0.584 ± 0.056 | 0.681 ± 0.036 | 0.793 ± 0.033 |
| kernel_pca        | 0.662 ± 0.027 | 0.579 ± 0.055 | 0.672 ± 0.024 | 0.762 ± 0.033 |
| fast_ica          | 0.669 ± 0.037 | 0.584 ± 0.056 | 0.681 ± 0.036 | 0.793 ± 0.033 |
| random_projection | 0.653 ± 0.058 | 0.600 ± 0.073 | 0.640 ± 0.021 | 0.738 ± 0.045 |

Findings:

- FastICA equals PCA to three decimals: its sources are an invertible linear
  map of the PCA scores (also without convergence in the 15 fits, default
  `max_iter` 200), and the ridge is weak (`weight_decay` about 1e-3).
- Kernel PCA (RBF) is 0.005 to 0.009 below PCA per modality and 0.031 on
  fusion: the synthetic signal is linear.
- The random projection keeps 8 random directions, not the high-variance ones:
  text drops by 0.041 and fusion by 0.055; hist is higher (0.600 vs. 0.584,
  within one std).
- The other differences are within about one std. This data favors PCA by
  design (the signal lies in the high-variance directions).

## Run on real data

PyHealth ships no TCGA data. You make the input folder with the public
[tcga-multimodal-embeddings][emb] repository, then run this example on it.

### 1. Make the input folder

Follow the sections [Quick start][emb-quick] and [PyHealth input
folder][emb-export] of that repository's README. Keep the default cohort of
`select_cohort.py` (10 TCGA cohorts chosen for small UNI2-h feature downloads
(at most 10 GB each), 873 patients) to reproduce the results below. Embedding
generation needs a GPU for `expr` and `text`, and your own approved Hugging
Face access for `hist`. The export writes this layout; `--root` must point
to it:

```
<root>/
    clinical.csv        # one row per case in split_cases.csv
    split_cases.csv     # case_id, split (the paper's CV fold 0-4)
    embeddings/
        {expr,hist,text}/<case_id>/<file_id>.npy  # one float32 vector
```

Sources and licences are in [Data access and licences][emb-licences]. The
clinical data and folds come from the authors' repository,
[StevenSong/multimodal-cancer-modeling][authors]. The `hist` (UNI2-h)
embeddings may not be shared: always generate your own.

[emb]: https://github.com/YaseenM2/tcga-multimodal-embeddings
[emb-quick]: https://github.com/YaseenM2/tcga-multimodal-embeddings#quick-start
[emb-export]: https://github.com/YaseenM2/tcga-multimodal-embeddings#pyhealth-input-folder
[emb-licences]: https://github.com/YaseenM2/tcga-multimodal-embeddings#data-access-and-licences
[authors]: https://github.com/StevenSong/multimodal-cancer-modeling

### 2. Run the example

From the root of this PyHealth repository, in a new virtual environment
(Python 3.12 or 3.13), not the one of the embeddings repository:

```bash
python3.12 -m venv .venv && . .venv/bin/activate  # uv: uv venv -p 3.12
pip install -e .  # or, with uv: uv pip install -e .
python examples/tcga_survival_cox_survival.py --root /path/to/pyhealth_root \
    --cache-dir /path/to/pyhealth_cache --num-workers 4
```

- `--root`: the absolute path of `pyhealth_root` (Quick start step 6, the
  `--output` of `export_pyhealth.py`). It must be writable: the loader writes
  its `tcga_multimodal_*-pyhealth.csv` index files into it.
- `--cache-dir`: a new folder, or the one of an earlier run to reuse its
  cache. Optional; see the cache note below.
- Add `--oof` to also run the "two-stage oof" ablation.
- Run `--reduction-ablation` separately: it runs only the reducer ablation and
  ignores `--reduction` and `--oof`.

Each fold prints `fold 0 (1/5): samples built, training`, where 0 is the
`split` value (prefixed by the reducer with `--reduction-ablation`); the first
one after about 30 s. PyHealth's `Worker` and `Rank` lines repeat per worker,
at the first task transform and at each fold's sample build. The macOS torch
warning `Redirects are currently not supported` is harmless. So is
`Ignoring token ... not seen in fit`: a demographic bin (e.g. `age:(0, 20]`)
occurs only in the test fold. At the end it prints the pooled, per-cohort and
hazard-ratio tables below and the runtime. Runtimes on an 8-core CPU with
`--num-workers 4`: about 270 s by default, 1,460 s with `--oof` and 1,240 s
with `--reduction-ablation`.

After a run that did not finish (also one you stopped on purpose), delete the
cache: PyHealth's `set_task` can leave a partial sample cache behind. With
`--cache-dir`, the cache is a subfolder `<cache-dir>/<id>`. Without it, a
warning at the start names the folder.

## Real data

10 TCGA cohorts chosen for small UNI2-h feature downloads (at most 10 GB each),
873 patients, PCA 256, 5 folds; runtimes are in "Run on real data". In the
`--oof` run, the 25 inner `set_task` caches were cold and the outer ones
warm; a warm rerun was not measured (it still has the 125 extra stage-1
fits). The paper column is the paper's 5-fold C-index on all 32 cohorts
(Tables 1 and 2, PCA 256), so it is not a like-for-like target.

| Configuration     | C-index       | Paper |
| ----------------- | ------------- | ----- |
| demo              | 0.656 ± 0.047 | 0.630 |
| canc              | 0.709 ± 0.016 | 0.737 |
| expr              | 0.660 ± 0.021 | 0.750 |
| hist              | 0.664 ± 0.037 | 0.754 |
| text              | 0.661 ± 0.060 | 0.752 |
| two-stage (paper) | 0.711 ± 0.030 | 0.793 |
| two-stage oof     | 0.739 ± 0.023 |       |
| two-stage mlp     | 0.726 ± 0.015 |       |
| joint             | 0.651 ± 0.038 |       |
| joint mlp         | 0.695 ± 0.031 |       |
| joint mlp h=16    | 0.693 ± 0.023 |       |
| joint mlp p=0.3   | 0.705 ± 0.021 |       |
| joint mlp lr=1e-3 | 0.689 ± 0.029 |       |

Per-cohort C-index, two-stage fusion vs. the best unimodal model of
demo/expr/hist/text (mean over the folds with a comparable pair; DLBC has one
in 4 folds). "Test" is the mean test-fold size, deaths in parentheses. The
paper's columns are from its Table 17 (same folds): models trained on all 32
cohorts, scored on each cohort's test cases; it does not report DLBC.

| Cohort    | Test        | Best uni   | Two-stage | Two-stage oof | Paper uni  | Paper fused |
| --------- | ----------- | ---------- | --------- | ------------- | ---------- | ----------- |
| TCGA-CESC | 49.6 (12.0) | hist 0.641 | 0.662     | 0.684         | expr 0.670 | 0.704       |
| TCGA-CHOL | 6.6 (3.4)   | demo 0.456 | 0.398     | 0.422         | text 0.557 | 0.493       |
| TCGA-DLBC | 8.6 (1.4)   | demo 0.623 | 0.294     | 0.563         | -          | -           |
| TCGA-ESCA | 23.2 (12.2) | expr 0.616 | 0.647     | 0.636         | hist 0.596 | 0.576       |
| TCGA-KICH | 13.0 (1.8)  | hist 0.681 | 0.654     | 0.663         | expr 0.937 | 0.937       |
| TCGA-MESO | 13.2 (11.2) | expr 0.688 | 0.635     | 0.630         | hist 0.643 | 0.665       |
| TCGA-OV   | 8.4 (4.2)   | expr 0.648 | 0.509     | 0.592         | expr 0.653 | 0.588       |
| TCGA-READ | 28.8 (4.0)  | demo 0.663 | 0.625     | 0.650         | demo 0.706 | 0.664       |
| TCGA-UCS  | 10.4 (6.8)  | demo 0.601 | 0.472     | 0.548         | text 0.654 | 0.628       |
| TCGA-UVM  | 12.8 (4.2)  | expr 0.697 | 0.720     | 0.680         | demo 0.684 | 0.739       |

Stage-2 hazard ratios exp(coef) per one std of each z-scored stage-1 risk,
mean over 5 folds. The paper's column is its Table 8 (mean over its 5 splits,
all 32 cohorts, in-sample stage 1).

| Modality | Two-stage | Two-stage oof | Paper |
| -------- | --------- | ------------- | ----- |
| demo     | 1.115     | 1.284         | 1.323 |
| canc     | 0.694     | 1.646         | 0.642 |
| expr     | 3.455     | 1.195         | 1.961 |
| hist     | 7.552     | 1.132         | 2.054 |
| text     | 15.522    | 1.122         | 2.193 |

Findings:

- Two-stage fusion (0.711 ± 0.030) is about 0.05 above each embedding modality
  (0.660-0.664; within one std for text, 0.060) and within the noise of cancer
  type alone (0.709 ± 0.016). The pooled C-index mostly ranks the 10 cohorts;
  the paper's 0.793 on 32 cohorts has much more data per model.
- Joint fusion is clearly worse (0.651, 0.06 below two-stage): one linear
  model over about 800 inputs fits the roughly 700 training patients perfectly
  (training C-index 1.0) and does not generalize.
- MLP heads (300 steps) beat the converged linear joint model (0.689-0.705 vs.
  0.651), likely by early stopping: the linear joint model at lr 1e-3 and
  300 steps scores 0.690 and 0.664 on folds 0 and 1, vs. 0.656 and 0.630
  converged. The sweep rows, and "two-stage mlp" (0.726) vs. "two-stage", are
  within about one std.
- Out-of-fold stage-1 risks lift fusion from 0.711 to 0.739 (about one std;
  above cancer type alone, 0.709) and OV from 0.509 to 0.592 (noise).
  In-sample, stage 2 trusts the overfit embedding models too much: hazard
  ratios 3.5 to 15.5 for expr/hist/text and 0.69 for canc; out-of-fold, 1.12
  to 1.20 and canc 1.65, so cancer type leads. The paper's Table 8 (in-sample,
  32 cohorts, about 9 times more patients) has 2.0 to 2.2 and canc 0.64.
- Within cohorts, two-stage fusion beats the best unimodal model in 3 of 10
  cohorts (CESC, ESCA, UVM). Test folds have 7 to 50 patients and 1.4 to 12.2
  deaths, so the per-cohort std over folds is 0.05 to 0.44. CHOL, DLBC and OV
  (under 10 test patients; OV about 8 with about 4 deaths per fold, std 0.103)
  are noise; the paper also reports CHOL below 0.5 and leaves out DLBC.

### Embedding reducer

`--reduction-ablation` on the same 873 patients: 256 components per modality,
5 folds, cold caches; the `pca` row equals the main table. Every reducer
completed on every fold, with no `n_components` bound hit. expr has 256 dims,
so PCA, FastICA and the random projection map it by an invertible linear map
and do not reduce it. Kernel PCA takes its 256 components from the kernel
matrix of about 700 training patients.

| Reducer           | expr          | hist          | text          | two-stage     |
| ----------------- | ------------- | ------------- | ------------- | ------------- |
| pca (paper)       | 0.660 ± 0.021 | 0.664 ± 0.037 | 0.661 ± 0.060 | 0.711 ± 0.030 |
| kernel_pca        | 0.709 ± 0.027 | 0.709 ± 0.032 | 0.706 ± 0.032 | 0.733 ± 0.023 |
| fast_ica          | 0.635 ± 0.029 | 0.663 ± 0.040 | 0.651 ± 0.049 | 0.700 ± 0.023 |
| random_projection | 0.674 ± 0.018 | 0.593 ± 0.039 | 0.620 ± 0.016 | 0.661 ± 0.035 |

Findings:

- Kernel PCA (RBF) is the best reducer here, the reverse of the synthetic
  data: +0.045 to +0.049 per embedding modality (more than one std on expr and
  hist, within one std on text) and +0.022 on fusion (within one std). Each
  kernel-PCA modality scores about as cancer type alone (0.709), likely
  because the leading RBF components encode the cohort (not checked).
- FastICA is no longer equal to PCA: expr drops by 0.025, text by 0.010 and
  fusion by 0.011, all within about one std. A likely cause (not verified):
  the ridge and the 3,000 Adam steps are not invariant to the unit-variance
  rescaling of the sources. FastICA converged in all 15 fits.
- The random projection loses on hist (-0.071) and text (-0.041), where it
  reduces 1536 and 4096 dims to 256, and on fusion (-0.050). On expr, where it
  does not reduce, it is within one std of PCA (0.674 vs. 0.660).
- Two-stage fusion is above every embedding modality for PCA, kernel PCA and
  FastICA, by about one std or less with PCA (text), kernel PCA and FastICA
  (hist); with the random projection it is below expr (0.661 vs. 0.674). With
  kernel PCA, fusion is above cancer type alone by about one std (0.733 vs.
  0.709). PCA, the paper's choice, is not the best reducer on these 10
  cohorts.
