# CoxSurvivalModel on TCGA multimodal survival: results

Results of `examples/tcga_survival_cox_survival.py` for the two-stage late
fusion of Song et al. ([Multimodal Cancer Modeling in the Age of Foundation
Model Embeddings](https://proceedings.mlr.press/v297/song26a.html), ML4H 2025;
preprint: [arXiv:2505.07683](https://arxiv.org/abs/2505.07683)) and its
ablations.

## Experimental setup

- **Data:** `--root` in the TCGAMultimodalDataset layout, or by default 300
  synthetic patients in 3 cohorts (`write_synthetic_data`, temp dir).
- **Folds:** the paper's 5 published CV folds, the `split` column of the
  authors' `results/split_cases.csv`; fold `k` is the test set, the other
  folds train. There is no validation set, as in the paper. Patients without
  a fold (79 of the 8061 in the authors' full `clinical.csv`) are left out,
  with one warning that gives the count.
- **Features:** expr/hist/text z-scored and PCA-reduced (EmbeddingPCAProcessor)
  to 256 components each, the paper's reported setting; 8 on synthetic data
  (32-dim embeddings). demo and canc are one-hot (multi_hot), unscaled.
- **Optimizer:** full-batch Adam, lr 1e-2, seed 0; one epoch of
  `steps_per_epoch` steps. Full batch makes the Cox risk sets match the
  paper's full partial likelihood. Steps: 3,000 for every linear fit (stage 1
  and stage 2), 10,000 for "joint", 300 for MLP heads.
- **Ridge:** the paper's `CoxPHSurvivalAnalysis(alpha=0.1)` becomes
  `weight_decay = ridge_weight_decay(0.1, n_events)`, with `n_events` counted
  on the training fold. The same decay is used for MLP heads, where it is not
  equivalent to a ridge penalty.
- **Fusion (paper, "two-stage"):** one linear model per modality; their
  in-sample training risks are z-scored with training statistics (a
  `"zscore"` TensorProcessor) and fed to a second linear model.
- **Metric:** Harrell's C-index on the fold's test patients, pooled and per
  cancer type, then mean ± std (ddof=1) over the 5 folds.

### Convergence

Real data, folds 0 and 1, against the exact Newton optimum of the same
objective:

- At 3,000 steps the unimodal and stage-2 fits are at most 5.2e-4 above the
  optimal loss and at most 3.4e-4 off its test C-index.
- At 300 steps the expr fit is 0.03 above the optimal loss and its test
  C-index is off by up to 0.017.
- "joint" (about 800 inputs) is 0.008 above the optimal loss at 3,000 steps
  and 2e-4 at 10,000.
- MLP heads are not trained to convergence; 300 steps act as early stopping.

## Ablations

All are novel relative to the paper.

- **Two-stage vs. joint fusion:** "joint" is one linear model over all five
  modalities, trained end to end.
- **In-sample vs. out-of-fold stage-1 risks:** "two-stage oof" trains stage 2
  on cross-fitted stage-1 risks. The training patients are dealt into 5 inner
  folds, stratified by event; each inner fold is scored by stage-1 models
  trained on the other 4. Each inner fold is a `set_task(split=FoldSplit)`
  call, so its processors are fitted on its inner training patients only.
  Test risks come from the stage-1 models on the full training fold, as in
  "two-stage". The paper feeds stage 2 the in-sample risks. This ablation is
  off by default; run the example with `--oof` to add it (with `--root` for
  the real data).
- **Linear vs. MLP head:** "two-stage mlp" uses MLP heads (64 hidden units) in
  stage 1, with the linear stage 2; "joint mlp" is one MLP over all
  modalities.
- **Hyperparameter sweep** on "joint mlp", one change at a time: hidden_dim 16
  ("h=16"), dropout 0.3 ("p=0.3"), learning rate 1e-3 ("lr=1e-3").

## Synthetic data

300 patients, 5 folds; about 100 s on an 8-core CPU by default, and about
265 s with `--oof`. The extra 160 s come from "two-stage oof": 25 inner
`set_task` calls and 125 more stage-1 fits over the 5 folds.

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

Per-cohort C-index (mean over 5 folds; canc is constant within a cohort, so it
scores 0.5):

| Cohort   | demo  | canc  | expr  | hist  | text  | two-stage | two-stage oof |
| -------- | ----- | ----- | ----- | ----- | ----- | --------- | ------------- |
| TCGA-OV  | 0.517 | 0.500 | 0.650 | 0.555 | 0.697 | 0.758     | 0.745         |
| TCGA-UCS | 0.570 | 0.500 | 0.712 | 0.611 | 0.709 | 0.832     | 0.819         |
| TCGA-UVM | 0.489 | 0.500 | 0.630 | 0.646 | 0.686 | 0.785     | 0.779         |

Stage-2 hazard ratios exp(coef), mean over 5 folds:

| Modality | two-stage | two-stage oof |
| -------- | --------- | ------------- |
| demo     | 1.161     | 0.991         |
| canc     | 1.416     | 1.403         |
| expr     | 2.241     | 2.125         |
| hist     | 1.679     | 1.489         |
| text     | 2.283     | 2.120         |

Findings:

- Every modality beats 0.5, and the paper's two-stage fusion (0.793) beats the
  best single modality (text, 0.681) by 0.11.
- Two-stage vs. joint fusion (0.793 vs. 0.777) is within one std. On a second
  synthetic draw (seed 1, 200 patients, 4 PCA components) joint wins (0.797
  vs. 0.776), so this data does not separate the two.
- Out-of-fold stage-1 risks do not help here (0.782 vs. 0.793, within one
  std). With 8 components per modality the linear stage-1 models likely
  overfit little, so in-sample risks are close to out-of-fold ones; the
  hazard ratios shrink only a little.
- MLP heads in stage 1 hurt fusion (0.649). Their in-sample training risks
  overfit, so stage 2 learns weights that do not hold on the test fold.
- On joint fusion, the MLP head is no better than the linear head (0.773 vs.
  0.777): the synthetic hazard is linear. Dropout 0.3 (0.783), hidden size 16
  (0.751) and lr 1e-3 (0.780) all stay within about one std of the default
  MLP, so the sweep shows no clear effect.

## Real data

The 10 smallest TCGA cohorts, 873 patients, PCA 256, 5 folds, 8-core CPU,
4 workers: about 270 s by default, and about 1,460 s with `--oof`. The
`--oof` run adds 125 stage-1 fits and 25 inner `set_task` calls; their
caches were cold, the outer ones warm. A warm rerun was not measured; it
still has the 125 extra fits. The paper column is the paper's 5-fold C-index
on all 32 cohorts (Tables 1 and 2, PCA 256), so it is not a like-for-like
target.

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

After an interrupted run, delete the cache: PyHealth's `set_task` can leave
a partial sample cache behind. The cache is the `--cache-dir` folder, or,
without it, `<user cache dir>/pyhealth` (`~/.cache/pyhealth` on Linux,
`~/Library/Caches/pyhealth` on macOS; the run logs the path).

Per-cohort C-index, two-stage fusion vs. the best unimodal model of
demo/expr/hist/text (mean over the folds with a comparable pair; DLBC has one
in 4 folds). The paper's columns are from its Table 17: models trained on all
32 cohorts, scored on each cohort's test cases. The paper does not report
DLBC.

"Test" is the mean test-fold size, deaths in parentheses (the paper's folds,
so the same as in its Table 17).

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

- Two-stage fusion (0.711) beats each embedding modality (0.660-0.664) by
  about 0.05, but not cancer type alone (0.709). The pooled C-index over 10
  cohorts mostly ranks the cohorts; the paper's 0.793 on 32 cohorts has much
  more data per model.
- Joint fusion is clearly worse (0.651, 0.06 below two-stage): one linear
  model over about 800 inputs fits the roughly 700 training patients perfectly
  (training C-index 1.0) and does not generalize.
- MLP heads trained for 300 steps beat the converged linear joint model
  (0.689-0.705 vs. 0.651); this is likely early stopping, not a better head:
  the linear joint model at lr 1e-3 and 300 steps (not converged) scores 0.690
  and 0.664 on folds 0 and 1, vs. 0.656 and 0.630 converged. The hidden-size,
  dropout and lr rows are within about one std of each other. "two-stage mlp"
  (0.726) vs. "two-stage" (0.711) is within one std.
- Out-of-fold stage-1 risks lift fusion from 0.711 to 0.739 (above cancer
  type alone, 0.709) and OV from 0.509 to 0.592. In-sample, the embedding
  models fit the roughly 700 training patients far better than the test
  fold, so stage 2 trusts them too much: hazard ratios 3.5 to 15.5 for
  expr/hist/text and 0.69 for canc. With out-of-fold risks the embeddings
  get 1.12 to 1.20 and canc 1.65, so cancer type leads. The paper's Table 8
  (in-sample, 32 cohorts) has embedding ratios of 2.0 to 2.2 and canc 0.64:
  with about 9 times more patients its in-sample risks overfit less.
- The exact-solver analysis of the same objective gives 0.736 pooled and
  0.567 on OV for the out-of-fold variant, vs. 0.739 and 0.592 here. Likely
  causes: the inner processor refit (the analysis reused the outer-fold PCA),
  the inner fold draw (it may equal the analysis's, which used the same
  seeds), and Adam at 3,000 steps vs. the exact Newton solver on the 125
  inner fits. The OV gap (0.025) is well within noise: OV has about 8 test
  patients and about 4 deaths per fold, and its std over folds is 0.103.
- Within cohorts, two-stage fusion beats the best unimodal model in 3 of 10
  cohorts (CESC, ESCA, UVM). Test folds have 7 to 50 patients and 1.4 to 12.2
  deaths, so the per-cohort std over folds is 0.05 to 0.44. CHOL and DLBC
  (under 10 test patients, DLBC with one fold without a comparable pair) are
  noise; the paper also reports CHOL below 0.5 and leaves out DLBC.
