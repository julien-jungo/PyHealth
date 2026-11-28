pyhealth.models.CoxSurvivalModel
================================

Overview
--------

Cox proportional hazards model over dense per-patient features, from Song et
al., `Multimodal Cancer Modeling in the Age of Foundation Model Embeddings
<https://proceedings.mlr.press/v297/song26a.html>`_, ML4H 2025.
A linear head reproduces the paper's ridge CoxPH on standardized, PCA-reduced
embeddings; an MLP head is offered as an extension.

Inputs and outputs
------------------

The inputs are dense feature fields: each must give one vector of fixed width
per sample, such as a ``tensor`` or ``embedding_reduction`` field.
``feature_keys`` selects the fields; it defaults to all input fields, and one
field gives a unimodal model.

``forward`` returns:

- ``risk``: log hazard ratio, shape ``[B]``. Higher means earlier events.
- ``logit``: the same values, shape ``[B, 1]``.
- ``y_prob``: also the risk, shape ``[B, 1]``. It is not a probability.
- ``loss``: negative Cox partial likelihood (Breslow ties). Only with labels.
- ``y_true``: duration and event, shape ``[B, 2]``. Only with labels.

Evaluate with :func:`~pyhealth.metrics.survival.concordance_index` on
``Trainer.inference`` output: ``y_true[:, 0]`` is the duration,
``y_true[:, 1]`` the event, and ``y_prob`` the risk.

Example and ablation
--------------------

``examples/tcga_survival_cox_survival.py`` runs the paper's two-stage late
fusion on :class:`~pyhealth.datasets.TCGAMultimodalDataset` per CV fold:
``stage1`` fits one unimodal model per modality (``feature_keys=[modality]``),
and ``stage2`` fits a linear model on the stage-1 risks, z-scored with a
``"zscore"`` :class:`~pyhealth.processors.TensorProcessor` fitted on the
training risks. All fits are full-batch with
``weight_decay = ridge_weight_decay(0.1, n_events)`` (defined in the example).
The example also runs the joint-fusion, MLP-head and reducer ablations. Setup,
results and the guide to get the real data are in
``examples/tcga_survival_cox_survival.md``.

API
---

.. autoclass:: pyhealth.models.CoxSurvivalModel
    :members:
    :undoc-members:
    :show-inheritance:
