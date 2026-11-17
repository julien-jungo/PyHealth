pyhealth.models.CoxSurvivalModel
================================

Cox proportional hazards model over dense per-patient features, from Song et
al., `Multimodal Cancer Modeling in the Age of Foundation Model Embeddings
<https://proceedings.mlr.press/v297/song26a.html>`_, ML4H 2025.
A linear head reproduces the paper's ridge CoxPH on standardized, PCA-reduced
embeddings; an MLP head is offered as an extension.

Inputs and outputs
------------------

The inputs are dense feature fields: each must give one vector of fixed width
per sample, such as a ``tensor`` field. ``feature_keys`` selects the fields;
it defaults to all input fields, and one field gives a unimodal model. The
model concatenates the selected fields. Labels follow
:class:`~pyhealth.models.TrajSurv`: ``duration`` and ``event`` tensor
outputs, or point ``duration_key`` and ``event_key`` at other output fields.

``forward`` returns:

- ``risk``: log hazard ratio, shape ``[B]``. Higher means earlier events.
- ``logit``: the same values, shape ``[B, 1]``.
- ``y_prob``: also the risk, shape ``[B, 1]``. It is not a probability.
- ``loss``: negative Cox partial likelihood (Breslow ties). Only with labels.
- ``y_true``: duration and event, shape ``[B, 2]``. Only with labels.

``mode`` is ``None``, so the Trainer reports only the loss. Evaluate with
:func:`~pyhealth.metrics.survival.concordance_index` on ``Trainer.inference``
output: ``y_true[:, 0]`` is the duration, ``y_true[:, 1]`` the event, and
``y_prob`` the risk.

Training
--------

The Cox risk sets are those of each batch. To get the paper's ridge CoxPH,
use the linear head, train full-batch (``batch_size`` = training set size),
and pass ``weight_decay = ridge_weight_decay(0.1, n_events)`` to
``Trainer.train``, with ``n_events`` counted on the training set.
``ridge_weight_decay`` is defined in the example below.

Example
-------

``examples/tcga_survival_cox_survival.py`` runs the TCGA pipeline with
cross-validation and ablations. It is work in progress.

TODO: two-stage fusion usage.

API
---

.. autoclass:: pyhealth.models.CoxSurvivalModel
    :members:
    :undoc-members:
    :show-inheritance:
