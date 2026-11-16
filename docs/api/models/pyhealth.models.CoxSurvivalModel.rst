pyhealth.models.CoxSurvivalModel
================================

Cox proportional hazards model over dense per-patient features, from Song et
al., `Multimodal Cancer Modeling in the Age of Foundation Model Embeddings
<https://arxiv.org/abs/2505.07683>`_, ML4H 2025. A linear head reproduces the
paper's ridge CoxPH on standardized, PCA-reduced embeddings; an MLP head is
offered as an extension.

TODO: inputs and outputs, two-stage fusion usage, and a pointer to
``examples/tcga_survival_cox.py``.

.. autoclass:: pyhealth.models.CoxSurvivalModel
    :members:
    :undoc-members:
    :show-inheritance:
