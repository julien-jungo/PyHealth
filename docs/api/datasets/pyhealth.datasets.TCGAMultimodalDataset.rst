pyhealth.datasets.TCGAMultimodalDataset
=======================================

Overview
--------

Pan-cancer cohort from The Cancer Genome Atlas (TCGA) used by Song et al.,
`Multimodal Cancer Modeling in the Age of Foundation Model Embeddings
<https://proceedings.mlr.press/v297/song26a.html>`_ (ML4H 2025): 7,982 patients
across 32 cancer types. Each patient has a clinical record (demographics, cancer
type, survival outcome, and the paper's cross-validation fold) and precomputed
foundation-model embeddings of gene expression, whole-slide histology, and
pathology reports. The clinical table and fold file are released in the
`authors' repository <https://github.com/StevenSong/multimodal-cancer-modeling>`_;
the raw data come from the `GDC portal <https://portal.gdc.cancer.gov/>`_.

API Reference
-------------

.. autoclass:: pyhealth.datasets.TCGAMultimodalDataset
    :members:
    :undoc-members:
    :show-inheritance:
