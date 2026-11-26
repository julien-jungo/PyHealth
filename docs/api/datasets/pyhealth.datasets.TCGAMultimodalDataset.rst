pyhealth.datasets.TCGAMultimodalDataset
=======================================

Overview
--------

Pan-cancer cohort from The Cancer Genome Atlas (TCGA) used by Song et al.,
`Multimodal Cancer Modeling in the Age of Foundation Model Embeddings
<https://proceedings.mlr.press/v297/song26a.html>`_ (ML4H 2025): 7,982 patients
across 32 cancer types. Each patient has a clinical record and precomputed
foundation-model embeddings of gene expression (BulkRNABert), whole-slide
histology (UNI2-h) and pathology reports (BioMistral). The clinical data and
folds come from the `authors' repository
<https://github.com/StevenSong/multimodal-cancer-modeling>`_; the
`tcga-multimodal-embeddings <https://github.com/YaseenM2/tcga-multimodal-embeddings>`_
repository generates the embeddings and writes the folder this loader reads.
The section "Run on real data" of ``examples/tcga_survival_cox_survival.md``
is the step-by-step guide.

API Reference
-------------

.. autoclass:: pyhealth.datasets.TCGAMultimodalDataset
    :members:
    :undoc-members:
    :show-inheritance:
