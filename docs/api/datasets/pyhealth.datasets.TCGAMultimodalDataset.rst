pyhealth.datasets.TCGAMultimodalDataset
=======================================

TCGA cases with clinical data and precomputed foundation-model embeddings of
RNA-seq (BulkRNABert), histology (UNI2-h) and pathology reports (BioMistral),
as used by Song et al., `Multimodal Cancer Modeling in the Age of Foundation
Model Embeddings <https://arxiv.org/abs/2505.07683>`_, ML4H 2025. Embeddings
are generated with the authors' code
(https://github.com/StevenSong/multimodal-cancer-modeling).

TODO: data access (GDC, gated UNI2-h features and their license) and the
expected file layout.

.. autoclass:: pyhealth.datasets.TCGAMultimodalDataset
    :members:
    :undoc-members:
    :show-inheritance:
