pyhealth.datasets.TCGAMultimodalDataset
=======================================

TCGA cases with clinical data and precomputed foundation-model embeddings of
RNA-seq (BulkRNABert), histology (UNI2-h) and pathology reports (BioMistral),
as used by Song et al., `Multimodal Cancer Modeling in the Age of Foundation
Model Embeddings <https://proceedings.mlr.press/v297/song26a.html>`_, ML4H 2025.
Embeddings are generated with the authors' code
(https://github.com/StevenSong/multimodal-cancer-modeling).

TODO: data access (GDC, gated UNI2-h features and their license) and the
expected file layout.

.. autoclass:: pyhealth.datasets.TCGAMultimodalDataset
    :members:
    :undoc-members:
    :show-inheritance:
