pyhealth.processors.EmbeddingReductionProcessor
===============================================

Standardizes fixed-length embeddings and reduces them with PCA (the default),
kernel PCA (RBF kernel), FastICA (unit-variance whitening) or a Gaussian
random projection. The scaler and the reducer are fitted on the training
split. Registered as ``"embedding_reduction"``; choose the reducer with the
``method`` argument (``"pca"``, ``"kernel_pca"``, ``"fast_ica"`` or
``"random_projection"``).

.. autoclass:: pyhealth.processors.EmbeddingReductionProcessor
    :members:
    :undoc-members:
    :show-inheritance:
