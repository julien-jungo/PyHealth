"""Embedding standardization and PCA processor.

Contributors: Julien David Jungo (jjungo2), Donovan Scott Below (dbelow2),
    Yaseen Mohamed (yaseenm2)
Paper: Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
    Embeddings", ML4H 2025
Paper link: https://proceedings.mlr.press/v297/song26a.html
Description: Z-scores fixed-length foundation-model embeddings and reduces
    them with PCA, both fitted on the training split, as in the paper.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import register_processor
from .base_processor import LegacyFeatureProcessor

PCA_RANDOM_STATE = 42
"""Seed of the randomized PCA solvers, as in the paper's code."""


@register_processor("embedding_pca")
class EmbeddingPCAProcessor(LegacyFeatureProcessor):
    """Standardizes a fixed-length embedding and reduces it with PCA.

    ``fit()`` learns a z-score (``StandardScaler``) and, unless
    ``n_components`` is ``None``, a ``PCA`` from the samples it is given. Fit
    it on training patients only via ``set_task(task, split=...)``; PyHealth
    warns otherwise because ``learns_statistics`` is set.

    Input:
        - One embedding per sample: a list of floats of a fixed length.

    Output:
        - ``torch.float32`` tensor of shape ``(n_components,)``, or of the
          embedding length when ``n_components`` is ``None``.

    Args:
        n_components: Number of principal components to keep, or ``None`` to
            only standardize.

    Raises:
        ValueError: If ``n_components`` is not a positive int or ``None``.

    Examples:
        >>> processor = EmbeddingPCAProcessor(n_components=2)
        >>> samples = [{"x": [1.0, 2.0, 3.0]}, {"x": [2.0, 0.0, 1.0]},
        ...            {"x": [0.0, 1.0, 5.0]}]
        >>> processor.fit(samples, "x")
        >>> processor.process([1.0, 1.0, 1.0]).shape
        torch.Size([2])
    """

    learns_statistics = True
    stores_tensor = True

    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and (
            isinstance(n_components, bool)
            or not isinstance(n_components, int)
            or n_components < 1
        ):
            raise ValueError(
                f"n_components must be a positive int or None, got {n_components!r}"
            )

        self.n_components = n_components
        self._input_dim: int | None = None
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Fits the scaler and PCA on the samples' embeddings.

        Args:
            samples: Sample dictionaries, usually the training split.
            field: The embedding field.

        Raises:
            ValueError: If there are no embeddings, an embedding is missing,
                not 1-D, not finite, or differs in length from the first, or
                ``n_components`` exceeds ``min(n_samples, embedding length)``.
        """
        vectors = [
            self._to_vector(sample.get(field), field, i)
            for i, sample in enumerate(samples)
        ]
        if not vectors:
            raise ValueError(f"No samples to fit field '{field}' on")

        dim = len(vectors[0])
        for i, vector in enumerate(vectors):
            if len(vector) != dim:
                raise ValueError(
                    f"Field '{field}': embedding {i} has length {len(vector)}, "
                    f"expected {dim} like the first embedding"
                )

        matrix = np.stack(vectors)
        max_components = min(matrix.shape)
        if self.n_components is not None and self.n_components > max_components:
            raise ValueError(
                f"Field '{field}': n_components={self.n_components} exceeds "
                f"min(n_samples={matrix.shape[0]}, dim={dim})={max_components}"
            )

        scaler = StandardScaler().fit(matrix)
        pca = None
        if self.n_components is not None:
            pca = PCA(n_components=self.n_components, random_state=PCA_RANDOM_STATE)
            pca.fit(scaler.transform(matrix))

        self._input_dim, self._scaler, self._pca = dim, scaler, pca

    def process(self, value: Any) -> torch.Tensor:
        """Transforms one embedding with the fitted scaler and PCA.

        Args:
            value: One embedding.

        Returns:
            The reduced embedding as a ``float32`` tensor.

        Raises:
            RuntimeError: If called before ``fit()``.
            ValueError: If the embedding is invalid or its length differs from
                the fitted one.
        """
        if self._scaler is None:
            raise RuntimeError("EmbeddingPCAProcessor.process called before fit()")

        vector = self._to_vector(value, "value", 0)
        if len(vector) != self._input_dim:
            raise ValueError(
                f"Embedding has length {len(vector)}, fitted on {self._input_dim}"
            )

        row = self._scaler.transform(vector.reshape(1, -1))
        if self._pca is not None:
            row = self._pca.transform(row)

        return torch.from_numpy(row[0].astype(np.float32))

    def size(self) -> int | None:
        """Returns the output width, or ``None`` before fitting."""
        if self._input_dim is None:
            return None

        return self.n_components if self._pca is not None else self._input_dim

    @staticmethod
    def _to_vector(value: Any, field: str, index: int) -> np.ndarray:
        if value is None:
            raise ValueError(f"Field '{field}': embedding {index} is missing")

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                f"Field '{field}': embedding {index} must be a non-empty 1-D "
                f"vector, got shape {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(f"Field '{field}': embedding {index} has NaN or inf")

        return vector

    def is_token(self) -> bool:
        """Returns ``False``: the output holds continuous values."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Returns ``("value",)``: one tensor per sample."""
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Returns ``(1,)``: the value is a 1-D vector."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        """Returns ``(False,)``: the vector has no spatial axis."""
        return (False,)

    def __repr__(self) -> str:
        return f"EmbeddingPCAProcessor(n_components={self.n_components})"
