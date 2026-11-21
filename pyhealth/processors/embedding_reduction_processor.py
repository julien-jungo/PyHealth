"""Embedding standardization and dimensionality reduction processor.

Contributors: Julien David Jungo (jjungo2), Donovan Scott Below (dbelow2),
    Yaseen Mohamed (yaseenm2)
Paper: Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
    Embeddings", ML4H 2025
Paper link: https://proceedings.mlr.press/v297/song26a.html
Description: Z-scores fixed-length foundation-model embeddings and reduces
    them with PCA (as in the paper), kernel PCA, FastICA or a Gaussian random
    projection, all fitted on the training split.
"""

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from . import register_processor
from .base_processor import LegacyFeatureProcessor

RANDOM_STATE = 42
"""Seed of every stochastic reducer, as the paper's code seeds its PCA."""

METHOD_PCA = "pca"
METHOD_KERNEL_PCA = "kernel_pca"
METHOD_FAST_ICA = "fast_ica"
METHOD_RANDOM_PROJECTION = "random_projection"
REDUCTION_METHODS = (
    METHOD_PCA,
    METHOD_KERNEL_PCA,
    METHOD_FAST_ICA,
    METHOD_RANDOM_PROJECTION,
)
"""Accepted values of ``method``; ``"pca"`` is the paper's."""

KERNEL_PCA_KERNEL = "rbf"
FAST_ICA_WHITEN = "unit-variance"


class _Reducer(Protocol):
    """The part of the sklearn transformer API the processor uses."""

    def fit(self, X: np.ndarray) -> Any: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...


def _make_reducer(method: str, n_components: int) -> _Reducer:
    """Builds the unfitted sklearn reducer of ``method``.

    Args:
        method: One of ``REDUCTION_METHODS``.
        n_components: Output width.

    Returns:
        ``PCA``, ``KernelPCA`` (RBF kernel), ``FastICA`` (unit-variance
        whitening) or ``GaussianRandomProjection``, seeded with
        ``RANDOM_STATE``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == METHOD_PCA:
        return PCA(n_components=n_components, random_state=RANDOM_STATE)
    if method == METHOD_KERNEL_PCA:
        return KernelPCA(
            n_components=n_components,
            kernel=KERNEL_PCA_KERNEL,
            random_state=RANDOM_STATE,
        )
    if method == METHOD_FAST_ICA:
        return FastICA(
            n_components=n_components,
            whiten=FAST_ICA_WHITEN,
            random_state=RANDOM_STATE,
        )
    if method == METHOD_RANDOM_PROJECTION:
        return GaussianRandomProjection(
            n_components=n_components, random_state=RANDOM_STATE
        )
    raise ValueError(f"method must be one of {list(REDUCTION_METHODS)}, got {method!r}")


def validate_reduction(n_components: int | None, method: str) -> None:
    """Checks a ``(n_components, method)`` pair before any data is seen.

    Args:
        n_components: Output width, or ``None`` to only standardize.
        method: Reduction method.

    Raises:
        ValueError: If ``method`` is not in ``REDUCTION_METHODS``,
            ``n_components`` is not a positive int or ``None``, or
            ``n_components`` is ``None`` with a method other than ``"pca"``.

    Examples:
        >>> validate_reduction(8, "kernel_pca")
        >>> validate_reduction(0, "pca")
        Traceback (most recent call last):
            ...
        ValueError: n_components must be a positive int or None, got 0
    """
    if method not in REDUCTION_METHODS:
        raise ValueError(
            f"method must be one of {list(REDUCTION_METHODS)}, got {method!r}"
        )
    if n_components is not None and (
        isinstance(n_components, bool)
        or not isinstance(n_components, int)
        or n_components < 1
    ):
        raise ValueError(
            f"n_components must be a positive int or None, got {n_components!r}"
        )
    if n_components is None and method != METHOD_PCA:
        raise ValueError(
            f"n_components=None only standardizes; method {method!r} needs "
            "a positive int"
        )


def _max_components(method: str, n_samples: int, dim: int) -> tuple[int, str]:
    """Returns the largest valid ``n_components`` of ``method`` and its bound.

    sklearn clamps or only warns past these bounds, so the processor checks
    them itself. Kernel PCA works on the ``n_samples x n_samples`` kernel
    matrix and is not bounded by ``dim``; centering gives that matrix rank at
    most ``n_samples - 1``, so a further component would always be zero. A
    random projection is not bounded by ``n_samples`` but would not reduce
    past ``dim``.
    """
    if method == METHOD_KERNEL_PCA:
        limit = n_samples - 1
        return limit, f"n_samples - 1={limit}"
    if method == METHOD_RANDOM_PROJECTION:
        return dim, f"dim={dim}"
    bound = min(n_samples, dim)
    return bound, f"min(n_samples={n_samples}, dim={dim})={bound}"


@register_processor("embedding_reduction")
class EmbeddingReductionProcessor(LegacyFeatureProcessor):
    """Standardizes a fixed-length embedding and reduces its dimension.

    ``fit()`` learns a z-score (``StandardScaler``) and then, unless
    ``n_components`` is ``None``, a reducer on the z-scored embeddings:

    - ``"pca"``: ``PCA``, the paper's choice (default).
    - ``"kernel_pca"``: ``KernelPCA`` with an RBF kernel.
    - ``"fast_ica"``: ``FastICA`` with unit-variance whitening.
    - ``"random_projection"``: ``GaussianRandomProjection``.

    All reducers are seeded with ``RANDOM_STATE``. Fit the processor on
    training patients only via ``set_task(task, split=...)``; PyHealth warns
    otherwise because ``learns_statistics`` is set.

    Input:
        - One embedding per sample: a list of floats of a fixed length.

    Output:
        - ``torch.float32`` tensor of shape ``(n_components,)``, or of the
          embedding length when ``n_components`` is ``None``.

    Args:
        n_components: Output width, or ``None`` to only standardize (only
            with ``method="pca"``).
        method: Reduction method, one of ``REDUCTION_METHODS``.

    Raises:
        ValueError: If ``method`` is unknown, ``n_components`` is not a
            positive int or ``None``, or ``n_components`` is ``None`` with a
            method other than ``"pca"``.

    Examples:
        >>> processor = EmbeddingReductionProcessor(
        ...     n_components=2, method="random_projection"
        ... )
        >>> samples = [{"x": [1.0, 2.0, 3.0]}, {"x": [2.0, 0.0, 1.0]},
        ...            {"x": [0.0, 1.0, 5.0]}]
        >>> processor.fit(samples, "x")
        >>> processor.process([1.0, 1.0, 1.0]).shape
        torch.Size([2])
    """

    learns_statistics = True
    stores_tensor = True

    def __init__(
        self, n_components: int | None = None, method: str = METHOD_PCA
    ) -> None:
        validate_reduction(n_components, method)
        self.method = method
        self.n_components = n_components
        self._input_dim: int | None = None
        self._scaler: StandardScaler | None = None
        self._reducer: _Reducer | None = None

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Fits the scaler and the reducer on the samples' embeddings.

        Args:
            samples: Sample dictionaries, usually the training split.
            field: The embedding field.

        Raises:
            ValueError: If there are no embeddings, an embedding is missing,
                not 1-D, not finite, or differs in length from the first, or
                ``n_components`` exceeds the method's bound:
                ``min(n_samples, embedding length)`` for ``"pca"`` and
                ``"fast_ica"``, ``n_samples - 1`` for ``"kernel_pca"``, and the
                embedding length for ``"random_projection"``.
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
        if self.n_components is not None:
            limit, bound = _max_components(self.method, len(matrix), dim)
            if self.n_components > limit:
                raise ValueError(
                    f"Field '{field}': n_components={self.n_components} exceeds "
                    f"{bound} for method '{self.method}'"
                )

        scaler = StandardScaler().fit(matrix)
        reducer = None
        if self.n_components is not None:
            reducer = _make_reducer(self.method, self.n_components)
            reducer.fit(scaler.transform(matrix))

        self._input_dim, self._scaler, self._reducer = dim, scaler, reducer

    def process(self, value: Any) -> torch.Tensor:
        """Transforms one embedding with the fitted scaler and reducer.

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
            raise RuntimeError(
                "EmbeddingReductionProcessor.process called before fit()"
            )

        vector = self._to_vector(value, "value", 0)
        if len(vector) != self._input_dim:
            raise ValueError(
                f"Embedding has length {len(vector)}, fitted on {self._input_dim}"
            )

        row = self._scaler.transform(vector.reshape(1, -1))
        if self._reducer is not None:
            row = self._reducer.transform(row)

        return torch.from_numpy(row[0].astype(np.float32))

    def size(self) -> int | None:
        """Returns the output width, or ``None`` before fitting."""
        if self._input_dim is None:
            return None

        return self.n_components if self._reducer is not None else self._input_dim

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
        return (
            f"EmbeddingReductionProcessor(method={self.method!r}, "
            f"n_components={self.n_components})"
        )
