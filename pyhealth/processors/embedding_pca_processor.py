from collections.abc import Iterable
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("embedding_pca")
class EmbeddingPCAProcessor(FeatureProcessor):
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
        # TODO: validate n_components (positive int or None).
        self.n_components = n_components

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Fits the scaler and PCA on the samples' embeddings.

        Args:
            samples: Sample dictionaries, usually the training split.
            field: The embedding field.

        Raises:
            ValueError: If embeddings differ in length, or ``n_components``
                exceeds ``min(n_samples, embedding length)``.
        """
        # TODO: stack embeddings, fit sklearn StandardScaler then PCA
        # (random_state=42 like the paper); fail fast with a clear message.
        raise NotImplementedError

    def process(self, value: Any) -> torch.Tensor:
        """Transforms one embedding with the fitted scaler and PCA.

        Args:
            value: One embedding.

        Returns:
            The reduced embedding as a ``float32`` tensor.
        """
        # TODO: raise a clear error when called before fit().
        raise NotImplementedError

    def size(self) -> int | None:
        """Returns the output width, or ``None`` before fitting."""
        # TODO
        raise NotImplementedError

    def is_token(self) -> bool:
        """Returns ``False``: the output holds continuous values."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return f"EmbeddingPCAProcessor(n_components={self.n_components})"
