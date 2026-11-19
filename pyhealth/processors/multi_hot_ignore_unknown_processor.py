"""Multi-hot processor that ignores tokens not seen during fitting.

Contributors: Julien David Jungo (jjungo2), Donovan Scott Below (dbelow2),
    Yaseen Mohamed (yaseenm2)
Paper: Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model
    Embeddings", ML4H 2025
Paper link: https://proceedings.mlr.press/v297/song26a.html
Description: One-hot encodes clinical tokens with a vocabulary fitted on the
    training split and sets no bit for unseen tokens, like scikit-learn's
    ``OneHotEncoder(handle_unknown="ignore")``.
"""

import logging
from typing import Any

import torch

from . import register_processor
from .multi_hot_processor import MultiHotProcessor

logger = logging.getLogger(__name__)


@register_processor("multi_hot_ignore_unknown")
class MultiHotIgnoreUnknownProcessor(MultiHotProcessor):
    """Multi-hot encoder that maps unseen tokens to no bit.

    :class:`MultiHotProcessor` raises ``KeyError`` on a token that ``fit()``
    did not see. With ``set_task(task, split=...)`` the vocabulary comes from
    the training part only, so a validation or test sample with a new value
    (e.g. a rare race) would fail. This subclass drops such tokens instead:
    their sample gets zeros for them, and known tokens are encoded as in the
    parent.

    Each unseen token is logged once at INFO level per processor instance,
    not once per sample.

    Input:
        - List of categorical tokens (e.g., ``["sex:female", "age:(60, 80]"]``).

    Output:
        - ``torch.float32`` tensor of shape ``(num_categories,)``.

    Examples:
        >>> processor = MultiHotIgnoreUnknownProcessor()
        >>> processor.fit([{"x": ["a", "b"]}, {"x": ["c"]}], "x")
        >>> processor.process(["a", "unseen"])
        tensor([1., 0., 0.])
    """

    def __init__(self) -> None:
        super().__init__()
        self._logged_unknown: set[Any] = set()

    def process(self, value: Any) -> torch.Tensor:
        """Encodes the known tokens of ``value`` and skips the unseen ones.

        Args:
            value: List of categorical tokens.

        Returns:
            Multi-hot ``float32`` tensor over the fitted vocabulary.

        Raises:
            ValueError: If ``value`` is not a list.
        """
        if not isinstance(value, list):
            return super().process(value)

        known = [token for token in value if token in self.label_vocab]
        for token in value:
            self._log_unknown(token)

        return super().process(known)

    def _log_unknown(self, token: Any) -> None:
        if token in self.label_vocab or token in self._logged_unknown:
            return

        self._logged_unknown.add(token)
        logger.info(
            "Ignoring token %r not seen in fit (vocab size %d)",
            token,
            len(self.label_vocab),
        )

    def __repr__(self) -> str:
        return (
            f"MultiHotIgnoreUnknownProcessor(label_vocab_size={len(self.label_vocab)})"
        )
