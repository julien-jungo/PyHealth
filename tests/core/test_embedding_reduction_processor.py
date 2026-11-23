"""Tests for EmbeddingReductionProcessor. Synthetic vectors only; each runs in ms."""

import os
import tempfile
import unittest
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyhealth.datasets import Split
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.processors import PROCESSOR_REGISTRY, EmbeddingReductionProcessor
from pyhealth.processors.embedding_reduction_processor import (
    METHOD_FAST_ICA,
    METHOD_KERNEL_PCA,
    METHOD_PCA,
    METHOD_RANDOM_PROJECTION,
    RANDOM_STATE,
    REDUCTION_METHODS,
    validate_reduction,
)

FIELD = "x"
N_SAMPLES = 12
DIM = 6
N_COMPONENTS = 3
OUTLIER = 1e6


def _samples(n: int = N_SAMPLES, dim: int = DIM, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=5.0, scale=3.0, size=(n, dim))
    return [
        {"patient_id": f"p{i}", FIELD: row.tolist()} for i, row in enumerate(matrix)
    ]


class _HoldoutSplit(Split):
    def __init__(self, test_patients: Sequence[str]) -> None:
        self.test_patients = sorted(test_patients)

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        test = set(self.test_patients)
        parts: tuple[list[int], list[int]] = ([], [])

        for pid, indices in patient_to_index.items():
            parts[pid in test].extend(indices)

        return [np.array(sorted(p), dtype=np.int64) for p in parts]

    def to_dict(self) -> dict:
        return {"kind": "holdout", "test_patients": self.test_patients}


def _fitted(
    n_components: int | None = N_COMPONENTS, method: str = METHOD_PCA
) -> EmbeddingReductionProcessor:
    processor = EmbeddingReductionProcessor(n_components=n_components, method=method)
    processor.fit(_samples(), FIELD)
    return processor


def _transform_all(processor: EmbeddingReductionProcessor, samples: list[dict]):
    return torch.stack([processor.process(s[FIELD]) for s in samples])


class TestEmbeddingReductionProcessor(unittest.TestCase):
    def test_registered_as_embedding_reduction(self) -> None:
        self.assertIs(
            PROCESSOR_REGISTRY["embedding_reduction"], EmbeddingReductionProcessor
        )
        self.assertTrue(EmbeddingReductionProcessor.learns_statistics)
        self.assertTrue(EmbeddingReductionProcessor.stores_tensor)

    def test_invalid_n_components_raise(self) -> None:
        for n in (0, -1, 2.5, True, "3"):
            with self.subTest(n=n), self.assertRaises(ValueError):
                EmbeddingReductionProcessor(n_components=n)

    def test_invalid_method_raises(self) -> None:
        for method in ("ica", "PCA", None, 1):
            with self.subTest(method=method), self.assertRaises(ValueError):
                EmbeddingReductionProcessor(n_components=2, method=method)

    def test_n_components_none_needs_pca(self) -> None:
        for method in REDUCTION_METHODS[1:]:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "only standardizes"):
                    EmbeddingReductionProcessor(method=method)

    def test_default_method_is_pca(self) -> None:
        processor = EmbeddingReductionProcessor(N_COMPONENTS)
        self.assertEqual(processor.method, METHOD_PCA)
        self.assertEqual(
            repr(processor),
            f"EmbeddingReductionProcessor(method='pca', n_components={N_COMPONENTS})",
        )

    def test_pca_matches_seeded_sklearn_pca(self) -> None:
        matrix = np.array([s[FIELD] for s in _samples()])
        scaled = StandardScaler().fit_transform(matrix)
        expected = PCA(N_COMPONENTS, random_state=RANDOM_STATE).fit_transform(scaled)

        torch.testing.assert_close(
            _transform_all(_fitted(), _samples()),
            torch.tensor(expected, dtype=torch.float32),
        )

    def test_every_method_shape_center_and_determinism(self) -> None:
        for method in REDUCTION_METHODS:
            with self.subTest(method=method):
                processor = _fitted(method=method)
                out = _transform_all(processor, _samples()).double()

                self.assertEqual(out.shape, (N_SAMPLES, N_COMPONENTS))
                self.assertEqual(processor.size(), N_COMPONENTS)
                self.assertTrue(torch.isfinite(out).all())
                torch.testing.assert_close(
                    out.mean(0),
                    torch.zeros(N_COMPONENTS, dtype=torch.float64),
                    atol=1e-5,
                    rtol=0,
                )
                torch.testing.assert_close(
                    _transform_all(_fitted(method=method), _samples()).double(), out
                )

    def test_fast_ica_sources_have_unit_variance(self) -> None:
        out = _transform_all(_fitted(method=METHOD_FAST_ICA), _samples()).double()

        torch.testing.assert_close(
            out.std(0, unbiased=False),
            torch.ones(N_COMPONENTS, dtype=torch.float64),
            atol=1e-4,
            rtol=0,
        )

    def test_per_method_component_bounds(self) -> None:
        n_small = 4
        cases = {
            METHOD_PCA: (DIM + 1, "exceeds min"),
            METHOD_FAST_ICA: (DIM + 1, "exceeds min"),
            METHOD_RANDOM_PROJECTION: (DIM + 1, f"exceeds dim={DIM}"),
            METHOD_KERNEL_PCA: (n_small, f"exceeds n_samples - 1={n_small - 1}"),
        }
        for method, (n, message) in cases.items():
            with self.subTest(method=method):
                processor = EmbeddingReductionProcessor(n, method=method)
                with self.assertRaisesRegex(ValueError, message):
                    processor.fit(_samples(n=n_small), FIELD)

    def test_kernel_pca_bound_is_centered_kernel_rank(self) -> None:
        n_small = 4
        processor = EmbeddingReductionProcessor(n_small - 1, method=METHOD_KERNEL_PCA)
        processor.fit(_samples(n=n_small), FIELD)
        self.assertEqual(processor.size(), n_small - 1)

    def test_validate_reduction(self) -> None:
        validate_reduction(None, METHOD_PCA)
        validate_reduction(N_COMPONENTS, METHOD_FAST_ICA)
        for n, method in ((0, METHOD_PCA), (2, "umap"), (None, METHOD_KERNEL_PCA)):
            with self.subTest(n=n, method=method), self.assertRaises(ValueError):
                validate_reduction(n, method)

    def test_kernel_pca_may_exceed_dim(self) -> None:
        processor = _fitted(n_components=DIM + 2, method=METHOD_KERNEL_PCA)
        self.assertEqual(processor.process(_samples()[0][FIELD]).shape, (DIM + 2,))

    def test_random_projection_ignores_sample_count(self) -> None:
        processor = EmbeddingReductionProcessor(DIM, method=METHOD_RANDOM_PROJECTION)
        processor.fit(_samples(n=2), FIELD)
        self.assertEqual(processor.size(), DIM)

    def test_output_shape_and_dtype(self) -> None:
        processor = _fitted()
        out = processor.process(_samples()[0][FIELD])

        self.assertEqual(out.shape, (N_COMPONENTS,))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(processor.size(), N_COMPONENTS)
        self.assertEqual(processor.schema(), ("value",))
        self.assertFalse(processor.is_token())

        tensor_out = processor.process(torch.tensor(_samples()[0][FIELD]))
        torch.testing.assert_close(tensor_out, out)

    def test_train_set_standardized(self) -> None:
        processor = _fitted(n_components=None)
        out = _transform_all(processor, _samples()).double()

        torch.testing.assert_close(
            out.mean(0), torch.zeros(DIM, dtype=torch.float64), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            out.std(0, unbiased=False),
            torch.ones(DIM, dtype=torch.float64),
            atol=1e-5,
            rtol=0,
        )

    def test_pca_components_are_centered_and_ordered(self) -> None:
        out = _transform_all(_fitted(), _samples()).double()

        torch.testing.assert_close(
            out.mean(0),
            torch.zeros(N_COMPONENTS, dtype=torch.float64),
            atol=1e-5,
            rtol=0,
        )

        variances = out.var(0)
        self.assertTrue(torch.all(variances[:-1] >= variances[1:]))

    def test_n_components_none_only_standardizes(self) -> None:
        processor = _fitted(n_components=None)
        self.assertEqual(processor.size(), DIM)

        matrix = np.array([s[FIELD] for s in _samples()])
        value = _samples(seed=1)[0][FIELD]
        expected = (np.array(value) - matrix.mean(0)) / matrix.std(0)

        torch.testing.assert_close(
            processor.process(value), torch.tensor(expected, dtype=torch.float32)
        )

    def test_too_many_components_raises(self) -> None:
        processor = EmbeddingReductionProcessor(n_components=DIM + 1)
        with self.assertRaisesRegex(ValueError, "exceeds min"):
            processor.fit(_samples(), FIELD)

        few = EmbeddingReductionProcessor(n_components=3)
        with self.assertRaisesRegex(ValueError, "n_samples=2"):
            few.fit(_samples(n=2), FIELD)

    def test_inconsistent_lengths_raise(self) -> None:
        samples = _samples()
        samples[3][FIELD] = samples[3][FIELD][:-1]

        with self.assertRaisesRegex(ValueError, "embedding 3 has length"):
            EmbeddingReductionProcessor(n_components=2).fit(samples, FIELD)

        processor = _fitted()
        with self.assertRaisesRegex(ValueError, "length"):
            processor.process([1.0] * (DIM + 1))

    def test_invalid_embeddings_raise(self) -> None:
        for bad in (None, [], [[1.0, 2.0]], [1.0, float("nan")]):
            samples = _samples(n=3, dim=2)
            samples[1][FIELD] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EmbeddingReductionProcessor().fit(samples, FIELD)

        with self.assertRaisesRegex(ValueError, "No samples"):
            EmbeddingReductionProcessor().fit([], FIELD)

    def test_process_before_fit_raises(self) -> None:
        processor = EmbeddingReductionProcessor(n_components=2)
        self.assertIsNone(processor.size())

        with self.assertRaisesRegex(RuntimeError, "before fit"):
            processor.process([1.0, 2.0])

    def test_fitted_on_train_split_only(self) -> None:
        samples = _samples()
        test_ids = ["p0", "p1"]
        for sample in samples[:2]:
            sample[FIELD] = [OUTLIER] * DIM

        builder = SampleBuilder(
            input_schema={FIELD: ("embedding_reduction", {"n_components": None})},
            output_schema={},
        )
        builder.fit(samples, split=_HoldoutSplit(test_ids))

        processor = builder.input_processors[FIELD]
        train = _transform_all(processor, samples[2:]).double()

        torch.testing.assert_close(
            train.mean(0), torch.zeros(DIM, dtype=torch.float64), atol=1e-5, rtol=0
        )
        self.assertTrue(torch.all(processor.process(samples[0][FIELD]) > 1e3))

    def test_save_load_round_trip(self) -> None:
        samples = _samples()
        for method in REDUCTION_METHODS:
            with self.subTest(method=method):
                kwargs = {"n_components": N_COMPONENTS, "method": method}
                builder = SampleBuilder(
                    input_schema={FIELD: ("embedding_reduction", kwargs)},
                    output_schema={},
                )
                builder.fit(samples, split=_HoldoutSplit(["p0"]))

                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "schema.pkl")
                    builder.save(path)
                    loaded = SampleBuilder.load(path).input_processors[FIELD]
                original = builder.input_processors[FIELD]

                self.assertEqual(loaded.method, method)
                self.assertEqual(loaded.size(), N_COMPONENTS)
                torch.testing.assert_close(
                    _transform_all(loaded, samples), _transform_all(original, samples)
                )


if __name__ == "__main__":
    unittest.main()
