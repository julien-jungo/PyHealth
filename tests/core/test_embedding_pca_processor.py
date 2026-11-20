"""Tests for EmbeddingPCAProcessor. Synthetic vectors only; each test runs in ms."""

import os
import tempfile
import unittest
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from pyhealth.datasets import Split
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.processors import PROCESSOR_REGISTRY, EmbeddingPCAProcessor

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


def _fitted(n_components: int | None = N_COMPONENTS) -> EmbeddingPCAProcessor:
    processor = EmbeddingPCAProcessor(n_components=n_components)
    processor.fit(_samples(), FIELD)
    return processor


def _transform_all(processor: EmbeddingPCAProcessor, samples: list[dict]):
    return torch.stack([processor.process(s[FIELD]) for s in samples])


class TestEmbeddingPCAProcessor(unittest.TestCase):
    def test_registered_as_embedding_pca(self) -> None:
        self.assertIs(PROCESSOR_REGISTRY["embedding_pca"], EmbeddingPCAProcessor)
        self.assertTrue(EmbeddingPCAProcessor.learns_statistics)
        self.assertTrue(EmbeddingPCAProcessor.stores_tensor)

    def test_invalid_n_components_raise(self) -> None:
        for n in (0, -1, 2.5, True, "3"):
            with self.subTest(n=n), self.assertRaises(ValueError):
                EmbeddingPCAProcessor(n_components=n)

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
        processor = EmbeddingPCAProcessor(n_components=DIM + 1)
        with self.assertRaisesRegex(ValueError, "exceeds min"):
            processor.fit(_samples(), FIELD)

        few = EmbeddingPCAProcessor(n_components=3)
        with self.assertRaisesRegex(ValueError, "n_samples=2"):
            few.fit(_samples(n=2), FIELD)

    def test_inconsistent_lengths_raise(self) -> None:
        samples = _samples()
        samples[3][FIELD] = samples[3][FIELD][:-1]

        with self.assertRaisesRegex(ValueError, "embedding 3 has length"):
            EmbeddingPCAProcessor(n_components=2).fit(samples, FIELD)

        processor = _fitted()
        with self.assertRaisesRegex(ValueError, "length"):
            processor.process([1.0] * (DIM + 1))

    def test_invalid_embeddings_raise(self) -> None:
        for bad in (None, [], [[1.0, 2.0]], [1.0, float("nan")]):
            samples = _samples(n=3, dim=2)
            samples[1][FIELD] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EmbeddingPCAProcessor().fit(samples, FIELD)

        with self.assertRaisesRegex(ValueError, "No samples"):
            EmbeddingPCAProcessor().fit([], FIELD)

    def test_process_before_fit_raises(self) -> None:
        processor = EmbeddingPCAProcessor(n_components=2)
        self.assertIsNone(processor.size())

        with self.assertRaisesRegex(RuntimeError, "before fit"):
            processor.process([1.0, 2.0])

    def test_fitted_on_train_split_only(self) -> None:
        samples = _samples()
        test_ids = ["p0", "p1"]
        for sample in samples[:2]:
            sample[FIELD] = [OUTLIER] * DIM

        builder = SampleBuilder(
            input_schema={FIELD: ("embedding_pca", {"n_components": None})},
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
        builder = SampleBuilder(
            input_schema={FIELD: ("embedding_pca", {"n_components": N_COMPONENTS})},
            output_schema={},
        )
        builder.fit(samples, split=_HoldoutSplit(["p0"]))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "schema.pkl")
            builder.save(path)
            loaded = SampleBuilder.load(path).input_processors[FIELD]
        original = builder.input_processors[FIELD]

        self.assertEqual(loaded.size(), N_COMPONENTS)
        torch.testing.assert_close(
            _transform_all(loaded, samples), _transform_all(original, samples)
        )


if __name__ == "__main__":
    unittest.main()
