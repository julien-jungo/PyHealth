"""Tests for MultiHotIgnoreUnknownProcessor. Synthetic tokens only; runs in ms."""

import logging
import unittest
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from pyhealth.datasets import Split
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.processors import (
    PROCESSOR_REGISTRY,
    MultiHotIgnoreUnknownProcessor,
    MultiHotProcessor,
)

FIELD = "demo"
LOGGER_NAME = "pyhealth.processors.multi_hot_ignore_unknown_processor"
TRAIN_SAMPLES = [
    {"patient_id": "p0", FIELD: ["sex:female", "race:white"]},
    {"patient_id": "p1", FIELD: ["sex:male", "race:asian"]},
    {"patient_id": "p2", FIELD: ["sex:female", "race:asian"]},
]
TEST_SAMPLE = {"patient_id": "p3", FIELD: ["sex:male", "race:black"]}


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


def _fitted() -> MultiHotIgnoreUnknownProcessor:
    processor = MultiHotIgnoreUnknownProcessor()
    processor.fit(TRAIN_SAMPLES, FIELD)
    return processor


class TestMultiHotIgnoreUnknownProcessor(unittest.TestCase):
    def test_registered_and_subclasses_multi_hot(self) -> None:
        self.assertIs(
            PROCESSOR_REGISTRY["multi_hot_ignore_unknown"],
            MultiHotIgnoreUnknownProcessor,
        )
        self.assertTrue(issubclass(MultiHotIgnoreUnknownProcessor, MultiHotProcessor))

    def test_known_tokens_match_parent(self) -> None:
        parent = MultiHotProcessor()
        parent.fit(TRAIN_SAMPLES, FIELD)
        processor = _fitted()

        for sample in TRAIN_SAMPLES:
            with self.subTest(sample=sample["patient_id"]):
                out = processor.process(sample[FIELD])
                self.assertTrue(torch.equal(out, parent.process(sample[FIELD])))
                self.assertEqual(out.dtype, torch.float32)

        self.assertEqual(processor.size(), parent.size())

    def test_unseen_token_gives_zeros(self) -> None:
        out = _fitted().process(["race:black"])

        self.assertTrue(torch.equal(out, torch.zeros(4)))

    def test_mix_of_known_and_unseen(self) -> None:
        processor = _fitted()
        out = processor.process(["sex:male", "race:black", "unseen"])

        expected = torch.zeros(processor.size())
        expected[processor.label_vocab["sex:male"]] = 1.0
        self.assertTrue(torch.equal(out, expected))

    def test_empty_list_gives_zeros(self) -> None:
        self.assertTrue(torch.equal(_fitted().process([]), torch.zeros(4)))

    def test_non_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            _fitted().process("sex:male")

    def test_parent_raises_on_unseen_token(self) -> None:
        parent = MultiHotProcessor()
        parent.fit(TRAIN_SAMPLES, FIELD)

        with self.assertRaises(KeyError):
            parent.process(["race:black"])

    def test_unseen_token_logged_once(self) -> None:
        processor = _fitted()

        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as logs:
            for _ in range(3):
                processor.process(["race:black", "race:black", "race:other"])

        self.assertEqual(len(logs.records), 2)
        self.assertIn("'race:black'", logs.output[0])
        self.assertIn("'race:other'", logs.output[1])

    def test_known_tokens_not_logged(self) -> None:
        processor = _fitted()

        with self.assertNoLogs(LOGGER_NAME, level=logging.DEBUG):
            processor.process(["sex:male", "race:asian"])

    def test_fitted_on_train_split_only(self) -> None:
        builder = SampleBuilder(
            input_schema={FIELD: "multi_hot_ignore_unknown"}, output_schema={}
        )
        builder.fit([*TRAIN_SAMPLES, TEST_SAMPLE], split=_HoldoutSplit(["p3"]))
        processor = builder.input_processors[FIELD]

        self.assertIsInstance(processor, MultiHotIgnoreUnknownProcessor)
        self.assertNotIn("race:black", processor.label_vocab)
        self.assertEqual(processor.size(), 4)

        out = processor.process(TEST_SAMPLE[FIELD])
        self.assertEqual(out.sum().item(), 1.0)
        self.assertEqual(out[processor.label_vocab["sex:male"]].item(), 1.0)

    def test_repr(self) -> None:
        self.assertEqual(
            repr(_fitted()), "MultiHotIgnoreUnknownProcessor(label_vocab_size=4)"
        )


if __name__ == "__main__":
    unittest.main()
