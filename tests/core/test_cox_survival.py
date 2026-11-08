"""Tests for CoxSurvivalModel. Synthetic tensors only; each test runs in ms."""

import unittest
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import concordance_index
from pyhealth.models import CoxSurvivalModel
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.trainer import Trainer

EXPR_DIM = 3
HIST_DIM = 2
CANCER_TYPES = ["TCGA-OV", "TCGA-UCS"]
N_PATIENTS = 8
SIGNAL_PATIENTS = 64
SIGNAL_EPOCHS = 50
SIGNAL_LR = 5e-2
MIN_SIGNAL_C_INDEX = 0.7  # an oracle ranking by expr[0] scores ~0.77

INPUT_SCHEMA = {"expr": "tensor", "hist": "tensor", "canc": "multi_hot"}
OUTPUT_SCHEMA = {"duration": "tensor", "event": "tensor"}


def _samples(n: int, seed: int = 0) -> list[dict]:
    """Patients whose log hazard is ``expr[0]``; every third is censored."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        expr = rng.normal(size=EXPR_DIM)
        duration = rng.exponential(scale=np.exp(-expr[0])) * 100.0 + 1.0
        samples.append(
            {
                "patient_id": f"p{i}",
                "expr": expr.tolist(),
                "hist": rng.normal(size=HIST_DIM).tolist(),
                "canc": [f"project:{CANCER_TYPES[i % len(CANCER_TYPES)]}"],
                "duration": [float(duration)],
                "event": [0.0 if i % 3 == 2 else 1.0],
            }
        )
    return samples


def _dataset(samples: list[dict], input_schema: dict = INPUT_SCHEMA):
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=OUTPUT_SCHEMA,
        dataset_name="synthetic_cox",
    )


class _NoSizeProcessor(FeatureProcessor):
    """A dense vector processor without ``size()``, like the image processor."""

    def fit(self, samples: Any, field: str) -> None:
        pass

    def process(self, value: Any) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)


class _UnsizedProcessor(_NoSizeProcessor):
    """A dense vector processor whose width is unknown."""

    def size(self) -> None:
        return None


class _NotFittedProcessor(_NoSizeProcessor):
    """A processor whose ``size()`` is unavailable, like an unfitted one."""

    def size(self) -> int:
        raise NotImplementedError("call fit() first")


class _NoValueProcessor(_NoSizeProcessor):
    """A processor whose output has no ``"value"`` part."""

    def size(self) -> int:
        return 2

    def schema(self) -> tuple[str, ...]:
        return ("mask",)


class TestCoxSurvivalModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset(_samples(N_PATIENTS))
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=N_PATIENTS)))

    def test_initialization(self):
        model = CoxSurvivalModel(self.dataset)
        self.assertEqual(model.feature_keys, ["expr", "hist", "canc"])
        self.assertEqual(model.duration_key, "duration")
        self.assertEqual(model.event_key, "event")
        self.assertIsInstance(model.head, torch.nn.Linear)
        self.assertEqual(
            model.head.in_features, EXPR_DIM + HIST_DIM + len(CANCER_TYPES)
        )
        self.assertEqual(model.head.out_features, 1)
        self.assertIsNone(model.head.bias)  # Cox loss is shift-invariant

    def test_mode_is_none(self):
        self.assertIsNone(CoxSurvivalModel(self.dataset).mode)

    def test_invalid_keys_raise(self):
        invalid = [
            {"feature_keys": ["missing"]},
            {"feature_keys": ["duration"]},
            {"feature_keys": []},
            {"duration_key": "missing"},
            {"event_key": "expr"},
        ]
        for kwargs in invalid:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                CoxSurvivalModel(self.dataset, **kwargs)

    def test_forward_output_keys_and_shapes(self):
        out = CoxSurvivalModel(self.dataset)(**self.batch)

        self.assertEqual(set(out), {"risk", "logit", "y_prob", "loss", "y_true"})
        self.assertEqual(out["risk"].shape, (N_PATIENTS,))
        self.assertEqual(out["logit"].shape, (N_PATIENTS, 1))
        self.assertEqual(out["y_prob"].shape, (N_PATIENTS, 1))
        self.assertEqual(out["y_true"].shape, (N_PATIENTS, 2))
        self.assertEqual(out["loss"].dim(), 0)
        torch.testing.assert_close(out["y_prob"].squeeze(-1), out["risk"])
        torch.testing.assert_close(
            out["y_true"],
            torch.cat([self.batch["duration"], self.batch["event"]], dim=-1),
        )

    def test_forward_without_labels_has_no_loss(self):
        features = {key: self.batch[key] for key in INPUT_SCHEMA}
        out = CoxSurvivalModel(self.dataset)(**features)
        self.assertEqual(set(out), {"risk", "logit", "y_prob"})

    def test_feature_keys_subset_for_unimodal_model(self):
        model = CoxSurvivalModel(self.dataset, feature_keys=["hist"])
        self.assertEqual(model.feature_keys, ["hist"])
        self.assertEqual(model.head.in_features, HIST_DIM)

        noisy = dict(
            self.batch,
            expr=torch.randn_like(self.batch["expr"]),
            canc=1.0 - self.batch["canc"],
        )
        torch.testing.assert_close(model(**noisy)["risk"], model(**self.batch)["risk"])

    def test_loss_finite_and_gradients_flow(self):
        model = CoxSurvivalModel(self.dataset)
        loss = model(**self.batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0.0)

    def test_all_censored_batch_has_zero_loss(self):
        model = CoxSurvivalModel(self.dataset)
        batch = dict(self.batch, event=torch.zeros_like(self.batch["event"]))

        loss = model(**batch)["loss"]
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(
            model.head.weight.grad, torch.zeros_like(model.head.weight)
        )

    def test_fields_that_are_not_dense_vectors_raise(self):
        samples = [
            dict(
                sample,
                codes=["a", "b", "c"][: 1 + i % 3],
                matrix=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                raw={"note": "free text"},
                no_size=[0.0, 1.0],
                unsized=[0.0, 1.0],
                not_fitted=[0.0, 1.0],
                no_value=[0.0, 1.0],
            )
            for i, sample in enumerate(_samples(N_PATIENTS))
        ]
        dataset = _dataset(
            samples,
            {
                **INPUT_SCHEMA,
                "codes": "sequence",
                "matrix": "tensor",
                "raw": "raw",
                "no_size": _NoSizeProcessor(),
                "unsized": _UnsizedProcessor(),
                "not_fitted": _NotFittedProcessor(),
                "no_value": _NoValueProcessor(),
            },
        )

        expected_errors = {
            "codes": "token indices",
            "matrix": "2-D values",
            "raw": "fixed width",
            "no_size": r"has no size\(\)",
            "unsized": "fixed width",
            "not_fitted": "not implemented or not fitted: call fit",
            "no_value": "no 'value' output",
        }
        for key, message in expected_errors.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                CoxSurvivalModel(dataset, feature_keys=[key])

    def test_training_on_known_signal_reaches_min_c_index(self):
        train = _dataset(_samples(SIGNAL_PATIENTS, seed=1))
        test = _dataset(_samples(SIGNAL_PATIENTS, seed=2))
        model = CoxSurvivalModel(train, feature_keys=["expr"])

        trainer = Trainer(model=model, enable_logging=False, device="cpu")
        # Full batch: the Cox risk sets then span the whole training set.
        trainer.train(
            get_dataloader(train, batch_size=SIGNAL_PATIENTS),
            epochs=SIGNAL_EPOCHS,
            optimizer_params={"lr": SIGNAL_LR},
        )
        y_true, y_prob, _ = trainer.inference(
            get_dataloader(test, batch_size=SIGNAL_PATIENTS)
        )

        c_index = concordance_index(y_true[:, 0], y_true[:, 1], y_prob[:, 0])
        self.assertGreater(c_index, MIN_SIGNAL_C_INDEX)


if __name__ == "__main__":
    unittest.main()
