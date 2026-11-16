"""Tests for CoxSurvivalModel. Synthetic tensors only; each test runs in ms."""

import unittest

TODO = "TODO: implement"


class TestCoxSurvivalModel(unittest.TestCase):
    # TODO: setUp with create_sample_dataset() on ~8 synthetic patients:
    # dense "expr"/"hist" fields, "duration"/"event" tensor outputs, at least
    # one event and one censored patient.

    @unittest.skip(TODO)
    def test_initialization_linear_and_mlp(self):
        pass

    @unittest.skip(TODO)
    def test_mode_is_none(self):
        pass

    @unittest.skip(TODO)
    def test_invalid_keys_raise(self):
        pass

    @unittest.skip(TODO)
    def test_forward_output_keys_and_shapes(self):
        pass

    @unittest.skip(TODO)
    def test_forward_without_labels_has_no_loss(self):
        pass

    @unittest.skip(TODO)
    def test_feature_keys_subset_for_unimodal_model(self):
        pass

    @unittest.skip(TODO)
    def test_loss_finite_and_gradients_flow(self):
        pass

    @unittest.skip(TODO)
    def test_all_censored_batch_has_zero_loss(self):
        pass

    @unittest.skip(TODO)
    def test_training_on_known_signal_beats_random_c_index(self):
        pass


if __name__ == "__main__":
    unittest.main()
