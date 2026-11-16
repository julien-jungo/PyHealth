"""Tests for TCGAMultimodalDataset and MultimodalSurvivalPrediction.

Synthetic data only: 2-5 patients written to a temporary directory.
"""

import unittest

TODO = "TODO: implement"


class TestTCGAMultimodalDataset(unittest.TestCase):
    # TODO: setUpClass writes clinical.csv and embeddings/{expr,hist,text}
    # .parquet for 4 synthetic patients into a tempfile.TemporaryDirectory;
    # tearDownClass cleans it up.

    @unittest.skip(TODO)
    def test_loads_all_patients(self):
        pass

    @unittest.skip(TODO)
    def test_patient_has_clinical_and_embedding_events(self):
        pass

    @unittest.skip(TODO)
    def test_embedding_round_trips_as_float_list(self):
        pass

    @unittest.skip(TODO)
    def test_missing_file_raises(self):
        pass


class TestMultimodalSurvivalPrediction(unittest.TestCase):
    @unittest.skip(TODO)
    def test_schema(self):
        pass

    @unittest.skip(TODO)
    def test_dead_patient_duration_and_event(self):
        pass

    @unittest.skip(TODO)
    def test_censored_patient_duration_and_event(self):
        pass

    @unittest.skip(TODO)
    def test_demo_and_canc_tokens(self):
        pass

    @unittest.skip(TODO)
    def test_patient_missing_modality_skipped(self):
        pass

    @unittest.skip(TODO)
    def test_patient_missing_outcome_skipped(self):
        pass

    @unittest.skip(TODO)
    def test_set_task_with_split_fits_on_train_only(self):
        pass


if __name__ == "__main__":
    unittest.main()
