"""Tests for dummy_model module."""

import unittest

import numpy as np
import pandas as pd


class TestDummyModel(unittest.TestCase):
    """Tests for the DummyModelGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "temp_test_dummy"

    def tearDown(self):
        """Clean up test fixtures."""
        import os
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dummy_model_generator_can_be_instantiated(self):
        """Test that DummyModelGenerator can be instantiated with valid ml_grid_object."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        # Create test CSV
        data = {
            "age": [25, 30, 35, 40, 45, 50],
            "sex_male": [0, 1, 0, 1, 0, 1],
            "some_blood_test_mean": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "outcome_var_1": [0, 1, 0, 1, 0, 1],
        }
        file_path = os.path.join(self.test_dir, "test.csv")
        os.makedirs(self.test_dir, exist_ok=True)
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 0

        local_param_dict = {
            "outcome_var_n": 1,
            "corr": 0.8,
            "percent_missing": 50,
            "scale": False,
            "n_features": "all",
            "resample": None,
            "data": {"age": True, "sex": True, "bloods": True},
            "param_space_size": 100,
        }

        pipeline = pipe(
            global_params=global_params,
            file_name=file_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=self.test_dir,
            param_space_index=0,
        )

        from ml_grid.model_classes_ga.dummy_model import DummyModelGenerator

        dummy_gen = DummyModelGenerator(pipeline, local_param_dict)

        self.assertIsNotNone(dummy_gen.fitted_perceptron)
        self.assertIsInstance(dummy_gen.dummy_columns, list)

    def test_dummy_model_gen_returns_valid_tuple(self):
        """Test that dummy_model_gen returns a valid result tuple."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        data = {
            "age": [25, 30, 35, 40, 45, 50],
            "sex_male": [0, 1, 0, 1, 0, 1],
            "some_blood_test_mean": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "outcome_var_1": [0, 1, 0, 1, 0, 1],
        }
        file_path = os.path.join(self.test_dir, "test.csv")
        os.makedirs(self.test_dir, exist_ok=True)
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 0

        local_param_dict = {
            "outcome_var_n": 1,
            "corr": 0.8,
            "percent_missing": 50,
            "scale": False,
            "n_features": "all",
            "resample": None,
            "data": {"age": True, "sex": True, "bloods": True},
            "param_space_size": 100,
        }

        pipeline = pipe(
            global_params=global_params,
            file_name=file_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=self.test_dir,
            param_space_index=0,
        )

        from ml_grid.model_classes_ga.dummy_model import DummyModelGenerator

        dummy_gen = DummyModelGenerator(pipeline, local_param_dict)
        result = dummy_gen.dummy_model_gen(pipeline, local_param_dict)

        self.assertEqual(len(result), 6)

        # Check tuple elements
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(mccscore, float)
        self.assertIsNotNone(model)
        self.assertIsInstance(feature_names, list)
        self.assertIsInstance(train_time, int)
        self.assertIsInstance(auc_score, float)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(pipeline.y_test))


if __name__ == "__main__":

    unittest.main()
