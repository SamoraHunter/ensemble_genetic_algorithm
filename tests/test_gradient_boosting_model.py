"""Tests for gradientBoostingClassifier_model module."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


class TestGradientBoostingClassifierIntegration(unittest.TestCase):
    """Integration tests for GradientBoostingClassifier_ModelGenerator with real data."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_gradient_boosting_model_generator_returns_valid_result(self):
        """Test that GradientBoostingClassifier_ModelGenerator returns valid result tuple with real data."""
        from ml_grid.model_classes_ga.gradientBoostingClassifier_model import (
            GradientBoostingClassifier_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        # Create test CSV with realistic binary classification data
        np.random.seed(42)
        n_samples = 100
        data = {
            "feature1": np.random.rand(n_samples),
            "feature2": np.random.rand(n_samples),
            "feature3": np.random.randint(0, 5, n_samples),
            "outcome_var_1": np.random.randint(0, 2, n_samples),
        }
        file_path = os.path.join(self.test_dir, "test.csv")
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
            "data": {"feature1": True, "feature2": True, "feature3": True},
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

        result = GradientBoostingClassifier_ModelGenerator(pipeline, local_param_dict)

        # Verify return tuple structure
        self.assertEqual(len(result), 6)
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(mccscore, float)
        self.assertIsNotNone(model)
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertIsInstance(train_time, int)
        self.assertGreaterEqual(train_time, 0)
        self.assertIsInstance(auc_score, float)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(pipeline.y_test))
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))
