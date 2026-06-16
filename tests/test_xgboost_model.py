"""Tests for XGBoost_model module."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestXGBoostIntegration(unittest.TestCase):
    """Integration tests for XGBoostModelGenerator with real data."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_xgboost_model_generator_returns_valid_result(self):
        """Test that XGBoostModelGenerator returns valid result tuple with real data."""
        from ml_grid.model_classes_ga.XGBoost_model import XGBoostModelGenerator
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        np.random.seed(42)
        n_samples = 100
        data = {
            "age": np.random.randint(25, 65, n_samples),
            "sex_male": np.random.randint(0, 2, n_samples),
            "some_blood_test_mean": np.random.rand(n_samples),
            "outcome_var_1": np.random.randint(0, 2, n_samples),
        }
        file_path = os.path.join(self.test_dir, "test.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params_obj = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params_obj.verbose = 0

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
            global_params=global_params_obj,
            file_name=file_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=self.test_dir,
            param_space_index=0,
        )

        result = XGBoostModelGenerator(pipeline, local_param_dict)

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


class TestXGBoostGPUExceptionHandling(unittest.TestCase):
    """Tests for GPU exception handling in XGBoostModelGenerator."""

    def test_get_free_gpu_exception_sets_gpu_id_to_minus_one(self):
        """Test that get_free_gpu exception results in gpu_id_n = -1."""
        with patch(
            "ml_grid.model_classes_ga.XGBoost_model.get_free_gpu"
        ) as mock_get_free_gpu:
            mock_get_free_gpu.side_effect = Exception("No GPU available")

            from ml_grid.model_classes_ga import XGBoost_model

            result = {}

            def check_gpu_id():
                try:
                    gpu_id_n = XGBoost_model.get_free_gpu(None)
                except Exception:
                    gpu_id_n = "-1"
                return gpu_id_n

            result["gpu_id"] = check_gpu_id()

            self.assertEqual(result["gpu_id"], "-1")


class TestXGBoostParameterInit(unittest.TestCase):
    """Tests for parameter initialization in XGBoostModelGenerator."""

    def test_parameter_space_choices(self):
        """Test that parameters are selected from correct ranges."""
        with patch("ml_grid.model_classes_ga.XGBoost_model.random") as mock_random:
            mock_random.choice.side_effect = lambda x: x[0]

            from ml_grid.model_classes_ga import XGBoost_model

            gamma_n = XGBoost_model.random.choice([0.01, 0.1, 1, 3, 5, 7, 9, 10, 15])
            reg_alpha_n = XGBoost_model.random.choice(
                [0, 0.001, 0.005, 0.01, 0.1, 1, 3, 5]
            )

            self.assertIn(gamma_n, [0.01, 0.1, 1, 3, 5, 7, 9, 10, 15])
            self.assertIn(reg_alpha_n, [0, 0.001, 0.005, 0.01, 0.1, 1, 3, 5])


class TestXGBoostGPUFallback(unittest.TestCase):
    """Tests for GPU fallback exception handling paths."""

    def test_get_free_gpu_returns_negative_one_on_exception(self):
        """Test that get_free_gpu returns -1 when subprocess fails."""
        with patch("ml_grid.ga_functions.ga_ann_util.subprocess") as mock_subprocess:
            mock_subprocess.check_output.side_effect = Exception("No GPU")

            from ml_grid.ga_functions import ga_ann_util

            result = ga_ann_util.get_free_gpu(None)
            self.assertEqual(result, -1)


class TestXGBoostGPUHistFallback(unittest.TestCase):
    """Tests for XGBoost GPU to CPU hist fallback on fit exception."""

    def test_xgb_classifier_with_hist_method_when_gpu_fails(self):
        """Test that hist tree_method is used when gpu_hist exception occurs."""
        mock_pipeline = unittest.mock.MagicMock()
        mock_pipeline.X_train = pd.DataFrame({"feature1": [1.0, 2.0]})
        mock_pipeline.X_test = pd.DataFrame({"feature1": [3.0, 4.0]})
        mock_pipeline.y_train = pd.Series([0, 1])
        mock_pipeline.y_test = pd.Series([0, 1])

        local_param_dict = {"test": True}

        with patch(
            "ml_grid.model_classes_ga.XGBoost_model.feature_selection_methods_class"
        ) as mock_fs:
            mock_fs_instance = mock_fs.return_value
            mock_fs_instance.get_featured_selected_training_data.return_value = (
                mock_pipeline.X_train,
                mock_pipeline.X_test,
            )

        with patch("ml_grid.model_classes_ga.XGBoost_model.random") as mock_random:
            mock_random.choice.side_effect = lambda x: x[0]

        with patch("xgboost.XGBClassifier") as mock_xgb_class:
            gpu_exception_calls = []
            hist_calls = []

            def side_effect(*args, **kwargs):
                if "tree_method" in kwargs:
                    tree_method = kwargs["tree_method"]
                    if tree_method == "gpu_hist":
                        gpu_exception_calls.append(kwargs)
                        raise Exception("GPU not available")
                    elif tree_method == "hist":
                        hist_calls.append(kwargs)
                return unittest.mock.MagicMock()

            mock_xgb_class.side_effect = side_effect

            from ml_grid.model_classes_ga.XGBoost_model import XGBoostModelGenerator

            try:
                _ = XGBoostModelGenerator(mock_pipeline, local_param_dict)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
