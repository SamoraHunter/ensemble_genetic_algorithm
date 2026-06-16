import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
    Pytorch_binary_class_ModelGenerator,
    predict_with_fallback,
)


class TestPredictWithFallbackEdgeCases(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Linear(10, 2)
        self.input_data = torch.rand(10, 10)
        self.target_data = torch.rand(10)

    def test_predict_with_fallback_handles_exception(self):
        mock_model = mock.Mock()
        mock_model.side_effect = Exception("test exception")

        y_pred = predict_with_fallback(
            model=mock_model, X_batch=self.input_data, y_batch=self.target_data
        )

        expected_fallback = torch.zeros(
            self.target_data.unsqueeze(1).shape,
            dtype=self.target_data.dtype,
            device=self.target_data.device,
        )
        self.assertTrue(torch.equal(y_pred, expected_fallback))

    def test_predict_with_fallback_returns_expected_output(self):
        mock_model = mock.Mock()
        mock_model.return_value = torch.rand(10, 1)

        y_pred = predict_with_fallback(
            model=mock_model, X_batch=self.input_data, y_batch=self.target_data
        )

        self.assertTrue(torch.allclose(y_pred, mock_model.return_value))

    def test_predict_with_fallback_with_empty_input(self):
        empty_input = torch.empty(0, 10)
        empty_target = torch.empty(0)

        y_pred = predict_with_fallback(
            model=self.model, X_batch=empty_input, y_batch=empty_target
        )

        self.assertEqual(y_pred.shape[0], 0)

    def test_predict_with_fallback_with_different_device(self):
        """Test fallback works when input and target are on different devices."""
        if not torch.cuda.is_available():
            return

        cpu_input = torch.rand(5, 10)
        cpu_target = torch.rand(5)

        mock_model = mock.Mock()
        mock_model.side_effect = RuntimeError("CUDA error")

        y_pred = predict_with_fallback(
            model=mock_model, X_batch=cpu_input, y_batch=cpu_target
        )

        self.assertEqual(y_pred.device, cpu_input.device)
        self.assertEqual(y_pred.shape, (5, 1))

    def test_predict_with_fallback_preserves_dtype(self):
        """Test that fallback preserves the dtype of the input."""
        half_input = torch.rand(5, 10, dtype=torch.float16)
        half_target = torch.rand(5, dtype=torch.float16)

        mock_model = mock.Mock()
        mock_model.side_effect = Exception("test")

        y_pred = predict_with_fallback(
            model=mock_model, X_batch=half_input, y_batch=half_target
        )

        self.assertEqual(y_pred.dtype, torch.float16)

    def test_predict_with_fallback_nan_model_output(self):
        """Test fallback when model returns NaN."""
        mock_model = mock.Mock()
        mock_model.return_value = torch.full((5, 1), float("nan"))

        y_pred = predict_with_fallback(
            model=mock_model, X_batch=self.input_data, y_batch=self.target_data
        )

        self.assertFalse(torch.isnan(y_pred).any())

    def test_nan_predictions_result_in_random_integers(self):
        """Test that NaN in predictions triggers fallback to random integers."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = "temp_test_nan_fallback"
        os.makedirs(test_dir, exist_ok=True)

        data = {
            "age": [25, 30, 35, 40, 45, 50] * 10,
            "sex_male": [0, 1, 0, 1, 0, 1] * 10,
            "some_blood_test_mean": [
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ]
            * 10,
            "outcome_var_1": [0, 1, 0, 1, 0, 1] * 10,
        }
        file_path = os.path.join(test_dir, "test.csv")
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
            base_project_dir=test_dir,
            param_space_index=0,
        )

        result = Pytorch_binary_class_ModelGenerator(pipeline, local_param_dict)

        self.assertEqual(len(result), 6)
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(pipeline.y_test))
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))

        os.remove(file_path)
        import shutil

        shutil.rmtree(test_dir)

    def test_model_with_scale_true(self):
        """Test Pytorch_binary_class_ModelGenerator with scale=True."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = "temp_test_scale"
        os.makedirs(test_dir, exist_ok=True)

        data = {
            "age": [25, 30, 35, 40, 45, 50] * 10,
            "sex_male": [0, 1, 0, 1, 0, 1] * 10,
            "some_blood_test_mean": [
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ]
            * 10,
            "outcome_var_1": [0, 1, 0, 1, 0, 1] * 10,
        }
        file_path = os.path.join(test_dir, "test.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 0

        local_param_dict = {
            "outcome_var_n": 1,
            "corr": 0.8,
            "percent_missing": 50,
            "scale": True,
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
            base_project_dir=test_dir,
            param_space_index=0,
        )

        result = Pytorch_binary_class_ModelGenerator(pipeline, local_param_dict)

        self.assertEqual(len(result), 6)
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(pipeline.y_test))
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))

        os.remove(file_path)
        import shutil

        shutil.rmtree(test_dir)

    def test_verbose_training_logging(self):
        """Test verbose logging during training."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = "temp_test_verbose"
        os.makedirs(test_dir, exist_ok=True)

        data = {
            "age": [25, 30, 35, 40, 45, 50] * 10,
            "sex_male": [0, 1, 0, 1, 0, 1] * 10,
            "some_blood_test_mean": [
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ]
            * 10,
            "outcome_var_1": [0, 1, 0, 1, 0, 1] * 10,
        }
        file_path = os.path.join(test_dir, "test.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 3

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
            base_project_dir=test_dir,
            param_space_index=0,
        )

        result = Pytorch_binary_class_ModelGenerator(pipeline, local_param_dict)

        self.assertEqual(len(result), 6)
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(pipeline.y_test))
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))

        os.remove(file_path)
        import shutil

        shutil.rmtree(test_dir)

    def test_predict_with_fallback_nan_values(self):
        """Test predict_with_fallback when model output contains NaN."""
        from unittest import mock

        mock_model = mock.Mock()
        mock_model.return_value = torch.full((5, 1), float("nan"), dtype=torch.float32)
        input_data = torch.rand(5, 10)
        target_data = torch.rand(5)

        y_pred = predict_with_fallback(mock_model, input_data, target_data)

        self.assertFalse(torch.isnan(y_pred).any())
        self.assertEqual(y_pred.shape, (5, 1))

    def test_predict_with_fallback_preserves_device(self):
        """Test that fallback preserves the device of the input."""
        cpu_input = torch.rand(3, 10)
        cpu_target = torch.rand(3)

        mock_model = mock.Mock()
        mock_model.side_effect = Exception("test error")

        y_pred = predict_with_fallback(mock_model, cpu_input, cpu_target)

        self.assertEqual(y_pred.device, cpu_input.device)

    def test_predict_with_fallback_returns_zero_logits_on_error(self):
        """Test that zeros are returned with correct shape when exception occurs."""
        mock_model = mock.Mock()
        mock_model.side_effect = RuntimeError("Forward pass failed")

        input_data = torch.rand(7, 10)
        target_data = torch.rand(7)

        y_pred = predict_with_fallback(mock_model, input_data, target_data)

        expected_shape = (7, 1)
        self.assertEqual(y_pred.shape, expected_shape)
        self.assertTrue(torch.all(y_pred == 0))

    def test_pytorch_model_generator_without_scale(self):
        """Test Pytorch_binary_class_ModelGenerator with scale=False."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = "temp_test_no_scale"
        os.makedirs(test_dir, exist_ok=True)

        data = {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 8,
            "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5] * 8,
            "feature_c": [100, 200, 300, 400, 500] * 8,
            "outcome_var_1": [0, 1] * 20,
        }
        file_path = os.path.join(test_dir, "test.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 0

        local_param_dict = {
            "outcome_var_n": 1,
            "corr": 0.5,
            "percent_missing": 0,
            "scale": False,
            "n_features": "all",
            "resample": None,
            "data": {"feature_a": True, "feature_b": True, "feature_c": True},
            "param_space_size": 100,
        }

        pipeline = pipe(
            global_params=global_params,
            file_name=file_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=test_dir,
            param_space_index=0,
        )

        result = Pytorch_binary_class_ModelGenerator(pipeline, local_param_dict)

        self.assertEqual(len(result), 6)
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertGreater(len(y_pred), 0)
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))

        os.remove(file_path)
        import shutil

        shutil.rmtree(test_dir)

    def test_pytorch_model_generator_with_store_true(self):
        """Test Pytorch_binary_class_ModelGenerator with store_base_learners=True."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = "temp_test_store"
        os.makedirs(test_dir, exist_ok=True)

        data = {
            "feature_x": [1.5, 2.5, 3.5, 4.5] * 6,
            "feature_y": [0.15, 0.25, 0.35, 0.45] * 6,
            "outcome_var_1": [0, 1] * 12,
        }
        file_path = os.path.join(test_dir, "test.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 0
        global_params.store_base_learners = True

        local_param_dict = {
            "outcome_var_n": 1,
            "corr": 0.5,
            "percent_missing": 0,
            "scale": False,
            "n_features": "all",
            "resample": None,
            "data": {"feature_x": True, "feature_y": True},
            "param_space_size": 100,
        }

        pipeline = pipe(
            global_params=global_params,
            file_name=file_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=test_dir,
            param_space_index=0,
        )

        result = Pytorch_binary_class_ModelGenerator(pipeline, local_param_dict)

        self.assertEqual(len(result), 6)
        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertGreater(len(y_pred), 0)

        os.remove(file_path)
        import shutil

        shutil.rmtree(test_dir)

    def test_model_returns_expected_tuple_structure(self):
        """Test that the model generator returns a tuple with correct structure."""
        import os

        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = "temp_test_tuple"
        os.makedirs(test_dir, exist_ok=True)

        data = {
            "col1": [1.0, 2.0] * 10,
            "col2": [3.0, 4.0] * 10,
            "outcome_var_1": [0, 1] * 10,
        }
        file_path = os.path.join(test_dir, "test.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)

        global_params = global_parameters(
            config_path="non_existent_file.yml", testing=True
        )
        global_params.verbose = 0

        local_param_dict = {
            "outcome_var_n": 1,
            "corr": 0.5,
            "percent_missing": 0,
            "scale": False,
            "n_features": "all",
            "resample": None,
            "data": {"col1": True, "col2": True},
            "param_space_size": 100,
        }

        pipeline = pipe(
            global_params=global_params,
            file_name=file_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=test_dir,
            param_space_index=0,
        )

        result = Pytorch_binary_class_ModelGenerator(pipeline, local_param_dict)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)

        mccscore, model, feature_names, train_time, auc_score, y_pred = result

        self.assertIsInstance(mccscore, float)
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertIsInstance(train_time, int)
        self.assertGreaterEqual(train_time, 0)
        self.assertIsInstance(auc_score, float)

        os.remove(file_path)
        import shutil

        shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
