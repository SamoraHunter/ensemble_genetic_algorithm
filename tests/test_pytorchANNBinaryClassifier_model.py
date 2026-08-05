"""Tests for pytorchANNBinaryClassifier_model module."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch


class TestPytorchBinaryclassifierModelGenerator:
    """Tests for the Pytorch_binary_class_ModelGenerator function."""

    def test_pytorch_ann_basic_functionality(self):
        """Test basic functionality of Pytorch_binary_class_ModelGenerator."""
        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            Pytorch_binary_class_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "feature1": np.random.rand(n_samples),
                "feature2": np.random.rand(n_samples),
                "feature3": np.random.randint(0, 5, n_samples),
                "outcome_var_1": np.random.randint(0, 2, n_samples),
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
                "data": {"feature1": True, "feature2": True, "feature3": True},
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

            assert len(result) == 6
            mccscore, model, feature_names, train_time, auc_score, y_pred = result

            assert isinstance(mccscore, float)
            assert -1 <= mccscore <= 1
            assert model is not None
            assert isinstance(feature_names, list)
            assert len(feature_names) > 0
            assert isinstance(train_time, int)
            assert train_time >= 0
            assert isinstance(auc_score, float)
            assert y_pred is not None
            assert len(y_pred) == len(pipeline.y_test)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_pytorch_ann_with_scale_true(self):
        """Test Pytorch_binary_class_ModelGenerator with scale=True."""
        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            Pytorch_binary_class_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "feature1": np.random.rand(n_samples),
                "feature2": np.random.rand(n_samples),
                "outcome_var_1": np.random.randint(0, 2, n_samples),
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
                "data": {"feature1": True, "feature2": True},
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

            assert len(result) == 6
            mccscore, model, feature_names, train_time, auc_score, y_pred = result

            assert isinstance(mccscore, float)
            assert -1 <= mccscore <= 1
            assert model is not None
            assert isinstance(y_pred, np.ndarray)
            assert len(y_pred) == len(pipeline.y_test)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_pytorch_ann_returns_correct_tuple_structure(self):
        """Test that Pytorch_binary_class_ModelGenerator returns a valid 6-element tuple."""
        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            Pytorch_binary_class_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "feature1": np.random.rand(n_samples),
                "feature2": np.random.rand(n_samples),
                "outcome_var_1": np.random.randint(0, 2, n_samples),
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
                "data": {"feature1": True, "feature2": True},
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

            assert isinstance(result, tuple)
            assert len(result) == 6

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_predict_with_fallback_handles_nan(self):
        """Test predict_with_fallback handles NaN in model output."""
        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            predict_with_fallback,
        )

        mock_model = torch.nn.Linear(10, 2)
        input_data = torch.rand(5, 10)
        target_data = torch.rand(5)

        y_pred = predict_with_fallback(mock_model, input_data, target_data)

        assert not torch.isnan(y_pred).any()

    def test_predict_with_fallback_on_error(self):
        """Test predict_with_fallback returns zero logits on exception."""
        from unittest import mock

        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            predict_with_fallback,
        )

        mock_model = mock.Mock()
        mock_model.side_effect = RuntimeError("test error")
        input_data = torch.rand(5, 10)
        target_data = torch.rand(5)

        y_pred = predict_with_fallback(mock_model, input_data, target_data)

        expected_shape = (5, 1)
        assert y_pred.shape == expected_shape
        assert torch.all(y_pred == 0)

    def test_pytorch_ann_with_verbose_logging(self):
        """Test verbose >= 3 triggers detailed training logging."""
        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            Pytorch_binary_class_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 50
            data = {
                "feature1": np.random.rand(n_samples),
                "feature2": np.random.rand(n_samples),
                "outcome_var_1": np.random.randint(0, 2, n_samples),
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
                "data": {"feature1": True, "feature2": True},
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

            assert len(result) == 6
            mccscore, model, feature_names, train_time, auc_score, y_pred = result

            assert isinstance(mccscore, float)
            assert -1 <= mccscore <= 1
            assert model is not None
            assert isinstance(y_pred, np.ndarray)
            assert len(y_pred) == len(pipeline.y_test)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_pytorch_ann_with_store_base_learners(self):
        """Test Pytorch_binary_class_ModelGenerator with store_base_learners=True."""
        from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
            Pytorch_binary_class_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "feature1": np.random.rand(n_samples),
                "feature2": np.random.rand(n_samples),
                "outcome_var_1": np.random.randint(0, 2, n_samples),
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
                "corr": 0.8,
                "percent_missing": 50,
                "scale": False,
                "n_features": "all",
                "resample": None,
                "data": {"feature1": True, "feature2": True},
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

            assert len(result) == 6
            mccscore, model, feature_names, train_time, auc_score, y_pred = result

            assert isinstance(mccscore, float)
            assert model is not None
            assert isinstance(y_pred, np.ndarray)
            assert len(y_pred) == len(pipeline.y_test)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
