"""Tests for gaussianNB_model module."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd


class TestGaussianNBModelGenerator:
    """Tests for the GaussianNB_ModelGenerator function."""

    def test_gaussian_nb_basic_functionality(self):
        """Test basic functionality of GaussianNB_ModelGenerator."""
        from ml_grid.model_classes_ga.gaussianNB_model import (
            GaussianNB_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "age": np.random.randint(25, 65, n_samples),
                "sex_male": np.random.randint(0, 2, n_samples),
                "some_blood_test_mean": np.random.rand(n_samples),
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

            result = GaussianNB_ModelGenerator(pipeline, local_param_dict)

            assert len(result) == 6
            mccscore, model, feature_names, train_time, auc_score, y_pred = result

            assert isinstance(mccscore, float)
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

    def test_gaussian_nb_with_verbose_logging(self):
        """Test that verbose >= 2 triggers debug_base_learner call."""
        from ml_grid.model_classes_ga.gaussianNB_model import (
            GaussianNB_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "age": np.random.randint(25, 65, n_samples),
                "sex_male": np.random.randint(0, 2, n_samples),
                "some_blood_test_mean": np.random.rand(n_samples),
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

            result = GaussianNB_ModelGenerator(pipeline, local_param_dict)

            assert len(result) == 6
            mccscore, model, feature_names, train_time, auc_score, y_pred = result

            assert isinstance(mccscore, float)
            assert model is not None
            assert isinstance(feature_names, list)
            assert isinstance(train_time, int)
            assert isinstance(auc_score, float)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_gaussian_nb_different_param_space_sizes(self):
        """Test GaussianNB_ModelGenerator with different param_space_size values."""
        from ml_grid.model_classes_ga.gaussianNB_model import (
            GaussianNB_ModelGenerator,
        )
        from ml_grid.pipeline.data import pipe
        from ml_grid.util.global_params import global_parameters

        test_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            n_samples = 100
            data = {
                "age": np.random.randint(25, 65, n_samples),
                "sex_male": np.random.randint(0, 2, n_samples),
                "some_blood_test_mean": np.random.rand(n_samples),
                "outcome_var_1": np.random.randint(0, 2, n_samples),
            }
            file_path = os.path.join(test_dir, "test.csv")
            pd.DataFrame(data).to_csv(file_path, index=False)

            global_params = global_parameters(
                config_path="non_existent_file.yml", testing=True
            )
            global_params.verbose = 0

            for param_space_size in [10, 50, 100]:
                local_param_dict = {
                    "outcome_var_n": 1,
                    "corr": 0.8,
                    "percent_missing": 50,
                    "scale": False,
                    "n_features": "all",
                    "resample": None,
                    "data": {"age": True, "sex": True, "bloods": True},
                    "param_space_size": param_space_size,
                }

                pipeline = pipe(
                    global_params=global_params,
                    file_name=file_path,
                    drop_term_list=[],
                    local_param_dict=local_param_dict,
                    base_project_dir=test_dir,
                    param_space_index=0,
                )

                result = GaussianNB_ModelGenerator(pipeline, local_param_dict)

                assert len(result) == 6
                mccscore, model, feature_names, train_time, auc_score, y_pred = result

                assert isinstance(mccscore, float)
                assert model is not None
                assert isinstance(y_pred, np.ndarray)
                assert len(y_pred) == len(pipeline.y_test)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
