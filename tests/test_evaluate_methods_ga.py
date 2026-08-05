"""Tests for evaluate_methods_ga module."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


class TestNormalize:
    """Tests for the normalize function."""

    def test_normalize_basic(self):
        """Test basic normalization of weights."""
        from ml_grid.pipeline.evaluate_methods_ga import (
            normalize,
        )

        weights = np.array([1.0, 2.0, 3.0])
        result = normalize(weights)

        expected = weights / 6.0
        np.testing.assert_allclose(result, expected)

    def test_normalize_already_normalized(self):
        """Test that already normalized weights stay the same."""
        from ml_grid.pipeline.evaluate_methods_ga import (
            normalize,
        )

        weights = np.array([0.2, 0.3, 0.5])
        result = normalize(weights)

        np.testing.assert_allclose(result, weights)

    def test_normalize_zero_vector(self):
        """Test normalization of zero vector (should return as-is)."""
        from ml_grid.pipeline.evaluate_methods_ga import (
            normalize,
        )

        weights = np.array([0.0, 0.0, 0.0])
        result = normalize(weights)

        np.testing.assert_array_equal(result, weights)


class TestGetYpredResolver:
    """Tests for the get_y_pred_resolver function."""

    def test_get_y_pred_resolver_svc_fallback(self):
        """Test get_y_pred_resolver handles SVC fitting error with fallback."""

        mod_key = "ml_grid.pipeline.evaluate_methods_ga"
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        from ml_grid.pipeline.evaluate_methods_ga import (
            get_y_pred_resolver,
        )

        ml_grid = MagicMock()
        type(ml_grid).verbose = 0
        ml_grid.local_param_dict = {"weighted": None}
        ml_grid.X_test_orig = np.array([[1, 2], [3, 4]])
        ml_grid.y_test = np.array([0, 1])

        with patch(
            "ml_grid.pipeline.evaluate_methods_ga.get_unweighted_ensemble_predictions"
        ) as mock_unweighted:
            mock_unweighted.side_effect = ValueError(
                "The dual coefficients or intercepts are not finite"
            )

            result = get_y_pred_resolver([[]], ml_grid, valid=False)

            assert len(result) == 2
            np.testing.assert_array_equal(result, np.zeros(2))

    def test_get_y_pred_resolver_de_weighted(self):
        """Test get_y_pred_resolver handles DE (Differential Evolution) weighted ensemble."""

        mod_key = "ml_grid.pipeline.evaluate_methods_ga"
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        from ml_grid.pipeline.evaluate_methods_ga import (
            get_y_pred_resolver,
        )

        ml_grid = MagicMock()
        type(ml_grid).verbose = 0
        ml_grid.local_param_dict = {"weighted": "de"}
        ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6]])
        ml_grid.y_test = np.array([0, 1, 0])

        with (
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.find_ensemble_weights_de"
            ) as mock_find_weights,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.get_weighted_ensemble_prediction_de_y_pred_valid"
            ) as mock_get_pred,
        ):
            mock_find_weights.return_value = np.array([0.3, 0.7])
            expected_pred = np.array([0.2, 0.8, 0.3])
            mock_get_pred.return_value = expected_pred

            result = get_y_pred_resolver(
                [[("model1", None, None, None, None, None)], []], ml_grid, valid=False
            )

            mock_find_weights.assert_called_once()
            mock_get_pred.assert_called_once()
            np.testing.assert_array_equal(result, expected_pred)

    def test_get_y_pred_resolver_linear_weighted(self):
        """Test get_y_pred_resolver handles linear-weighted ensemble."""

        mod_key = "ml_grid.pipeline.evaluate_methods_ga"
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        from ml_grid.pipeline.evaluate_methods_ga import (
            get_y_pred_resolver,
        )

        ml_grid = MagicMock()
        type(ml_grid).verbose = 0
        ml_grid.local_param_dict = {"weighted": "linear"}
        ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6]])
        ml_grid.y_test = np.array([0, 1, 0])

        with (
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.find_linear_weights"
            ) as mock_find_weights,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.get_linear_weighted_ensemble_predictions"
            ) as mock_get_pred,
        ):
            mock_find_weights.return_value = np.array([0.4, 0.6])
            expected_pred = np.array([0.3, 0.7, 0.2])
            mock_get_pred.return_value = expected_pred

            result = get_y_pred_resolver(
                [[("model1", None, None, None, None, None)], []], ml_grid, valid=False
            )

            mock_find_weights.assert_called_once()
            mock_get_pred.assert_called_once()
            np.testing.assert_array_equal(result, expected_pred)


class TestEvaluateWeightedEnsembleAuc:
    """Tests for the evaluate_weighted_ensemble_auc function."""

    def test_evaluate_weighted_ensemble_auc_basic(self):
        """Test basic functionality of evaluate_weighted_ensemble_auc."""

        mod_key = "ml_grid.pipeline.evaluate_methods_ga"
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        from ml_grid.pipeline.evaluate_methods_ga import (
            evaluate_weighted_ensemble_auc,
        )

        with (
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver"
            ) as mock_get_y_pred,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.roc_auc_score"
            ) as mock_auc,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.matthews_corrcoef"
            ) as mock_mcc,
            patch("ml_grid.pipeline.evaluate_methods_ga.metrics.f1_score") as mock_f1,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.precision_score"
            ) as mock_precision,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.recall_score"
            ) as mock_recall,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.accuracy_score"
            ) as mock_accuracy,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.measure_diversity_wrapper"
            ) as mock_measure_diversity,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.apply_diversity_penalty"
            ) as mock_apply_diversity,
            patch("ml_grid.pipeline.evaluate_methods_ga.pd.DataFrame.to_csv"),
        ):
            mock_get_y_pred.return_value = np.array([0, 1, 0, 1, 0])
            mock_auc.return_value = 0.85
            mock_mcc.return_value = 0.75
            mock_f1.return_value = 0.80
            mock_precision.return_value = 0.82
            mock_recall.return_value = 0.78
            mock_accuracy.return_value = 0.83
            mock_measure_diversity.return_value = 0.5
            mock_apply_diversity.return_value = (0.78, 0.68)

            global_params_mock = MagicMock()
            global_params_mock.verbose = 0

            ml_grid = MagicMock()
            type(ml_grid).verbose = 0
            ml_grid.y_test = np.array([0, 1, 0, 1, 0])
            ml_grid.local_param_dict = {"div_p": 0}
            ml_grid.original_feature_names = ["feat1", "feat2"]
            ml_grid.logging_paths_obj.log_folder_path = "/tmp/test"
            ml_grid.global_params = global_params_mock
            type(ml_grid.global_params).log_store_dataframe_path = "test.csv"

            mock_model = (
                0.7,
                "TestModel",
                ["feat1"],
                None,
                0.85,
                np.array([0, 1, 0, 1, 0]),
            )
            individual = [[mock_model], []]

            result = evaluate_weighted_ensemble_auc(individual, ml_grid)

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], (float, np.floating))
            assert result[0] == 0.85

    def test_evaluate_weighted_ensemble_auc_with_diversity_penalty(self):
        """Test evaluate_weighted_ensemble_auc with diversity penalty enabled."""

        mod_key = "ml_grid.pipeline.evaluate_methods_ga"
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        from ml_grid.pipeline.evaluate_methods_ga import (
            evaluate_weighted_ensemble_auc,
        )

        def apply_diversity_side_effect(auc, mcc, diversity_metric, params):
            if diversity_metric < 1.0:
                return (auc * 0.9, mcc * 0.9)
            return (auc, mcc)

        with (
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver"
            ) as mock_get_y_pred,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.measure_diversity_wrapper",
                return_value=0.3,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.apply_diversity_penalty"
            ) as mock_apply_diversity,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.roc_auc_score",
                return_value=0.85,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.matthews_corrcoef",
                return_value=0.75,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.f1_score",
                return_value=0.80,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.precision_score",
                return_value=0.82,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.recall_score",
                return_value=0.78,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.accuracy_score",
                return_value=0.83,
            ),
            patch("ml_grid.pipeline.evaluate_methods_ga.pd.DataFrame.to_csv"),
        ):
            mock_get_y_pred.return_value = np.array([0, 1, 0, 1, 0])
            mock_apply_diversity.side_effect = apply_diversity_side_effect

            global_params_mock = MagicMock()
            global_params_mock.verbose = 0

            ml_grid = MagicMock()
            type(ml_grid).verbose = 0
            ml_grid.y_test = np.array([0, 1, 0, 1, 0])
            ml_grid.local_param_dict = {"div_p": 0.5}
            ml_grid.original_feature_names = ["feat1", "feat2"]
            ml_grid.logging_paths_obj.log_folder_path = "/tmp/test"
            ml_grid.global_params = global_params_mock
            type(ml_grid.global_params).log_store_dataframe_path = "test.csv"

            mock_model = (
                0.7,
                "TestModel",
                ["feat1"],
                None,
                0.85,
                np.array([0, 1, 0, 1, 0]),
            )
            individual = [[mock_model], []]

            result = evaluate_weighted_ensemble_auc(individual, ml_grid)

            assert isinstance(result, tuple)
            assert len(result) == 1
            np.testing.assert_allclose(result[0], 0.765, rtol=1e-3)

    def test_evaluate_weighted_ensemble_auc_auc_error_handling(self):
        """Test evaluate_weighted_ensemble_auc handles ValueError in AUC calculation."""

        mod_key = "ml_grid.pipeline.evaluate_methods_ga"
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        from ml_grid.pipeline.evaluate_methods_ga import (
            evaluate_weighted_ensemble_auc,
        )

        with (
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver"
            ) as mock_get_y_pred,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.roc_auc_score"
            ) as mock_auc,
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.measure_diversity_wrapper",
                return_value=0.5,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.apply_diversity_penalty",
                return_value=(0.78, 0.68),
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.matthews_corrcoef",
                return_value=0.75,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.f1_score",
                return_value=0.80,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.precision_score",
                return_value=0.82,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.recall_score",
                return_value=0.78,
            ),
            patch(
                "ml_grid.pipeline.evaluate_methods_ga.metrics.accuracy_score",
                return_value=0.83,
            ),
            patch("ml_grid.pipeline.evaluate_methods_ga.pd.DataFrame.to_csv"),
        ):
            mock_get_y_pred.return_value = np.array([1, 1, 1, 1, 1])
            mock_auc.side_effect = ValueError("Only one class present in y_test")

            global_params_mock = MagicMock()
            global_params_mock.verbose = 0

            ml_grid = MagicMock()
            type(ml_grid).verbose = 0
            ml_grid.y_test = np.array([1, 1, 1, 1, 1])
            ml_grid.local_param_dict = {"div_p": 0}
            ml_grid.original_feature_names = ["feat1", "feat2"]
            ml_grid.logging_paths_obj.log_folder_path = "/tmp/test"
            ml_grid.global_params = global_params_mock
            type(ml_grid.global_params).log_store_dataframe_path = "test.csv"

            mock_model = (
                0.7,
                "TestModel",
                ["feat1"],
                None,
                0.85,
                np.array([1, 1, 1, 1, 1]),
            )
            individual = [[mock_model], []]

            result = evaluate_weighted_ensemble_auc(individual, ml_grid)

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert result[0] == 0.5
