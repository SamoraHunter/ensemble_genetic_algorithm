"""Tests for evaluate_methods_ga module."""

from unittest.mock import MagicMock, patch

import numpy as np

from ml_grid.pipeline.evaluate_methods_ga import (
    evaluate_weighted_ensemble_auc,
    normalize,
)


class TestNormalize:
    """Tests for the normalize function."""

    def test_normalize_basic(self):
        """Test basic normalization of weights."""
        weights = np.array([1.0, 2.0, 3.0])
        result = normalize(weights)

        # L1 norm = 1+2+3 = 6, so normalized values should be [1/6, 2/6, 3/6]
        expected = weights / 6.0
        np.testing.assert_allclose(result, expected)

    def test_normalize_already_normalized(self):
        """Test that already normalized weights stay the same."""
        weights = np.array([0.2, 0.3, 0.5])
        result = normalize(weights)

        # Already sums to 1 in L1 norm
        np.testing.assert_allclose(result, weights)

    def test_normalize_zero_vector(self):
        """Test normalization of zero vector (should return as-is)."""
        weights = np.array([0.0, 0.0, 0.0])
        result = normalize(weights)

        # Zero vector should be returned unchanged
        np.testing.assert_array_equal(result, weights)


class TestGetYpredResolver:
    """Tests for the get_y_pred_resolver function."""

    @patch("ml_grid.pipeline.evaluate_methods_ga.get_unweighted_ensemble_predictions")
    def test_get_y_pred_resolver_svc_fallback(self, mock_unweighted):
        """Test get_y_pred_resolver handles SVC fitting error with fallback."""
        from ml_grid.pipeline.evaluate_methods_ga import get_y_pred_resolver

        # Mock ml_grid_object
        ml_grid = MagicMock()
        type(ml_grid).verbose = 0  # Make verbose accessible as attribute
        ml_grid.local_param_dict = {"weighted": None}
        ml_grid.X_test_orig = np.array([[1, 2], [3, 4]])
        ml_grid.y_test = np.array([0, 1])

        # Make unweighted function raise a ValueError about SVC
        mock_unweighted.side_effect = ValueError(
            "The dual coefficients or intercepts are not finite"
        )

        result = get_y_pred_resolver([[]], ml_grid, valid=False)

        # Should return zeros with expected length
        assert len(result) == 2
        np.testing.assert_array_equal(result, np.zeros(2))

    @patch(
        "ml_grid.pipeline.evaluate_methods_ga.get_weighted_ensemble_prediction_de_y_pred_valid"
    )
    @patch("ml_grid.pipeline.evaluate_methods_ga.find_ensemble_weights_de")
    def test_get_y_pred_resolver_de_weighted(self, mock_find_weights, mock_get_pred):
        """Test get_y_pred_resolver handles DE (Differential Evolution) weighted ensemble."""
        from ml_grid.pipeline.evaluate_methods_ga import get_y_pred_resolver

        # Mock ml_grid_object
        ml_grid = MagicMock()
        type(ml_grid).verbose = 0
        ml_grid.local_param_dict = {"weighted": "de"}
        ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6]])
        ml_grid.y_test = np.array([0, 1, 0])

        # Mock the DE-specific functions
        mock_find_weights.return_value = np.array([0.3, 0.7])
        expected_pred = np.array([0.2, 0.8, 0.3])
        mock_get_pred.return_value = expected_pred

        result = get_y_pred_resolver(
            [[("model1", None, None, None, None, None)], []], ml_grid, valid=False
        )

        # Should call DE functions and return predictions
        mock_find_weights.assert_called_once()
        mock_get_pred.assert_called_once()
        np.testing.assert_array_equal(result, expected_pred)


class TestEvaluateWeightedEnsembleAuc:
    """Tests for the evaluate_weighted_ensemble_auc function."""

    @patch("ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver")
    @patch("ml_grid.pipeline.evaluate_methods_ga.metrics.roc_auc_score")
    @patch("ml_grid.pipeline.evaluate_methods_ga.metrics.matthews_corrcoef")
    @patch("ml_grid.pipeline.evaluate_methods_ga.metrics.f1_score")
    @patch("ml_grid.pipeline.evaluate_methods_ga.metrics.precision_score")
    @patch("ml_grid.pipeline.evaluate_methods_ga.metrics.recall_score")
    @patch("ml_grid.pipeline.evaluate_methods_ga.metrics.accuracy_score")
    @patch("ml_grid.pipeline.evaluate_methods_ga.measure_diversity_wrapper")
    @patch("ml_grid.pipeline.evaluate_methods_ga.apply_diversity_penalty")
    def test_evaluate_weighted_ensemble_auc_basic(
        self,
        mock_apply_diversity,
        mock_measure_diversity,
        mock_accuracy,
        mock_recall,
        mock_precision,
        mock_f1,
        mock_mcc,
        mock_auc,
        mock_get_y_pred,
    ):
        """Test basic functionality of evaluate_weighted_ensemble_auc."""
        # Mock get_y_pred_resolver to return predictions
        mock_get_y_pred.return_value = np.array([0, 1, 0, 1, 0])

        # Mock sklearn metrics
        mock_auc.return_value = 0.85
        mock_mcc.return_value = 0.75
        mock_f1.return_value = 0.80
        mock_precision.return_value = 0.82
        mock_recall.return_value = 0.78
        mock_accuracy.return_value = 0.83

        # Mock diversity functions
        mock_measure_diversity.return_value = 0.5
        mock_apply_diversity.return_value = (0.78, 0.68)

        with patch("ml_grid.pipeline.evaluate_methods_ga.pd.DataFrame.to_csv"):
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

    @patch("ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver")
    def test_evaluate_weighted_ensemble_auc_with_diversity_penalty(
        self,
        mock_get_y_pred,
    ):
        """Test evaluate_weighted_ensemble_auc with diversity penalty enabled."""
        # Mock get_y_pred_resolver to return predictions
        mock_get_y_pred.return_value = np.array([0, 1, 0, 1, 0])

        with (
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

            def apply_diversity_side_effect(auc, mcc, diversity_metric, params):
                if diversity_metric < 1.0:
                    return (auc * 0.9, mcc * 0.9)
                return (auc, mcc)

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

    @patch("ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver")
    def test_evaluate_weighted_ensemble_auc_auc_error_handling(
        self,
        mock_get_y_pred,
    ):
        """Test evaluate_weighted_ensemble_auc handles ValueError in AUC calculation."""
        # Mock get_y_pred_resolver to return predictions
        mock_get_y_pred.return_value = np.array([1, 1, 1, 1, 1])

        with (
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
