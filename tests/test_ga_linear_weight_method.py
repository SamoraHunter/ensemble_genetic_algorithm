"""Tests for ga_linear_weight_method module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd


def test_get_linear_weighted_ensemble_predictions_valid_false():
    """Test get_linear_weighted_ensemble_predictions with valid=False."""
    from ml_grid.ga_functions import ga_linear_weight_method

    # Mock ml_grid_object
    ml_grid_object = Mock()
    ml_grid_object.verbose = 1
    ml_grid_object.X_test_orig = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    ml_grid_object.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    ml_grid_object.y_train = np.array([0, 1, 0, 1])

    # Mock best with pre-computed predictions (5th element is prediction array)
    ensemble_size = 2
    target_ensemble = []
    for i in range(ensemble_size):
        model_mock = Mock()
        feature_cols = [0, 1]
        pred_array = np.array([0.3, 0.7, 0.2, 0.8])
        target_ensemble.append(
            (model_mock, model_mock, feature_cols, None, None, pred_array)
        )

    best = [target_ensemble]

    # Simple equal weights
    weights = np.array([1.0, 1.0])

    result = ga_linear_weight_method.get_linear_weighted_ensemble_predictions(
        best, weights, ml_grid_object, valid=False
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 4


def test_get_linear_weighted_ensemble_predictions_valid_true_sklearn():
    """Test get_linear_weighted_ensemble_predictions with valid=True and sklearn model."""
    from sklearn.linear_model import LogisticRegression

    from ml_grid.ga_functions import ga_linear_weight_method

    # Mock ml_grid_object - X_train must be a DataFrame for proper column indexing
    ml_grid_object = Mock()
    ml_grid_object.verbose = 0  # Suppress logging during test
    ml_grid_object.y_test = np.array([0, 1])
    ml_grid_object.X_test_orig = pd.DataFrame(
        [[1, 2], [3, 4]], columns=["col_0", "col_1"]
    )
    ml_grid_object.y_train = np.array([0, 1])
    ml_grid_object.X_train = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])

    # Create actual sklearn model
    model = LogisticRegression(random_state=42)
    feature_cols = ["col_0", "col_1"]

    # Fit model on training data first
    model.fit(ml_grid_object.X_train[feature_cols], ml_grid_object.y_train)

    ensemble_size = 2
    target_ensemble = [
        (Mock(), model, feature_cols, None, None, None) for _ in range(ensemble_size)
    ]

    best = [target_ensemble]

    weights = np.array([1.0, 1.0])

    result = ga_linear_weight_method.get_linear_weighted_ensemble_predictions(
        best, weights, ml_grid_object, valid=True
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 2


def test_find_linear_weights_basic():
    """Test find_linear_weights with minimal valid input."""
    from unittest.mock import Mock

    import numpy as np

    from ml_grid.ga_functions import ga_linear_weight_method

    # Create mock with required attributes
    ml_grid_object = Mock()
    ml_grid_object.verbose = 0  # Suppress logging
    ml_grid_object.y_test = np.array([0, 1, 0, 1])

    # Create ensemble with pre-computed predictions (5th element)
    model_mock = Mock()
    pred_array_1 = np.array([0.3, 0.7, 0.2, 0.8])
    pred_array_2 = np.array([0.4, 0.6, 0.3, 0.7])
    target_ensemble = [
        (model_mock, model_mock, None, None, None, pred_array_1),
        (model_mock, model_mock, None, None, None, pred_array_2),
    ]

    best = [target_ensemble]

    # This should run without error and return weights
    result = ga_linear_weight_method.find_linear_weights(best, ml_grid_object)

    assert isinstance(result, np.ndarray)
    assert len(result) == 2
    # Weights should be normalized (sum close to 1)
    np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=5)


def test_find_linear_weights_with_zero_prediction_variance():
    """Test find_linear_weights handles edge case where predictions have zero variance."""
    from ml_grid.ga_functions import ga_linear_weight_method

    # Create mock
    ml_grid_object = Mock()
    ml_grid_object.y_test = np.array([0, 1, 0, 1])

    # All models predict the same value (zero variance)
    pred_array = np.array([0.5, 0.5, 0.5, 0.5])
    target_ensemble = [(Mock(), Mock(), None, None, None, pred_array)]

    best = [target_ensemble]

    # Should handle zero variance and return equal weights
    result = ga_linear_weight_method.find_linear_weights(best, ml_grid_object)

    assert isinstance(result, np.ndarray)
    assert len(result) == 1


def test_find_linear_weights_returns_non_negative_weights():
    """Test find_linear_weights ensures non-negative weights."""
    from ml_grid.ga_functions import ga_linear_weight_method

    # Create mock with y_test that might produce negative coefficients
    ml_grid_object = Mock()
    ml_grid_object.y_test = np.array([1, 0, 1, 0])

    # Create predictions that might lead to negative coefficients
    pred_array_1 = np.array([0.9, 0.1, 0.8, 0.2])
    pred_array_2 = np.array([0.1, 0.9, 0.2, 0.8])
    target_ensemble = [
        (Mock(), Mock(), None, None, None, pred_array_1),
        (Mock(), Mock(), None, None, None, pred_array_2),
    ]

    best = [target_ensemble]

    result = ga_linear_weight_method.find_linear_weights(best, ml_grid_object)

    # All weights should be non-negative
    assert np.all(result >= 0)


def test_find_linear_weights_normalizes_weights():
    """Test find_linear_weights normalizes the weight vector."""
    from ml_grid.ga_functions import ga_linear_weight_method

    # Create mock
    ml_grid_object = Mock()
    ml_grid_object.y_test = np.array([0, 1, 0, 1])

    # Creates predictions with different scales
    pred_array_1 = np.array([0.3, 0.7, 0.2, 0.8])
    pred_array_2 = np.array([0.4, 0.6, 0.3, 0.7])
    pred_array_3 = np.array([0.5, 0.5, 0.6, 0.5])
    target_ensemble = [
        (Mock(), Mock(), None, None, None, pred_array_1),
        (Mock(), Mock(), None, None, None, pred_array_2),
        (Mock(), Mock(), None, None, None, pred_array_3),
    ]

    best = [target_ensemble]

    result = ga_linear_weight_method.find_linear_weights(best, ml_grid_object)

    # Weights should sum to 1 after normalization
    assert np.abs(np.sum(result) - 1.0) < 1e-6
