"""Tests for ga_de_weight_method module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch


def test_get_weighted_ensemble_prediction_de_y_pred_valid_false():
    """Test get_weighted_ensemble_prediction_de_y_pred_valid with valid=False."""
    from ml_grid.ga_functions import ga_de_weight_method

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

    result = ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid(
        best, weights, ml_grid_object, valid=False
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 4


def test_get_weighted_ensemble_prediction_de_y_pred_valid_true_sklearn():
    """Test get_weighted_ensemble_prediction_de_y_pred_valid with valid=True and sklearn model."""
    from sklearn.linear_model import LogisticRegression

    from ml_grid.ga_functions import ga_de_weight_method

    # Mock ml_grid_object - X_train must be a DataFrame for proper column indexing
    ml_grid_object = Mock()
    ml_grid_object.verbose = 0  # Suppress logging during test
    ml_grid_object.y_test = np.array([0, 1])
    ml_grid_object.y_test_orig = np.array([0, 1])
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

    result = ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid(
        best, weights, ml_grid_object, valid=True
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 2


def test_get_weighted_ensemble_prediction_de_y_pred_valid_true_binary_classification():
    """Test get_weighted_ensemble_prediction_de_y_pred_valid with valid=True and BinaryClassification model."""
    from ml_grid.ga_functions import ga_de_weight_method
    from ml_grid.ga_functions.ga_ann_util import BinaryClassification

    # Mock ml_grid_object - X_train must be a DataFrame for proper column indexing
    ml_grid_object = Mock()
    ml_grid_object.verbose = 0
    ml_grid_object.y_test = np.array([0, 1])
    ml_grid_object.y_test_orig = np.array([0, 1])
    ml_grid_object.X_test_orig = pd.DataFrame(
        [[1, 2], [3, 4]], columns=["col_0", "col_1"]
    )
    ml_grid_object.y_train = np.array([0, 1])
    ml_grid_object.X_train = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])

    # Create a BinaryClassification model
    input_dim = 2
    model = BinaryClassification(input_dim, 64, 64, 0.1)

    feature_cols = ["col_0", "col_1"]

    # Train on training data first with proper sigmoid activation in loss calculation
    X_train_tensor = torch.FloatTensor(ml_grid_object.X_train[feature_cols].values)
    y_train_tensor = torch.FloatTensor(ml_grid_object.y_train.reshape(-1, 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    target_ensemble = [(Mock(), model, feature_cols, None, None, None)]

    best = [target_ensemble]

    weights = np.array([1.0])

    result = ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid(
        best, weights, ml_grid_object, valid=True
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 2


def test_get_weighted_ensemble_prediction_de_y_pred_valid_value_error():
    """Test get_weighted_ensemble_prediction_de_y_pred_valid with ValueError during model fit."""
    from sklearn.base import BaseEstimator, RegressorMixin

    from ml_grid.ga_functions import ga_de_weight_method

    # Create a model that always raises ValueError on fit
    class FailingModel(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None):
            raise ValueError("Test error")

        def predict(self, X):
            return np.array([0] * len(X))

    # Mock ml_grid_object
    ml_grid_object = Mock()
    ml_grid_object.verbose = 1  # Enable logging to see the error logs
    ml_grid_object.X_test_orig = pd.DataFrame(
        [[1, 2], [3, 4]], columns=["col_0", "col_1"]
    )
    ml_grid_object.X_train = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])
    ml_grid_object.y_train = np.array([0, 1])

    model_failing = FailingModel()
    feature_cols = ["col_0", "col_1"]

    target_ensemble = [(Mock(), model_failing, feature_cols, None, None, None)]

    best = [target_ensemble]

    weights = np.array([1.0])

    # Should handle ValueError and return prediction
    result = ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid(
        best, weights, ml_grid_object, valid=True
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 2


def test_get_weighted_ensemble_prediction_de_y_pred_valid_nan_handling():
    """Test get_weighted_ensemble_prediction_de_y_pred_valid handles NaN predictions."""
    from ml_grid.ga_functions import ga_de_weight_method
    from ml_grid.ga_functions.ga_ann_util import BinaryClassification

    # Mock ml_grid_object
    ml_grid_object = Mock()
    ml_grid_object.verbose = 0
    ml_grid_object.X_test_orig = pd.DataFrame(
        [[1, 2], [3, 4]], columns=["col_0", "col_1"]
    )
    ml_grid_object.X_train = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])
    ml_grid_object.y_train = np.array([0, 1])

    # Create a BinaryClassification model
    input_dim = 2
    model = BinaryClassification(input_dim, 64, 64, 0.1)

    feature_cols = ["col_0", "col_1"]

    target_ensemble = [(Mock(), model, feature_cols, None, None, None)]

    best = [target_ensemble]

    weights = np.array([1.0])

    result = ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid(
        best, weights, ml_grid_object, valid=True
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 2


def test_get_weighted_ensemble_prediction_de_y_pred_zero_weights():
    """Test get_weighted_ensemble_prediction_de handles zero weights (normalize edge case)."""
    import numpy as np

    from ml_grid.ga_functions import ga_ensemble_weight_finder_de

    # Test with all-zero weights (normalize will return original)
    weights = np.array([0.0, 0.0])
    prediction_matrix_raw = np.array([[0.3, 0.7], [0.4, 0.6]])
    y_test = np.array([0, 1])

    # Should not raise and should compute average (since normalized to equal weights)
    result = ga_ensemble_weight_finder_de.get_weighted_ensemble_prediction_de(
        weights, prediction_matrix_raw, y_test
    )

    assert isinstance(result, float)


def test_find_ensemble_weights_de_basic():
    """Test find_ensemble_weights_de with minimal valid input."""
    from unittest.mock import Mock

    import numpy as np

    from ml_grid.ga_functions import ga_ensemble_weight_finder_de

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
    result = ga_ensemble_weight_finder_de.find_ensemble_weights_de(best, ml_grid_object)

    assert isinstance(result, np.ndarray)
    assert len(result) == 2
