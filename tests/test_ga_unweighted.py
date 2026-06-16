"""Tests for ga_unweighted module."""

from unittest.mock import Mock

import numpy as np


def test_get_unweighted_ensemble_predictions_valid_false():
    """Test get_unweighted_ensemble_predictions with valid=False."""
    from ml_grid.ga_functions import ga_unweighted

    # Mock ml_grid_object
    ml_grid_object = Mock()
    ml_grid_object.verbose = 1
    ml_grid_object.X_test_orig = np.array([[1, 2], [3, 4], [5, 6]])
    ml_grid_object.X_train = np.array([[1, 2], [3, 4], [5, 6]])
    ml_grid_object.y_train = np.array([0, 1, 0])

    # Mock best with pre-computed predictions (5th element is prediction array)
    ensemble_size = 3
    target_ensemble = []
    for i in range(ensemble_size):
        model_mock = Mock()
        feature_cols = [0, 1]
        pred_array = np.array([0.0, 1.0, 0.0])  # Integers for mode calculation
        target_ensemble.append(
            (model_mock, model_mock, feature_cols, None, None, pred_array)
        )

    best = [target_ensemble]

    result = ga_unweighted.get_unweighted_ensemble_predictions(
        best, ml_grid_object, valid=False
    )

    assert isinstance(result, list)
    assert len(result) == 3


def test_get_unweighted_ensemble_predictions_valid_true_sklearn():
    """Test get_unweighted_ensemble_predictions with valid=True and sklearn model."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from ml_grid.ga_functions import ga_unweighted

    # Mock ml_grid_object
    ml_grid_object = Mock()
    ml_grid_object.verbose = 0
    ml_grid_object.y_test = np.array([0, 1])
    ml_grid_object.y_test_orig = np.array([0, 1])
    ml_grid_object.X_test_orig = pd.DataFrame(
        [[1, 2], [3, 4]], columns=["col_0", "col_1"]
    )
    ml_grid_object.y_train = np.array([0, 1])
    ml_grid_object.X_train = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])

    # Create a sklearn model
    model = LogisticRegression(random_state=42)
    feature_cols = ["col_0", "col_1"]
    model.fit(ml_grid_object.X_train[feature_cols], ml_grid_object.y_train)

    ensemble_size = 2
    target_ensemble = [
        (Mock(), model, feature_cols, None, None, None) for _ in range(ensemble_size)
    ]

    best = [target_ensemble]

    result = ga_unweighted.get_unweighted_ensemble_predictions(
        best, ml_grid_object, valid=True
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_get_unweighted_ensemble_predictions_valid_true_binary_classification():
    """Test get_unweighted_ensemble_predictions with valid=True and BinaryClassification model."""
    import pandas as pd
    import torch

    from ml_grid.ga_functions import ga_unweighted
    from ml_grid.ga_functions.ga_ann_util import BinaryClassification

    # Mock ml_grid_object
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

    # Train on training data
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

    result = ga_unweighted.get_unweighted_ensemble_predictions(
        best, ml_grid_object, valid=True
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_get_unweighted_ensemble_predictions_verbose_logging():
    """Test get_unweighted_ensemble_predictions with verbose >= 2 for debug logging."""

    from ml_grid.ga_functions import ga_unweighted

    # Mock ml_grid_object with verbose = 2
    ml_grid_object = Mock()
    ml_grid_object.verbose = 2
    ml_grid_object.X_test_orig = np.array([[1, 2], [3, 4]])
    ml_grid_object.X_train = np.array([[1, 2], [3, 4]])
    ml_grid_object.y_train = np.array([0, 1])

    # Mock best with pre-computed predictions
    ensemble_size = 2
    target_ensemble = []
    for i in range(ensemble_size):
        model_mock = Mock()
        feature_cols = [0, 1]
        pred_array = np.array([0.0, 1.0])
        target_ensemble.append(
            (model_mock, model_mock, feature_cols, None, None, pred_array)
        )

    best = [target_ensemble]

    result = ga_unweighted.get_unweighted_ensemble_predictions(
        best, ml_grid_object, valid=False
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_get_unweighted_ensemble_predictions_edge_case_single_model():
    """Test get_unweighted_ensemble_predictions with a single model."""
    from ml_grid.ga_functions import ga_unweighted

    # Mock ml_grid_object
    ml_grid_object = Mock()
    ml_grid_object.verbose = 1
    ml_grid_object.X_test_orig = np.array([[1, 2], [3, 4]])
    ml_grid_object.X_train = np.array([[1, 2], [3, 4]])
    ml_grid_object.y_train = np.array([0, 1])

    # Single model ensemble
    target_ensemble = [(Mock(), Mock(), [0, 1], None, None, np.array([0.0, 1.0]))]

    best = [target_ensemble]

    result = ga_unweighted.get_unweighted_ensemble_predictions(
        best, ml_grid_object, valid=False
    )

    assert isinstance(result, list)
    assert len(result) == 2
