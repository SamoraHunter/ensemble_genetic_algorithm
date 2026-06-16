"""Tests for ga_ann_weight_methods module."""

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from ml_grid.ga_functions.ga_ann_util import BinaryClassification


class TestGAAnnWeightMethods(unittest.TestCase):
    """Test cases for ga_ann_weight_methods functions."""

    def test_get_y_pred_ann_torch_weighting_valid_false(self):
        """Test get_y_pred_ann_torch_weighting with valid=False (uses pre-computed predictions)."""
        from ml_grid.ga_functions.ga_ann_weight_methods import (
            get_y_pred_ann_torch_weighting,
        )

        # Mock ml_grid_object
        ml_grid_object = Mock()
        ml_grid_object.verbose = 1
        ml_grid_object.y_test = np.array([0, 1, 0, 1])
        ml_grid_object.y_test_orig = np.array([0, 1, 0, 1])
        ml_grid_object.X_test_orig = np.random.rand(4, 3)
        ml_grid_object.y_train = np.array([0, 1, 0, 1])
        ml_grid_object.X_train = np.random.rand(4, 3)

        # Mock best with pre-computed predictions (5th element is prediction array)
        ensemble_size = 3
        target_ensemble = []
        for i in range(ensemble_size):
            model_mock = Mock()
            feature_cols = [f"col_{i}"]
            pred_array = np.array([0.1, 0.9, 0.2, 0.8])
            target_ensemble.append(
                (model_mock, model_mock, feature_cols, None, None, pred_array)
            )

        best = [target_ensemble]

        result = get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=False)

        # Result should be a numpy array with same length as test data
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 4)

    def test_get_y_pred_ann_torch_weighting_valid_true(self):
        """Test get_y_pred_ann_torch_weighting with valid=True (fits models and predicts on validation set)."""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        from ml_grid.ga_functions.ga_ann_weight_methods import (
            get_y_pred_ann_torch_weighting,
        )

        # Mock ml_grid_object - X_train must be a DataFrame for proper column indexing
        ml_grid_object = Mock()
        ml_grid_object.verbose = 1
        ml_grid_object.y_test = np.array([0, 1, 0, 1])
        ml_grid_object.y_test_orig = np.array([0, 1, 0, 1])
        ml_grid_object.X_test_orig = pd.DataFrame(
            np.random.rand(4, 3), columns=["col_0", "col_1", "col_2"]
        )
        ml_grid_object.y_train = np.array([0, 1, 0, 1])
        ml_grid_object.X_train = pd.DataFrame(
            np.random.rand(4, 3), columns=["col_0", "col_1", "col_2"]
        )

        # Create mock models
        model1 = LogisticRegression()
        model2 = LogisticRegression()

        # Mock best with actual models (not BinaryClassification)
        feature_cols1 = ["col_0", "col_1"]
        feature_cols2 = ["col_1", "col_2"]

        # Fit models on training data first
        model1.fit(ml_grid_object.X_train[feature_cols1], ml_grid_object.y_train)
        model2.fit(ml_grid_object.X_train[feature_cols2], ml_grid_object.y_train)

        target_ensemble = [
            (model1, model1, feature_cols1),
            (model2, model2, feature_cols2),
        ]

        best = [target_ensemble]

        result = get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=True)

        # Result should be a numpy array with same length as test data
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 4)

    def test_get_y_pred_ann_torch_weighting_binary_classification(self):
        """Test get_y_pred_ann_torch_weighting with BinaryClassification models."""
        import pandas as pd

        from ml_grid.ga_functions.ga_ann_util import TestData
        from ml_grid.ga_functions.ga_ann_weight_methods import (
            get_y_pred_ann_torch_weighting,
        )

        # Mock ml_grid_object - X_train must be a DataFrame for proper column indexing
        ml_grid_object = Mock()
        ml_grid_object.verbose = 1
        ml_grid_object.y_test = np.array([0, 1, 0, 1])
        ml_grid_object.y_test_orig = np.array([0, 1, 0, 1])
        ml_grid_object.X_test_orig = pd.DataFrame(
            np.random.rand(4, 3), columns=["col_0", "col_1", "col_2"]
        )
        ml_grid_object.y_train = np.array([0, 1, 0, 1])
        ml_grid_object.X_train = pd.DataFrame(
            np.random.rand(4, 3), columns=["col_0", "col_1", "col_2"]
        )

        # Create BinaryClassification model
        input_shape = 2
        hidden_size = 32
        deep_layers = 2
        dropout_val = 0.5

        model = BinaryClassification(
            column_length=input_shape,
            deep_layers_1=deep_layers,
            hidden_layer_size=hidden_size,
            dropout_val=dropout_val,
        )

        # Fit the model on training data using DataFrame column selection
        X_train_tensor = torch.FloatTensor(
            ml_grid_object.X_train[["col_0", "col_1"]].values
        )
        y_train_tensor = torch.FloatTensor(ml_grid_object.y_train)

        train_data = TestData(X_train_tensor)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(train_data.X_data)
            loss = criterion(outputs, y_train_tensor.unsqueeze(1))
            loss.backward()
            optimizer.step()

        feature_cols = ["col_0", "col_1"]
        target_ensemble = [(model, model, feature_cols)]

        best = [target_ensemble]

        result = get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=True)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 4)

    def test_get_y_pred_ann_torch_weighting_nan_fallback(self):
        """Test get_y_pred_ann_torch_weighting handles NaN predictions with fallback."""
        from ml_grid.ga_functions.ga_ann_weight_methods import (
            get_y_pred_ann_torch_weighting,
        )

        # Mock ml_grid_object
        ml_grid_object = Mock()
        ml_grid_object.verbose = 15  # Set high verbosity to trigger logs
        ml_grid_object.y_test = np.array([0, 1, 0, 1])
        ml_grid_object.y_test_orig = np.array([0, 1, 0, 1])
        ml_grid_object.X_test_orig = np.random.rand(4, 3)
        ml_grid_object.y_train = np.array([0, 1, 0, 1])
        ml_grid_object.X_train = np.random.rand(4, 3)

        # Mock best with pre-computed predictions that will cause ANN to produce NaN
        ensemble_size = 2
        target_ensemble = []
        for i in range(ensemble_size):
            model_mock = Mock()
            feature_cols = [f"col_{i}"]
            pred_array = np.array([0.1, 0.9, 0.2, 0.8])
            target_ensemble.append(
                (model_mock, model_mock, feature_cols, None, None, pred_array)
            )

        best = [target_ensemble]

        # This should trigger the NaN fallback code path
        result = get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=False)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
