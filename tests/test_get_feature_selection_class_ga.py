"""Tests for get_feature_selection_class_ga module."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd


class TestFeatureSelectionMethodsClass(unittest.TestCase):
    """Test cases for feature_selection_methods_class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock ml_grid_object
        self.ml_grid_object = Mock()
        self.ml_grid_object.X_train = pd.DataFrame(
            {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20),
                "feature3": np.random.rand(20),
                "feature4": np.random.rand(20),
                "feature5": np.random.rand(20),
            }
        )
        self.ml_grid_object.y_train = pd.Series([0, 1] * 10)
        self.ml_grid_object.X_test = pd.DataFrame(
            {
                "feature1": np.random.rand(10),
                "feature2": np.random.rand(10),
                "feature3": np.random.rand(10),
                "feature4": np.random.rand(10),
                "feature5": np.random.rand(10),
            }
        )

    def test_init_sets_feature_parameter_vector(self):
        """Test that initialization correctly sets feature_parameter_vector."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        # With 5 features, starting from 2, should have values [2, 3, 4, 5]
        self.assertGreaterEqual(len(selector.feature_parameter_vector), 1)
        self.assertIn(2, selector.feature_parameter_vector)

    def test_getNfeaturesANOVAF(self):
        """Test ANOVA F-test feature selection."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        # Select top 3 features
        result = selector.getNfeaturesANOVAF(3)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(feat, str) for feat in result))

    def test_getRandomForestFeatureColumns(self):
        """Test Random Forest feature selection."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        # Select top 3 features
        result = selector.getRandomForestFeatureColumns(
            self.ml_grid_object.X_train, self.ml_grid_object.y_train, 3
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(feat, str) for feat in result))

    def test_getXGBoostFeatureColumns(self):
        """Test XGBoost feature selection."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        # Select top 3 features
        result = selector.getXGBoostFeatureColumns(
            self.ml_grid_object.X_train, self.ml_grid_object.y_train, 3
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(feat, str) for feat in result))

    def test_getExtraTreesFeatureColumns(self):
        """Test Extra Trees feature selection."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        # Select top 3 features
        result = selector.getExtraTreesFeatureColumns(
            self.ml_grid_object.X_train, self.ml_grid_object.y_train, 3
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(feat, str) for feat in result))

    def test_get_featured_selected_training_data_anova(self):
        """Test get_featured_selected_training_data with anova method."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        X_train_fs, X_test_fs = selector.get_featured_selected_training_data(
            method="anova"
        )

        self.assertIsInstance(X_train_fs, pd.DataFrame)
        self.assertIsInstance(X_test_fs, pd.DataFrame)

    def test_get_featured_selected_training_data_randomforest(self):
        """Test get_featured_selected_training_data with randomforest method."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        X_train_fs, X_test_fs = selector.get_featured_selected_training_data(
            method="randomforest"
        )

        self.assertIsInstance(X_train_fs, pd.DataFrame)
        self.assertIsInstance(X_test_fs, pd.DataFrame)

    def test_get_featured_selected_training_data_xgb(self):
        """Test get_featured_selected_training_data with xgb method."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        X_train_fs, X_test_fs = selector.get_featured_selected_training_data(
            method="xgb"
        )

        self.assertIsInstance(X_train_fs, pd.DataFrame)
        self.assertIsInstance(X_test_fs, pd.DataFrame)

    def test_get_featured_selected_training_data_extratrees(self):
        """Test get_featured_selected_training_data with extratrees method."""
        from ml_grid.util.get_feature_selection_class_ga import (
            feature_selection_methods_class,
        )

        selector = feature_selection_methods_class(self.ml_grid_object)

        X_train_fs, X_test_fs = selector.get_featured_selected_training_data(
            method="extratrees"
        )

        self.assertIsInstance(X_train_fs, pd.DataFrame)
        self.assertIsInstance(X_test_fs, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
