import unittest

import numpy as np
import pandas as pd

from ml_grid.pipeline.data_train_test_split import get_data_split, is_valid_shape


class TestDataSplit(unittest.TestCase):

    def setUp(self):
        """Set up a sample imbalanced dataset."""
        X_data = np.random.rand(100, 5)
        # Create an imbalanced target variable (e.g., 80% class 0, 20% class 1)
        y_data = np.array([0] * 80 + [1] * 20)

        # Shuffle the data to mix classes
        p = np.random.permutation(len(y_data))

        self.X = pd.DataFrame(X_data[p])
        self.y = pd.Series(y_data[p])

    def test_split_no_resample(self):
        """Test standard stratified split without any resampling."""
        local_param_dict = {"resample": None}
        X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = get_data_split(
            self.X, self.y, local_param_dict
        )

        # Initial 75/25 split: 100 samples -> ~75 train + ~25 test
        assert len(X_test_orig) == 25, "Validation set should be 25% of total"
        assert len(y_test_orig) == 25

        # Second 75/25 split on the first 75%: 75 samples -> ~56 train + ~19 test
        # These values are deterministic based on sklearn's train_test_split random_state=1
        assert (
            54 <= len(X_train) <= 58
        ), f"Expected ~56 training samples, got {len(X_train)}"
        assert 54 <= len(y_train) <= 58
        assert 17 <= len(X_test) <= 21, f"Expected ~19 test samples, got {len(X_test)}"
        assert 17 <= len(y_test) <= 21

    def test_split_undersample(self):
        """Test split with undersampling."""
        local_param_dict = {"resample": "undersample"}
        X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = get_data_split(
            self.X, self.y, local_param_dict
        )

        # After undersampling, total samples should be approximately 2 * min_class_count
        # Since we have 80 class 0 and 20 class 1, undersampling gives ~40 samples total

        # The validation set is 25% of the undersampled data
        assert (
            8 <= len(X_test_orig) <= 12
        ), f"Expected ~10 validation samples, got {len(X_test_orig)}"

        # Check that classes are balanced after undersampling
        y_train_counts = y_train.value_counts()
        total_y_train = len(y_train)

        # Both classes should have similar counts (±2 for small variations)
        assert (
            abs(y_train_counts[0] - y_train_counts[1]) <= 4
        ), "After undersampling, both classes should have approximately equal counts"

        # The training set should be roughly balanced
        train_ratio_class_0 = y_train_counts[0] / total_y_train
        train_ratio_class_1 = y_train_counts[1] / total_y_train

        assert (
            0.4 <= train_ratio_class_0 <= 0.6
        ), "Class 0 should be ~50% after undersampling"
        assert (
            0.4 <= train_ratio_class_1 <= 0.6
        ), "Class 1 should be ~50% after undersampling"

        # The original validation set (test_orig) should NOT be undersampled
        y_orig_counts = y_test_orig.value_counts()
        assert (
            abs(y_orig_counts[0] - y_orig_counts[1]) > 2
        ), "Original validation set should retain class imbalance"

    def test_split_oversample(self):
        """Test split with oversampling, ensuring no data leakage."""
        local_param_dict = {"resample": "oversample"}
        X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = get_data_split(
            self.X, self.y, local_param_dict
        )

        # Original validation set should not be oversampled
        assert len(X_test_orig) == 25
        assert len(y_test_orig) == 25

        # The oversampling should have balanced the classes
        y_train_counts = y_train.value_counts()
        assert (
            abs(y_train_counts[0] - y_train_counts[1]) <= 15
        ), "Training set should be roughly balanced after oversampling"

        # Check that splits are reasonable sizes based on the data
        # With random oversampling, exact counts vary but proportions should be consistent
        # train_ratio is calculated but not used

        # Train set should be approximately 75% of (oversampled training data)
        # The original 75 samples become ~124 after oversampling, then 75% is ~93
        assert (
            80 <= len(X_train) <= 110
        ), f"X_train length {len(X_train)} should be in reasonable range after oversampling"
        assert (
            20 <= len(X_test) <= 40
        ), f"X_test length {len(X_test)} should be in reasonable range"

        # The validation set (test_orig) should NOT be oversampled
        # It's a 75/25 split of the original data: 100 * 0.25 = 25
        assert len(X_test_orig) == 25, "Validation set size should match expected"
        assert len(y_test_orig) == 25

        # The original validation set should retain class imbalance
        y_orig_counts = y_test_orig.value_counts()
        assert (
            abs(y_orig_counts[0] - y_orig_counts[1]) > 5
        ), "Original validation set should remain imbalanced (not oversampled)"

    def test_invalid_shape_disables_resample(self):
        """Test that resampling is disabled for invalid (1D) input shapes."""
        X_1d = np.random.rand(100)  # Invalid 1D shape
        y_1d = self.y
        local_param_dict = {"resample": "oversample"}  # Should be overridden

        X_train, _, y_train, _, _, _ = get_data_split(X_1d, y_1d, local_param_dict)
        # If resampling was disabled, the training set should be imbalanced
        self.assertNotEqual(y_train.value_counts()[0], y_train.value_counts()[1])

    def test_is_valid_shape(self):
        """Test the is_valid_shape helper function."""
        self.assertTrue(is_valid_shape(pd.DataFrame(np.random.rand(5, 2))))
        self.assertTrue(is_valid_shape(np.random.rand(5, 2)))
        self.assertFalse(is_valid_shape(np.random.rand(5)))  # 1D array
        self.assertFalse(is_valid_shape([1, 2, 3]))  # list


if __name__ == "__main__":
    unittest.main()
