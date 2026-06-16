"""Tests for validate_param_methods module."""

import unittest

from ml_grid.util.validate_param_methods import (
    hidden_layer_size,
    validate_max_leaf_nodes,
    validate_min_samples_split,
    validate_subsample,
    validate_warm_start,
)


class TestValidateMaxLeafNodes(unittest.TestCase):
    """Tests for validate_max_leaf_nodes function."""

    def test_valid_integer_value(self):
        """Test that valid integer values are preserved."""
        param_space = {"max_leaf_nodes": 10}
        result = validate_max_leaf_nodes(param_space)
        self.assertEqual(result["max_leaf_nodes"], 10)

    def test_invalid_string_values_converted_to_default(self):
        """Test that non-integer values are converted to default value of 2."""
        param_space = {"max_leaf_nodes": "invalid"}
        result = validate_max_leaf_nodes(param_space)
        self.assertEqual(result["max_leaf_nodes"], 2)

    def test_value_too_small_converted_to_default(self):
        """Test that values less than 2 are converted to default value of 2."""
        param_space = {"max_leaf_nodes": 1}
        result = validate_max_leaf_nodes(param_space)
        self.assertEqual(result["max_leaf_nodes"], 2)

    def test_float_value_converted_to_default(self):
        """Test that float values are converted to default value of 2."""
        param_space = {"max_leaf_nodes": 3.5}
        result = validate_max_leaf_nodes(param_space)
        self.assertEqual(result["max_leaf_nodes"], 2)

    def test_missing_key_returns_original_dict(self):
        """Test that missing max_leaf_nodes key leaves dict unchanged."""
        param_space = {"other_param": "value"}
        result = validate_max_leaf_nodes(param_space)
        self.assertEqual(result, param_space)


class TestHiddenLayerSize(unittest.TestCase):
    """Tests for hidden_layer_size function."""

    def test_valid_integer_value(self):
        """Test that valid integer values are preserved."""
        param_space = {"hidden_layer_size": 10}
        result = hidden_layer_size(param_space)
        self.assertEqual(result["hidden_layer_size"], 10)

    def test_invalid_string_values_converted_to_default(self):
        """Test that non-integer values are converted to default value of 2."""
        param_space = {"hidden_layer_size": "invalid"}
        result = hidden_layer_size(param_space)
        self.assertEqual(result["hidden_layer_size"], 2)

    def test_value_too_small_converted_to_default(self):
        """Test that values less than 2 are converted to default value of 2."""
        param_space = {"hidden_layer_size": 1}
        result = hidden_layer_size(param_space)
        self.assertEqual(result["hidden_layer_size"], 2)

    def test_float_value_converted_to_default(self):
        """Test that float values are converted to default value of 2."""
        param_space = {"hidden_layer_size": 1.5}
        result = hidden_layer_size(param_space)
        self.assertEqual(result["hidden_layer_size"], 2)

    def test_missing_key_returns_original_dict(self):
        """Test that missing hidden_layer_size key leaves dict unchanged."""
        param_space = {"other_param": "value"}
        result = hidden_layer_size(param_space)
        self.assertEqual(result, param_space)


class TestValidateSubsample(unittest.TestCase):
    """Tests for validate_subsample function."""

    def test_valid_float_value(self):
        """Test that valid float values between 0 and 1 are preserved."""
        param_space = {"subsample": 0.5}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], 0.5)

    def test_list_of_valid_floats_preserved(self):
        """Test that list of valid floats is preserved."""
        param_space = {"subsample": [0.3, 0.6, 0.9]}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], [0.3, 0.6, 0.9])

    def test_invalid_string_converted_to_clamped_value(self):
        """Test that non-numeric values are converted to minimum of 0.01."""
        param_space = {"subsample": "invalid"}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], 0.01)

    def test_negative_float_converted_to_min(self):
        """Test that negative floats are clamped to minimum of 0.01."""
        param_space = {"subsample": -0.5}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], 0.01)

    def test_zero_float_converted_to_min(self):
        """Test that zero is converted to minimum of 0.01."""
        param_space = {"subsample": 0.0}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], 0.01)

    def test_one_converted_to_max(self):
        """Test that 1.0 is converted to max of 1.0."""
        param_space = {"subsample": 1.0}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], 1.0)

    def test_exactly_one_point_oh_converted_to_max(self):
        """Test that 1.0 exactly is clamped to max of 1.0."""
        param_space = {"subsample": 1.0}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], 1.0)

    def test_float_conversion_with_string_input_in_list(self):
        """Test that string in list gets converted to float then clamped."""
        param_space = {"subsample": ["0.5", "1.5"]}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"][0], 0.5)
        self.assertEqual(result["subsample"][1], 1.0)

    def test_list_with_valid_float_boundary(self):
        """Test list with boundary valid floats."""
        param_space = {"subsample": [0.01, 0.5, 0.99]}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"], [0.01, 0.5, 0.99])

    def test_list_with_invalid_values_clamped(self):
        """Test that list with invalid values gets clamped."""
        param_space = {"subsample": [-0.5, 1.5, "invalid"]}
        result = validate_subsample(param_space)
        self.assertEqual(result["subsample"][0], 0.01)
        self.assertEqual(result["subsample"][1], 1.0)
        self.assertEqual(result["subsample"][2], 0.01)

    def test_missing_key_returns_original_dict(self):
        """Test that missing subsample key leaves dict unchanged."""
        param_space = {"other_param": "value"}
        result = validate_subsample(param_space)
        self.assertEqual(result, param_space)


class TestValidateWarmStart(unittest.TestCase):
    """Tests for validate_warm_start function."""

    def test_valid_boolean_preserved(self):
        """Test that valid boolean values are preserved."""
        param_space = {"warm_start": True}
        result = validate_warm_start(param_space)
        self.assertEqual(result["warm_start"], True)

        param_space = {"warm_start": False}
        result = validate_warm_start(param_space)
        self.assertEqual(result["warm_start"], False)

    def test_numpy_boolean_preserved(self):
        """Test that numpy boolean values are preserved."""
        import numpy as np

        param_space = {"warm_start": np.bool_(True)}
        result = validate_warm_start(param_space)
        self.assertTrue(result["warm_start"])

    def test_numpy_false_preserved(self):
        """Test that numpy False boolean is preserved."""
        import numpy as np

        param_space = {"warm_start": np.bool_(False)}
        result = validate_warm_start(param_space)
        self.assertFalse(result["warm_start"])

    def test_string_value_converted_to_default(self):
        """Test that non-boolean string values are converted to True."""
        param_space = {"warm_start": "invalid"}
        result = validate_warm_start(param_space)
        self.assertEqual(result["warm_start"], True)

    def test_integer_value_converted_to_default(self):
        """Test that integer values are converted to True."""
        param_space = {"warm_start": 1}
        result = validate_warm_start(param_space)
        self.assertEqual(result["warm_start"], True)

    def test_missing_key_returns_original_dict(self):
        """Test that missing warm_start key leaves dict unchanged."""
        param_space = {"other_param": "value"}
        result = validate_warm_start(param_space)
        self.assertEqual(result, param_space)


class TestValidateMinSamplesSplit(unittest.TestCase):
    """Tests for validate_min_samples_split function."""

    def test_valid_integer_preserved(self):
        """Test that valid integer >= 2 is preserved."""
        param_space = {"min_samples_split": 2}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 2)

        param_space = {"min_samples_split": 10}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 10)

    def test_valid_float_preserved(self):
        """Test that valid float between 0 and 1 is preserved."""
        param_space = {"min_samples_split": 0.5}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 0.5)

    def test_integer_too_small_converted_to_default(self):
        """Test that integer < 2 is converted to default value of 2."""
        param_space = {"min_samples_split": 1}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 2)

    def test_float_zero_converted_to_default(self):
        """Test that float <= 0 is converted to default value of 2."""
        param_space = {"min_samples_split": 0.0}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 2)

    def test_float_one_converted_to_default(self):
        """Test that float >= 1 is converted to default value of 2."""
        param_space = {"min_samples_split": 1.0}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 2)

    def test_string_value_converted_to_default(self):
        """Test that non-numeric values are converted to default value of 2."""
        param_space = {"min_samples_split": "invalid"}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 2)

    def test_missing_key_returns_original_dict(self):
        """Test that missing min_samples_split key leaves dict unchanged."""
        param_space = {"other_param": "value"}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result, param_space)

    def test_float_boundary_one_minus_epsilon_converted_to_default(self):
        """Test that float very close to 1 but less is preserved."""
        param_space = {"min_samples_split": 0.999}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 0.999)

    def test_float_negative_converted_to_default(self):
        """Test that negative float is converted to default value of 2."""
        param_space = {"min_samples_split": -0.5}
        result = validate_min_samples_split(param_space)
        self.assertEqual(result["min_samples_split"], 2)


if __name__ == "__main__":
    unittest.main()
