"""Tests for param_space module."""

import unittest

import numpy as np

from ml_grid.util.param_space import ParamSpace


class TestParamSpace(unittest.TestCase):
    """Tests for the ParamSpace class."""

    def test_medium_size_initialization(self):
        """Test that medium-sized parameter space is initialized correctly."""
        ps = ParamSpace("medium")

        self.assertIsNotNone(ps.param_dict)
        self.assertIn("log_small", ps.param_dict)
        self.assertIsInstance(ps.param_dict["log_small"], np.ndarray)
        self.assertEqual(len(ps.param_dict["log_small"]), 3)

    def test_xsmall_size_initialization(self):
        """Test that xsmall-sized parameter space is initialized correctly."""
        ps = ParamSpace("xsmall")

        self.assertIsNotNone(ps.param_dict)
        self.assertIn("bool_param", ps.param_dict)
        self.assertEqual(len(ps.param_dict["bool_param"]), 2)

    def test_xwide_size_initialization(self):
        """Test that xwide-sized parameter space is initialized correctly."""
        ps = ParamSpace("xwide")

        self.assertIsNotNone(ps.param_dict)
        self.assertIn("log_large", ps.param_dict)

    def test_medium_and_xsmall_have_different_param_counts(self):
        """Test that different sizes produce different parameter counts."""
        ps_medium = ParamSpace("medium")
        ps_xsmall = ParamSpace("xsmall")

        # Medium should have larger arrays than xsmall
        self.assertGreater(
            len(ps_medium.param_dict["log_small"]),
            len(ps_xsmall.param_dict["log_small"]),
        )

    def test_bool_param_contains_true_and_false(self):
        """Test that bool_param contains both True and False."""
        ps = ParamSpace("medium")

        self.assertTrue(True in ps.param_dict["bool_param"])
        self.assertTrue(False in ps.param_dict["bool_param"])

    def test_invalid_size_sets_none(self):
        """Test that invalid size string sets param_dict to None."""
        ps = ParamSpace("invalid_size")

        self.assertIsNone(ps.param_dict)


if __name__ == "__main__":
    unittest.main()
