"""Tests for data_plot_split module."""

import unittest

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend for testing

from ml_grid.pipeline.data_plot_split import (
    create_bar_chart,
    plot_candidate_feature_category_lists,
    plot_dict_values,
    plot_pie_chart_with_counts,
)


class TestPlotPieChartWithCounts(unittest.TestCase):
    """Test cases for plot_pie_chart_with_counts function."""

    def test_basic_plot(self):
        """Test basic pie chart creation with data."""
        X_train = pd.DataFrame({"a": [1, 2, 3]})
        X_test = pd.DataFrame({"a": [4, 5]})
        X_test_orig = pd.DataFrame({"a": [6]})

        # Should not raise an exception
        plot_pie_chart_with_counts(X_train, X_test, X_test_orig)

    def test_empty_datasets(self):
        """Test behavior when all datasets are empty - should raise ValueError due to empty sizes."""
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        X_test_orig = pd.DataFrame()

        # Empty data causes division by zero - verify this is the expected behavior
        with self.assertRaises(ValueError):
            plot_pie_chart_with_counts(X_train, X_test, X_test_orig)


class TestPlotDictValues(unittest.TestCase):
    """Test cases for plot_dict_values function."""

    def test_basic_plot(self):
        """Test basic bar chart with mixed boolean values."""
        data_dict = {
            "feature_a": True,
            "feature_b": False,
            "feature_c": True,
        }

        # Should not raise an exception
        plot_dict_values(data_dict)

    def test_all_true(self):
        """Test with all True values."""
        data_dict = {"a": True, "b": True, "c": True}

        plot_dict_values(data_dict)

    def test_all_false(self):
        """Test with all False values."""
        data_dict = {"a": False, "b": False, "c": False}

        plot_dict_values(data_dict)


class TestCreateBarChart(unittest.TestCase):
    """Test cases for create_bar_chart function."""

    def test_basic_plot(self):
        """Test basic bar chart creation."""
        data_dict = {"category_a": 10, "category_b": 20, "category_c": 15}

        # Should not raise an exception
        create_bar_chart(data_dict)

    def test_with_custom_labels(self):
        """Test with custom title and axis labels."""
        data_dict = {"cat1": 5, "cat2": 15}
        create_bar_chart(
            data_dict,
            title="Custom Title",
            x_label="X Axis Label",
            y_label="Y Axis Label",
        )

    def test_empty_dict(self):
        """Test with empty dictionary."""
        data_dict = {}
        # Should handle empty dict (plot will be empty)
        create_bar_chart(data_dict)

    def test_single_category(self):
        """Test with single category."""
        data_dict = {"only_cat": 42}
        create_bar_chart(data_dict)


class TestPlotCandidateFeatureCategoryLists(unittest.TestCase):
    """Test cases for plot_candidate_feature_category_lists function."""

    def test_basic_plot(self):
        """Test basic wrapper function."""
        data = {
            "text_features": 5,
            "numerical_features": 10,
            "categorical_features": 3,
        }

        # Should not raise an exception
        plot_candidate_feature_category_lists(data)


if __name__ == "__main__":
    unittest.main()
