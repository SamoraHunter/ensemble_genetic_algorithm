"""Tests for plot_auc_ga module."""

import os
import tempfile

import numpy as np
import pytest

from ml_grid.pipeline.plot_methods.plot_auc_ga import plot_auc


class TestPlotAuc:
    """Tests for the plot_auc function."""

    def test_plot_auc_basic(self):
        """Test basic functionality with valid binary labels."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        result = plot_auc(y_true, y_pred, title="Test Plot", plot_dir=None)

        assert result is None

    def test_plot_auc_with_plot_dir(self):
        """Test functionality with a valid plot directory."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "test_plot.png")
            result = plot_auc(y_true, y_pred, title="Test Plot", plot_dir=plot_path)

            assert result is None
            assert os.path.exists(plot_path)

    @pytest.mark.filterwarnings(
        "ignore::sklearn.metrics._ranking.UndefinedMetricWarning"
    )
    def test_plot_auc_single_sample(self):
        """Test edge case with single sample."""
        y_true = np.array([1])
        y_pred = np.array([0.5])

        result = plot_auc(y_true, y_pred, title="Single Sample", plot_dir=None)

        assert result is None

    def test_plot_auc_identical_predictions(self):
        """Test edge case with identical predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])

        result = plot_auc(y_true, y_pred, title="Identical Predictions", plot_dir=None)

        assert result is None

    def test_plot_auc_perfect_predictions(self):
        """Test edge case with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])

        result = plot_auc(y_true, y_pred, title="Perfect Predictions", plot_dir=None)

        assert result is None

    @pytest.mark.filterwarnings(
        "ignore::sklearn.metrics._ranking.UndefinedMetricWarning"
    )
    def test_plot_auc_all_negative(self):
        """Test edge case with all negative samples."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.2, 0.4, 0.6, 0.8])

        result = plot_auc(y_true, y_pred, title="All Negative", plot_dir=None)

        assert result is None

    @pytest.mark.filterwarnings(
        "ignore::sklearn.metrics._ranking.UndefinedMetricWarning"
    )
    def test_plot_auc_all_positive(self):
        """Test edge case with all positive samples."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.2, 0.4, 0.6, 0.8])

        result = plot_auc(y_true, y_pred, title="All Positive", plot_dir=None)

        assert result is None
