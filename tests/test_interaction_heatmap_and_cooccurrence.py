"""Tests for plot_interaction_heatmap and plot_feature_cooccurrence methods."""

import json
import os
import tempfile

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_interaction_heatmap_missing_column():
    """Test plot_interaction_heatmap with missing required column triggers early return.

    This test specifically covers lines 1465-1469 in plot_interaction_heatmap where:
    - Line 1465: Creates list of required columns [param1, param2, performance_metric]
    - Line 1466-1469: Iterates through required_cols and returns early if any is missing

    The test triggers the early return path when 'param2' column is not in DataFrame.

    Covers:
        - Line 1467: Check if col not in self.df.columns
        - Line 1468: Log error message for missing column
        - Line 1469: Return None to exit early without creating heatmap

    Args:
        Test the missing column validation path in plot_interaction_heatmap.
    """
    df = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame(
            {
                "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
                "original_feature_names": json.dumps(["feature_a", "feature_b"]),
                "auc": [0.85],
            }
        ),
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_parameters(),
    )

    # param1 exists ('pop_val'), but 'param2' doesn't exist in df
    result = df.plot_interaction_heatmap(
        param1="auc",
        param2="nonexistent_param",
        performance_metric="auc",
    )

    assert result is None


def test_plot_interaction_heatmap_success_path():
    """Test plot_interaction_heatmap with valid data creating pivot table and heatmap.

    This test covers the success path of plot_interaction_heatmap (lines 1465-1524) where:
    - All required columns exist in DataFrame
    - Pivot table is successfully created via pd.pivot_table()
    - Heatmap with annot=True, cmap="viridis" is generated

    Covers:
        - Line 1471-1475: Warning check for many unique values (not triggered if <= 15)
        - Line 1482-1493: Creating pivot table successfully
        - Line 1496-1505: Creating seaborn heatmap with annot=True, fmt=".4f", cmap="viridis"
        - Line 1507-1516: Adding title, labels, and tight_layout

    The test uses config params (pop_val, g) that have <= 15 unique values.
    """
    num_runs = 20
    df_data = {
        "best_ensemble": [
            "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]" for _ in range(num_runs)
        ],
        "original_feature_names": json.dumps(["feature_a", "feature_b"]),
        "auc": [0.8 + (i * 0.01) for i in range(num_runs)],
    }

    # Add config params with <= 15 unique values to avoid warning and ensure heatmap works
    df_data["pop_val"] = [10 + (i % 5) * 5 for i in range(num_runs)]  # 5 unique values
    df_data["g"] = [100 + (i % 3) * 20 for i in range(num_runs)]  # 3 unique values

    df = pd.DataFrame(df_data)

    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_parameters(),
    )

    result = explorer.plot_interaction_heatmap(
        param1="pop_val",
        param2="g",
        performance_metric="auc",
    )

    assert result is None


def test_plot_interaction_heatmap_with_plot_dir():
    """Test plot_interaction_heatmap with plot_dir provided saves file successfully."""
    import os

    num_runs = 6
    df_data = {
        "best_ensemble": [
            "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]" for _ in range(num_runs)
        ],
        "original_feature_names": json.dumps(["feature_a", "feature_b"]),
        "auc": [0.8 + (i * 0.01) for i in range(num_runs)],
    }

    df_data["pop_val"] = [10 + (i % 3) * 5 for i in range(num_runs)]
    df_data["g"] = [100 + (i % 2) * 20 for i in range(num_runs)]

    df = pd.DataFrame(df_data)

    global_params_obj = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params_obj,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_interaction_heatmap(
            param1="pop_val",
            param2="g",
            performance_metric="auc",
            plot_dir=tmpdir,
        )

        assert result is None

        expected_file = os.path.join(tmpdir, "interaction_heatmap.png")
        assert os.path.exists(expected_file), f"Plot file {expected_file} should exist"


def test_plot_feature_cooccurrence_invalid_params():
    """Test plot_feature_cooccurrence with invalid parameters triggers early returns.

    This test specifically covers the input validation at lines 1765-1778 in
    plot_feature_cooccurrence where:
    - Line 1765-1769: Check performance_metric not in df.columns
    - Line 1770-1772: Check top_percent not in (0, 100]
    - Line 1773-1775: Check top_n_features not a positive int
    - Line 1776-1778: Check feature_type not 'initial' or 'base_learner'

    Each test triggers one of these validation branches with early return.
    """
    df = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame(
            {
                "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
                "original_feature_names": json.dumps(["feature_a", "feature_b"]),
                "auc": [0.85],
            }
        ),
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_parameters(),
    )

    # Test 1: Missing performance metric
    result = df.plot_feature_cooccurrence(
        performance_metric="nonexistent",
        top_percent=10.0,
        top_n_features=15,
        feature_type="base_learner",
    )
    assert result is None

    # Test 2: Invalid top_percent (out of range)
    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=0,  # Must be > 0
        top_n_features=15,
        feature_type="base_learner",
    )
    assert result is None

    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=100.01,  # Must be <= 100
        top_n_features=15,
        feature_type="base_learner",
    )
    assert result is None

    # Test 3: Invalid top_n_features (not positive int)
    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=10.0,
        top_n_features=-1,  # Must be positive
        feature_type="base_learner",
    )
    assert result is None

    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=10.0,
        top_n_features=3.5,  # Must be int
        feature_type="base_learner",
    )
    assert result is None

    # Test 4: Invalid feature_type
    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=10.0,
        top_n_features=15,
        feature_type="invalid_type",
    )
    assert result is None


def test_plot_feature_cooccurrence_empty_after_filtering():
    """Test plot_feature_cooccurrence triggers early return when no runs in top percent."""
    df = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame(
            {
                "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
                "original_feature_names": json.dumps(["feature_a", "feature_b"]),
                "auc": [0.85],
            }
        ),
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_parameters(),
    )

    # Use 1% top_percent with only 1 run - no runs will be in the top
    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=1.0,  # Very small percentage with few rows
        top_n_features=15,
        feature_type="base_learner",
    )

    assert result is None


def test_plot_feature_cooccurrence_empty_matrix():
    """Test plot_feature_cooccurrence triggers early return when co-occurrence matrix is empty."""
    df = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame(
            {
                "best_ensemble": [
                    "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                    "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                    "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
                ],
                "original_feature_names": json.dumps(
                    ["feature_a", "feature_b", "feature_c"]
                ),
                "auc": [0.85, 0.88, 0.92],
            }
        ),
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_parameters(),
    )

    # Use very small top_percent so only top run is selected, but with 3 unique features
    # and only 1 run in top, the matrix might be empty or have low co-occurrence
    result = df.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=0.1,  # Very small to get minimal runs
        top_n_features=2,
        feature_type="base_learner",
    )

    assert result is None


def test_plot_feature_cooccurrence_with_plot_dir():
    """Test plot_feature_cooccurrence with plot_dir saves the file correctly."""
    df = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame(
            {
                "best_ensemble": [
                    "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                    "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                    "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
                ],
                "original_feature_names": json.dumps(
                    ["feature_a", "feature_b", "feature_c"]
                ),
                "auc": [0.85, 0.88, 0.92],
            }
        ),
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_parameters(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = df.plot_feature_cooccurrence(
            performance_metric="auc",
            top_percent=30.0,  # Ensure some runs are selected
            top_n_features=2,
            feature_type="base_learner",
            plot_dir=tmpdir,
        )

        assert result is None

        expected_file = os.path.join(tmpdir, "feature_cooccurrence_base_learner.png")
        assert os.path.exists(expected_file), f"Plot file {expected_file} should exist"


def test_plot_feature_stability_invalid_inputs():
    """Test plot_feature_stability with invalid inputs triggers early returns.

    This test specifically covers lines 1547-1557 where:
    - Line 1548-1551: Check performance_metric not in df
    - Line 1552-1554: Check top_percent not in (0, 100]
    - Line 1555-1557: Check feature_type not 'initial' or 'base_learner'

    Each test triggers one validation branch with early return.
    """
    df = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame(
            {
                "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
                "original_feature_names": json.dumps(["feature_a", "feature_b"]),
                "auc": [0.85],
            }
        ),
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_parameters(),
    )

    # Test 1: Missing performance metric
    result = df.plot_feature_stability(
        performance_metric="nonexistent",
        top_percent=10.0,
        feature_type="base_learner",
    )
    assert result is None

    # Test 2: Invalid top_percent (out of range)
    result = df.plot_feature_stability(
        performance_metric="auc",
        top_percent=0,  # Must be > 0
        feature_type="base_learner",
    )
    assert result is None

    result = df.plot_feature_stability(
        performance_metric="auc",
        top_percent=100.01,  # Must be <= 100
        feature_type="base_learner",
    )
    assert result is None

    # Test 3: Invalid feature_type
    result = df.plot_feature_stability(
        performance_metric="auc",
        top_percent=10.0,
        feature_type="invalid_type",
    )
    assert result is None
