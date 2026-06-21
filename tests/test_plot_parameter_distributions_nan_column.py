"""Test plot_parameter_distributions with NaN-only columns."""

import json

import pandas as pd


def test_plot_parameter_distributions_with_all_nan_column():
    """Test plot_parameter_distributions when a parameter column contains only NaN values.

    This test specifically covers lines 1032-1034 in plot_parameter_distributions where
    data.empty is True after dropna() on a column that's all NaNs.

    Scenario:
        - config or run_details subtest with a column containing only None/NaN values
        - This triggers the path: data = self.df[param].dropna(); if data.empty:
        - Lines 1032-1034 handle this by displaying "No Data" text in the subplot

    Covers:
        - Line 1032-1034: Handling empty DataFrame after dropna for a parameter
        - The fallback behavior that displays "No Data" text
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    # Add a config param column with all NaN values
    # 'weighted' is in config_params but we want to test the edge case where it's all NaN
    explorer.df["weighted"] = [None]  # This will be converted to NaN

    result = explorer.plot_parameter_distributions(param_type="config")

    assert result is None
