"""Test plot_parameter_distributions with initial_features param_type when f_list contains empty lists."""

import json

import pandas as pd


def test_plot_parameter_distributions_initial_features_empty_lists():
    """Test plot_parameter_distributions handles initial_features when f_list has empty arrays.

    This test specifically covers lines 1064-1085 in the plot_parameter_distributions method:

    - Line 1065-1070: Try block processes f_list column with param_type="initial_features"
    - Line 1067: title = "Selection Frequency of Initial Features" (when try succeeds)
    - Line 1083-1085: Early return when features_flat is empty after explode

    Scenario:
        - DataFrame has f_list column where each row contains an empty list []
        - After explode().dropna(), features_flat will be an empty list
        - This triggers the warning and early return at lines 1084-1085

    The method should handle this edge case gracefully by:
        - Logging a warning about no features found
        - Returning None without crashing or attempting to plot
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    # Create DataFrame with f_list column containing empty lists for each row
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            # f_list contains empty arrays for each run
            "f_list": [
                [],
                [],
            ],  # Empty lists - won't produce any features after explode
            "auc": [0.85, 0.78],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify f_list column exists
    assert "f_list" in explorer.df.columns

    # Call plot_parameter_distributions with param_type="initial_features"
    result = explorer.plot_parameter_distributions(param_type="initial_features")

    # Should complete without error and return None due to empty features_flat
    assert result is None


def test_plot_parameter_distributions_base_learner_features_empty_lists():
    """Test plot_parameter_distributions handles base_learner_features when BL columns are empty.

    This test specifically covers the else branch in lines 1071-1085 of plot_parameter_distributions:

    - Line 1072: bl_cols = [col for col in self.df.columns if col.startswith("BL_")]
    - Lines 1077-1080: Accumulates all_bl_features from BL_* columns
    - Line 1083-1085: Early return when features_flat is empty

    Scenario:
        - DataFrame has BL_0, BL_1 columns where each row contains empty lists []
        - After explode().dropna(), features_flat will be an empty list
        - This triggers the warning and early return at lines 1084-1085

    The method should handle this edge case gracefully.
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    # Create DataFrame with BL columns containing empty lists for each row
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            # BL columns contain empty lists - won't produce any features after explode
            "BL_0": [[], []],
            "BL_1": [[], []],
            "auc": [0.85, 0.78],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify BL columns exist
    bl_cols = [col for col in explorer.df.columns if col.startswith("BL_")]
    assert len(bl_cols) == 2

    # Call plot_parameter_distributions with param_type="base_learner_features"
    result = explorer.plot_parameter_distributions(param_type="base_learner_features")

    # Should complete without error and return None due to empty features_flat
    assert result is None
