"""Test plot_base_learner_feature_importance with missing feature_names column."""

import json

import pandas as pd


def test_plot_base_learner_feature_importance_missing_feature_names():
    """Test plot_base_learner_feature_importance when feature_names column is missing.

    This test specifically covers lines 841-845 in plot_base_learner_feature_importance where
    the method checks if 'feature_names' column exists and returns early with an error if not.

    Scenario:
        - DataFrame lacks the 'feature_names' column that was added during initialization
        - This could happen with manually constructed DataFrames or corrupted state

    Covers:
        - Line 841: Check if feature_names_col not in self.df.columns
        - Line 842-845: Error logging and early return when feature_names is missing
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85, 0.78],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    # Remove the feature_names column that would normally exist after init
    del explorer.df["feature_names"]

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None
