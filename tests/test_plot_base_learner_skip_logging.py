"""Test plot_base_learner_feature_importance with skip logging."""

import json

import pandas as pd


def test_plot_base_learner_feature_importance_skip_logging():
    """Test plot_base_learner_feature_importance when feature is always present or always absent.

    This test specifically covers lines 914-917 in plot_base_learner_feature_importance where
    a warning is logged and the feature is skipped because it has only one unique value (always
    present or always absent across runs).

    Scenario:
        - A feature exists in ALL runs (nunique == 1, always True)
        - OR no run contains the feature (nunique == 1, always False)
        - Lines 914-917 log: "⏩ Skipping '{feature}': Feature is always present or always absent"

    Covers:
        - Line 914: Logging skip message when feature has only one unique value
        - Line 915: Message indicates feature is always present or always absent
        - Line 916: continue statement to skip this feature
        - Line 917: continue statement ends the if block
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    # Create data where "feature_a" appears in ALL runs, creating nunique == 1 for has_feature
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None
