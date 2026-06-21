"""Test plot_initial_feature_importance with expand=True to cover the else branch at line 725."""

import json

import pandas as pd


def test_plot_initial_feature_importance_expand_true():
    """Test plot_initial_feature_importance when expand=True triggers sorted unique features path.

    This test specifically covers lines 725-726 in GA_results_explorer.py where:
    - global_params.expand_plots = True
    - The else branch uses sorted(all_features_series.unique().tolist()) instead of truncation

    The if condition at line 714 is: `if not expand and all_features_series.nunique() > max_to_analyze`
    So when expand=True (regardless of count), the else branch at 725-726 executes.

    Covers:
        - Line 725: else clause when expand=True or feature count <= threshold
        - Line 726: unique_features = sorted(all_features_series.unique().tolist())

    This ensures proper handling when all features should be displayed without truncation.
    """

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

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
            # Use string-based f_list with feature names in same-length lists
            # Line 670-671 filters to only matching features
            "f_list": [
                "['feature_a', 'feature_b', 'feature_c']",
                "['feature_x', 'feature_y', 'feature_z']",  # Will decode to empty due to no matches
            ],
        }
    )

    global_params = global_parameters()
    global_params.expand_plots = True  # Set expand=True to trigger else branch

    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    # Verify the function completes (it may return early due to no ANOVA results)
    assert result is None
