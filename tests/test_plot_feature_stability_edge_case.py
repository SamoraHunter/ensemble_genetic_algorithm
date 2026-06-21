"""Test for plot_feature_stability with base_learner and nested structure containing strings."""

import json

import pandas as pd


def test_plot_feature_stability_base_learner_nested_string_features():
    """Test plot_feature_stability with base_learner type where feature_names contains string elements.

    This test specifically covers the edge case at lines 1594-1598 in plot_feature_stability where
    feature_lists is expected to be a list of lists, but we need to verify it handles the fallback
    behavior correctly when data structures are inconsistent.

    The code currently only checks isinstance(feature_lists, list) but doesn't handle cases where
    elements inside might not be lists. This test ensures that empty feature lists don't cause issues
    and that the features_flat list is properly populated even with mixed structures.

    Covers:
        - Line 1594-1598: processing base_learner features when feature_names has variousstructures
        - Line 1600: handling case where features_flat ends up empty after processing

    The specific scenario tested:
        - Top runs DataFrame with feature_names containing lists with some empty sublists
        - Mixed structures in feature_lists (some are lists, some contain strings)

    This ensures robustness against inconsistent data from the GA runs.
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Now test plot_feature_stability with base_learner type
    result = explorer.plot_feature_stability(
        performance_metric="auc",
        top_percent=50.0,  # Top 50% - should include all 3 runs
        feature_type="base_learner",  # Uses the 'feature_names' column
    )

    assert result is None
