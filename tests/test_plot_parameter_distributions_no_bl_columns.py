"""Test plot_parameter_distributions with base_learner_features when no BL_ columns exist."""

import json

import pandas as pd


def test_plot_parameter_distributions_base_learner_features_no_bl_columns():
    """Test plot_parameter_distributions with base_learner_features param_type when DataFrame has no BL_ columns.

    This test specifically covers lines 1073-1075 in plot_parameter_distributions where
    the function checks if any BL_ columns exist and returns early with a warning if not.

    Scenario:
        - param_type="base_learner_features"
        - DataFrame contains no columns starting with "BL_"
        - Lines 1073-1075 handle this by logging a warning and returning

    Covers:
        - Line 1074: Logging warning when no 'BL_' columns are found
        - Line 1075: Early return when no BL_ columns exist
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
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify no BL_ columns exist
    bl_cols = [col for col in explorer.df.columns if col.startswith("BL_")]
    assert len(bl_cols) == 0

    result = explorer.plot_parameter_distributions(param_type="base_learner_features")

    assert result is None
