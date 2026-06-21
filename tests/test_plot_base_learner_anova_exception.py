"""Test plot_base_learner_feature_importance exception handling with NaN values."""

import json

import pandas as pd


def test_plot_base_learner_feature_importance_anova_exception_handling():
    """Test plot_base_learner_feature_importance handles ANOVA exceptions gracefully when NaN values are present.

    This test specifically covers lines 926-927: the exception handler that logs warnings
    when ANOVA computation fails for a feature, even though the feature has sufficient
    unique values to pass the initial check at line 909.

    The scenario uses NaN values in 'auc' (the outcome_variable) which causes
    sm.stats.anova_lm() to fail, triggering the exception handler while
    other valid features still execute successfully.
    """
    import numpy as np

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
                "[[(0.8, 'Model4', [1, 1, 1], 0, 0.92, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            # Introduce NaN values in auc which will cause ANOVA to fail
            "auc": [0.85, np.nan, 0.92, 0.88],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    # Should return None - exception is handled gracefully
    assert result is None


def test_plot_base_learner_feature_importance_typeerror_handler():
    """Test plot_base_learner_feature_importance handles TypeError when feature_names contains non-iterable values.

    This test specifically covers lines 892-896: the TypeError exception handler that logs errors
    when pd.Series creation fails due to invalid data types in all_bl_features.

    The scenario uses None values in the feature_names column which cause the list comprehension
    `for feature_set in temp_df['all_bl_features']` to fail with 'TypeError: 'NoneType' object is not iterable'.
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

    # Set feature_names to contain None values instead of proper lists
    # This will cause TypeError when pd.Series tries to iterate over None
    explorer.df["feature_names"] = [None]

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    # Should return None - TypeError is handled gracefully
    assert result is None
