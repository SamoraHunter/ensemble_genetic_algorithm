"""Test plot_base_learner_feature_importance edge cases.

This module provides tests for specific uncovered behaviors in GA_results_explorer.py:

1. test_plot_base_learner_feature_importance_all_features_always_present:
   Tests when ALL features are always present/absent, triggering early return due to
   insufficient unique values (nunique < 2). This covers lines 933-935.

2. Analysis of TypeError handler (lines 896-1000):
   This handler is designed to catch errors from:
   - pd.Series creation with generator expression at lines 867-873
   - .unique()/.nunique() calls that fail with unhashable types

   The TypeError handler can only be triggered if temp_df["all_bl_features"] contains
   non-iterable items (like None or integers). Since combine_bl_features_from_names
   always returns set() for any input, this requires external DataFrame corruption.
"""

import json

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_base_learner_feature_importance_all_features_always_present():
    """Test when all features appear in ALL runs (nunique() < 2 for all).

    This causes every feature to be skipped by the check at line 913,
    resulting in empty anova_results and early return at lines 934-935.

    Covers:
        - Lines 907-931: Loop through features, but all are skipped due to nunique < 2
        - Line 933-935: Empty results trigger early return without plotting
    """
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 0], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [1, 0, 0], 0, 0.8, None)]]",
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

    # Verify feature names are as expected
    assert explorer.df["feature_names"].iloc[0] == [["feature_a"]]
    assert explorer.df["feature_names"].iloc[1] == [["feature_a"]]

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    # Should return None - all features are always present, so no valid ANOVA
    assert result is None
