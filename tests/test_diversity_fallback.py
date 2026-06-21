"""Test for plot_ensemble_feature_diversity BL_ columns fallback branch."""

import json

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_ensemble_feature_diversity_bl_columns_fallback():
    """Test plot_ensemble_feature_diversity uses BL_ columns when feature_names is deleted.

    This test specifically covers the else branch (lines 1160-1227) in plot_ensemble_feature_diversity
    where the method falls back to using BL_ columns after manually deleting 'feature_names'.

    Covers:
        - Line 1160: else branch (when feature_names NOT in df.columns)
        - Lines 1161-1169: BL_ columns detection and logging
        - Lines 1171-1196: decode_bl_features function with string feature names
        - Lines 1208-1214: Jaccard similarity calculation for multiple base learners
        - Lines 1229-1237: Plotting section (regplot)
    """
    import pandas as pd

    # Create a DataFrame with BL_ columns containing string feature names
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
            # BL_ columns with list of feature names
            "BL_0": [
                ["feature_a", "feature_b"],
                ["feature_b", "feature_c"],
            ],
            "BL_1": [
                ["feature_c"],
                ["feature_a"],
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Manually delete feature_names to trigger the else branch at line 1160
    del explorer.df["feature_names"]

    # Verify the column was removed
    assert "feature_names" not in explorer.df.columns, "feature_names should be deleted"

    result = explorer.plot_ensemble_feature_diversity(outcome_variable="auc")

    # Should return None since plot is displayed but no return value expected
    assert result is None


if __name__ == "__main__":
    test_plot_ensemble_feature_diversity_bl_columns_fallback()
    print("Test passed!")
