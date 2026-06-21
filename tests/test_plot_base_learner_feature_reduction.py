"""Test plot_base_learner_feature_importance with feature reduction triggered (>max_features_to_plot)."""

import json

import pandas as pd


def test_plot_base_learner_feature_importance_feature_reduction_with_max_features():
    """Test plot_base_learner_feature_importance with feature reduction triggered when >300 features.

    This test specifically covers lines 881-886 where the truncation branch is taken:
        - max_to_analyze = 60 * 5 = 300
        - expand_plots = False (default)
        - When all_features_series.nunique() > 300, use value_counts().head(max_to_analyze)

    Covers:
        - Line 877-880: Check if truncation should occur (expand=False and features > max_to_analyze)
        - Line 881-885: Truncation via value_counts().head(max_to_analyze).index.tolist()
        - Line 892: Else branch (if we had fewer features) is not taken here

    The key difference from test_plot_initial_feature_importance_feature_reduction_with_max_features:
        - This uses 'feature_names' column (already decoded list of lists)
        - That test uses 'f_list' column (binary masks that need decoding)
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    num_runs = 15
    # Create >300 unique features to trigger truncation
    all_feature_names = [f"feature_{i}" for i in range(350)]

    df_data = {
        "best_ensemble": [],
        "original_feature_names": json.dumps(all_feature_names),
        "auc": [],
        "feature_names": [],  # Using feature_names instead of f_list
    }

    for i in range(num_runs):
        start_idx = (i * 20) % 350
        end_idx = min(start_idx + 25, 350)

        df_data["best_ensemble"].append(
            f"[[[0.5, 'Model{i}', [1] * 350, 0, 0.9, None]]]"
        )
        df_data["auc"].append(0.8 + (i * 0.01))

        # feature_names contains list of lists for each base learner's features
        # Each run has different feature subsets to maximize unique combinations
        active_features = [f"feature_{j}" for j in range(start_idx, end_idx)]
        df_data["feature_names"].append([active_features])

    df = pd.DataFrame(df_data)

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=all_feature_names,
        global_params_obj=global_params,
    )

    # Verify we have >300 unique features (need to count from our test data)
    temp_df = explorer.df[["auc", "feature_names"]].copy()

    def combine_bl_features_from_names(row):
        feature_lists = row["feature_names"]
        if not isinstance(feature_lists, list):
            return set()
        all_features = set()
        for feature_list in feature_lists:
            if isinstance(feature_list, list):
                all_features.update(feature_list)
        return all_features

    temp_df["all_bl_features"] = temp_df.apply(combine_bl_features_from_names, axis=1)
    all_features_series = pd.Series(
        [
            feature
            for feature_set in temp_df["all_bl_features"]
            for feature in feature_set
        ]
    )

    unique_count = all_features_series.nunique()
    assert (
        unique_count > 300
    ), f"Test setup error: need >300 unique features but got {unique_count}"

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None
