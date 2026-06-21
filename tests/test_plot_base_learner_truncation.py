"""Test for plot_base_learner_feature_importance feature truncation edge case."""

import json

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_base_learner_feature_importance_with_max_features_truncation():
    """Test plot_base_learner_feature_importance when features exceed max_to_analyze and truncation is triggered (lines 876-881).

    This test specifically covers the feature reduction branch where:
    - expand_plots=False (default)
    - all_features_series.nunique() > max_to_analyze (60 * 5 = 300, because default max_features_to_plot=60)
    - The feature truncation logic at lines 876-881 is executed

    Covers:
        - Line 875: Condition check 'not expand and all_features_series.nunique() > max_to_analyze'
        - Lines 876-880: Truncation path using value_counts().head(max_to_analyze).index.tolist()
        - Line 884: Debug log message with reduced feature count

    The test creates an ensemble with more than 300 unique base learner features to trigger
    the truncation logic, ensuring that only the top max_to_analyze most frequent features
    are included in the ANOVA analysis.
    """

    num_runs = 50
    total_unique_features = (
        350  # More than max_to_analyze (60 * 5 = 300) to trigger truncation
    )

    all_feature_names = [f"feature_{i}" for i in range(total_unique_features)]

    df_data = {
        "best_ensemble": [],
        "original_feature_names": json.dumps(all_feature_names),
        "auc": [],
    }

    for i in range(num_runs):
        # Each run uses ~50 features, with some overlap across runs
        # This creates a scenario where we have >300 unique features total
        start_idx = (i * 10) % total_unique_features
        end_idx = min(start_idx + 50, total_unique_features)
        active_features = list(range(start_idx, end_idx))

        mask = [0] * total_unique_features
        for idx in active_features:
            mask[idx] = 1

        # Create feature arrays with varied composition to ensure frequency differences
        df_data["best_ensemble"].append(
            f"[[[0.5 + {i * 0.01}, 'Model{i}', {str(mask)}, 0, 0.9, None]]]"
        )
        df_data["auc"].append(0.7 + (i * 0.01))

    df = pd.DataFrame(df_data)

    global_params = global_parameters()
    # Explicitly ensure expand_plots is False to trigger truncation
    assert not getattr(global_params, "expand_plots", False)

    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=all_feature_names,
        global_params_obj=global_params,
    )

    # Verify we have more unique features than max_to_analyze
    all_features = set()
    for _, row in explorer.df.iterrows():
        feature_names_list = row["feature_names"]
        if isinstance(feature_names_list, list):
            for f in feature_names_list:
                if isinstance(f, list):
                    all_features.update(f)

    assert len(all_features) > 300, "Test setup error: need >300 unique features"

    # This should trigger the truncation path (lines 876-881)
    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None
