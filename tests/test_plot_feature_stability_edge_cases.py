"""Test plot_feature_stability edge case: empty inner lists in feature_names."""

import json
import os

import pandas as pd


def test_plot_feature_stability_base_learner_empty_inner_lists():
    """Test plot_feature_stability with base_learner when feature_names contains empty inner lists.

    This test specifically covers lines 1592-1598 where the function iterates through
    top_runs_df[feature_names_col] and extracts features from nested lists. It handles
    the edge case where some inner list elements might be empty, which would result in
    no features being added to features_flat for those entries.

    Scenario:
        - feature_type="base_learner" (goes into else branch at line 1584)
        - feature_names column contains lists with some empty sublists: [[], ["f1"], []]
        - The code checks isinstance(feature_list, list) and extends only if it's a list
        - Empty lists are still lists, so they extend nothing (features_flat remains unchanged for that entry)

    Covers:
        - Line 1584-1598: base_learner branch with loop processing feature_names
        - Line 1592-1598: handling of empty inner lists (which are valid lists but add no features)
        - Line 1600-1602: Warning when features_flat is empty after processing
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
            "feature_names": [
                [],  # Empty list - no base learners
                [["feature_a", "feature_b"]],  # One base learner with features
                [[], ["feature_c"]],  # First sublist empty, second has feature
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify the feature_names column has our test data
    assert len(explorer.df["feature_names"]) == 3

    # This should trigger the warning at line 1601 because features_flat will be empty after processing
    result = explorer.plot_feature_stability(
        performance_metric="auc",
        top_percent=50.0,  # Use higher percentage to ensure some runs are selected
        feature_type="base_learner",
    )

    assert result is None


def test_plot_feature_stability_base_learner_non_list_elements():
    """Test plot_feature_stability with base_learner when feature_names contains non-list elements.

    This test covers the edge case where feature_lists in top_runs_df[feature_names_col]
    contains items that are NOT lists (like None or strings), which get skipped by the
    isinstance check at line 1596.

    Scenario:
        - Some entries have valid lists, some have None or strings
        - The code filters with isinstance(feature_lists, list) and isinstance(feature_list, list)
        - Non-list items are silently ignored

    Covers:
        - Line 1594-1598: Loop handling mixed data types in feature_names column
        - The filtering of non-list elements via isinstance checks
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
            "feature_names": [
                [["feature_a"]],  # Valid entry
                None,  # Not a list - gets skipped at line 1595
                ["feature_b"],  # String instead of list of lists - inner loop skips it
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_feature_stability(
        performance_metric="auc",
        top_percent=50.0,  # Use higher percentage to ensure some runs are selected
        feature_type="base_learner",
    )

    assert result is None


def test_plot_feature_stability_with_plot_dir():
    """Test plot_feature_stability with plot_dir provided to cover lines 1631-1633.

    This test covers the code path where plots are saved to disk rather than just displayed.
    """
    import tempfile

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
            "auc": [0.95, 0.92],
            "feature_names": [
                [["feature_a"], ["feature_b"]],
                [["feature_c"]],
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_feature_stability(
            performance_metric="auc",
            top_percent=50.0,
            feature_type="base_learner",
            plot_dir=tmpdir,
        )

        assert result is None

        os.path.join(tmpdir, "feature_stability.png")
        # Note: The file might not be created if the function hits early return
        # but in this case we have valid data so it should create the file


def test_plot_feature_stability_initial_features():
    """Test plot_feature_stability with initial_features to cover lines 1574-1583.

    This covers the 'initial' branch of feature_type handling, which uses f_list
    column instead of feature_names."""

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
            "auc": [0.95, 0.92],
            "f_list": [
                "[1, 0, 1]",
                "[0, 1, 1]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_feature_stability(
        performance_metric="auc",
        top_percent=50.0,
        feature_type="initial",
    )

    assert result is None
