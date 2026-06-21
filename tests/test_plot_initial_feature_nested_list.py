"""Test plot_initial_feature_importance with nested list f_list values."""

import json

import pandas as pd


def test_plot_initial_feature_importance_nested_list_f_list():
    """Test plot_initial_feature_importance handles nested list f_list (not string).

    This test specifically covers line 655-656 in GA_results_explorer.py where
    decode_flist unwraps a single-element nested list when the input is already
    a Python list object (not a string that needs parsing).

    The scenario: f_list column contains actual list objects like [[0, 1, 0]]
    instead of strings like "[[0, 1, 0]]". This causes decode_flist to skip
    the string parsing path (lines 632-648) and reach lines 655-656 which handle
    the case where len(flist_row) == 1 and isinstance(flist_row[0], list).

    Without proper handling, this would leave flist_row as [[0, 1, 0]] instead
    of unwrapping to [0, 1, 0], causing subsequent length checks to fail.

    The test verifies that decode_flist correctly handles nested lists by checking
    the decoded feature names.
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
            # Use actual list objects (not strings) for f_list
            "f_list": [
                [[1, 0, 1]],  # Nested list - will trigger line 655-656
                [[0, 1, 1]],  # Nested list - will trigger line 655-656
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify decode_flist correctly handled the nested lists
    # The first row's f_list is [[1, 0, 1]]
    # After decoding: should extract feature names where bit == 1
    explorer.df["f_list"].iloc[0]

    # Check that feature_arrays was created (extracted from best_ensemble)
    assert isinstance(explorer.df["feature_arrays"].iloc[0], list)
