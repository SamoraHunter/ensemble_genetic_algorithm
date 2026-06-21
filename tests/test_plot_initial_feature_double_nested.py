"""Test decode_flist double-nested list branch at lines 651-652."""

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_initial_feature_importance_double_nested_list():
    """Test plot_initial_feature_importance with double-nested list f_list values (not string).

    This test specifically covers lines 651-652 in decode_flist where:
    - flist_row is already a list (not coming from string parsing)
    - len(flist_row) == 1
    - isinstance(flist_row[0], list)

    The code flattens [[0,1,0]] → [0,1,0] to handle this case.

    Covers the branch where f_list contains actual Python lists (not strings).
    """
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": '["feature_a", "feature_b", "feature_c"]',
            "auc": [0.85, 0.78],
            "f_list": [
                [[1, 0, 1]],  # Double-nested list
                [[0, 1, 1]],  # Double-nested list
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify decoding worked correctly for double-nested lists
    feature_arrays = explorer.df["feature_arrays"].iloc[0]
    assert isinstance(feature_arrays, list)
    assert feature_arrays == [[1, 0, 1]]

    feature_names = explorer.df["feature_names"].iloc[0]
    assert len(feature_names) == 1  # One base learner
    assert "feature_a" in feature_names[0]  # From [1, 0, 1]
    assert "feature_b" not in feature_names[0]
    assert "feature_c" in feature_names[0]

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_base_learner_double_nested_list():
    """Test plot_base_learner_feature_importance with nested list structure.

    Tests when 'feature_names' column contains double-nested lists that need
    to be processed at lines 847-859 for combining features from base learners.
    """
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": '["feature_a", "feature_b", "feature_c"]',
            "auc": [0.85, 0.78],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify feature_names column was created correctly
    feature_names = explorer.df["feature_names"].iloc[0]
    assert isinstance(feature_names, list)

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None
