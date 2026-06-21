"""Test for plot_initial_feature_importance TypeError handling.

This test specifically covers lines 730-738 (the TypeError exception handler in
plot_initial_feature_importance) which is triggered when the f_list column contains
data that causes pd.Series() creation to fail with TypeError due to unhashable types.
"""

import json

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_initial_feature_importance_unhashable_features():
    """Test plot_initial_feature_importance TypeError handler when f_list contains lists that leak into feature values.

    This test specifically covers the TypeError exception at line 734 which occurs when:
    - The decoded features contain unhashable types (like nested lists)
    - pd.Series() fails to create a series from these unhashable values
    - This raises TypeError caught at line 734

    Scenario: f_list contains [[1, [0]]] where the inner list [0] somehow leaks through
    to become part of feature names as a nested structure that's unhashable.
    """

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
            # Use a malformed f_list that contains list elements - this should trigger TypeError
            # when pd.Series tries to create features from nested structures
            "f_list": [
                [[1, 0], [1]],
                [[0, 1], [1]],
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # This should trigger TypeError in the pd.Series creation at line 702
    # when valid_decoded_feature_lists contains nested lists that can't be flattened properly
    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None
