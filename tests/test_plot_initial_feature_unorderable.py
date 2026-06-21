"""Test plot_initial_feature_importance TypeError handler for sorting edge cases.

This test covers lines 734-738 (TypeError exception handler) which occurs when
the decoded features contain unorderable types that cause sorted() to fail.
"""

import json

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_initial_feature_importance_unorderable_features():
    """Test plot_initial_feature_importance TypeError handler when features can't be sorted.

    This test specifically covers lines 734-738 where a TypeError is raised during
    the try block at line 701. The TypeError occurs at line 727 when sorted() tries
    to sort a list containing unorderable types.

    Scenario: f_list contains string representations that decode to lists with varying
    structures, some containing lists which are unorderable in Python 3.
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
            # f_list contains strings that decode to lists with mixed types
            # Including a list inside which makes the result unorderable
            "f_list": [
                "[['feature_a', 'feature_b'], [1, 0]]",
                "[['feature_c'], [0, 1]]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # The decoded feature lists will contain mixed types after decode_flist:
    # Each f_list string gets parsed, and decode_flist may return lists with nested structures
    # that are unorderable.
    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None
