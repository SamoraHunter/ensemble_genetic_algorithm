"""Test for plot_initial_feature_importance TypeError handler with global_params edge case.

This test specifically covers lines 734-738 (TypeError exception handler) which is
triggered when max_features_to_plot in global_params is None, causing a TypeError
when multiplying by 5 at line 712.

The scenario requires:
1. f_list values that decode to non-empty feature lists (pass the len > 0 filter)
2. global_params with max_features_to_plot = None (causes TypeError on multiplication)
"""

import json

import pandas as pd

from ml_grid.util import GA_results_explorer


def test_plot_initial_feature_importance_type_error_from_none_max_features():
    """Test plot_initial_feature_importance TypeError handler when global_params.max_features_to_plot is None.

    This test specifically covers lines 734-738 where a TypeError is raised during
    the try block at line 701 due to max_features_to_plot being None.

    The TypeError occurs at line 712: `max_features_to_plot * 5` when max_features_to_plot is None.

    Scenario:
        - f_list contains valid binary mask like "[1, 0]" that decodes to feature names
        - This passes the len > 0 filter (valid_decoded_features_df is non-empty)
        - Code reaches try block at line 701
        - Line 712: getattr(global_params, "max_features_to_plot", 20) * 5
        - If max_features_to_plot = None, this raises TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
    """

    class MockGlobalParams:
        """Mock global_parameters with max_features_to_plot = None to trigger TypeError."""

        def __init__(self):
            self.max_features_to_plot = None
            self.expand_plots = False

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
            # f_list with valid binary masks that decode to feature names
            "f_list": [
                "[1, 0, 1]",
                "[0, 1, 1]",
            ],
        }
    )

    mock_params = MockGlobalParams()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=mock_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    # The function should return None due to TypeError being caught
    assert result is None
