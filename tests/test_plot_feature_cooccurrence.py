"""Test plot_feature_cooccurrence uncovered branches."""

import pandas as pd

from ml_grid.util import GA_results_explorer
from ml_grid.util.global_params import global_parameters


def test_plot_feature_cooccurrence_base_learner_features():
    """Test plot_feature_cooccurrence with base learner features.

    This test covers the plot_feature_cooccurrence method lines 1762-1905,
    specifically the _calculate_cooccurrence_matrix helper which calculates
    how often features appear together in top-performing runs.
    """
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": '["feature_a", "feature_b", "feature_c"]',
            "auc": [0.85, 0.78, 0.92],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=50.0,  # Top 50%
        top_n_features=3,
        feature_type="base_learner",
    )

    assert result is None


def test_plot_feature_cooccurrence_initial_features():
    """Test plot_feature_cooccurrence with initial features from f_list.

    Tests the path where feature_type='initial' and f_list contains
    binary mask strings that need to be decoded.
    """
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": '["feature_a", "feature_b", "feature_c"]',
            "auc": [0.85, 0.78, 0.92],
            "f_list": [
                "[1, 0, 1]",
                "[0, 1, 1]",
                "[1, 1, 0]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_feature_cooccurrence(
        performance_metric="auc",
        top_percent=50.0,
        top_n_features=3,
        feature_type="initial",
    )

    assert result is None
