"""Test plot_performance_tradeoff method."""

import json

import pandas as pd


def test_plot_performance_tradeoff_success():
    """Test plot_performance_tradeoff with valid data covering the success path.

    This test covers lines 1276-1337 where:
    - All required columns exist (performance_metric, cost_metric, hue_parameter)
    - Data is not empty after dropna
    - Plotting executes successfully

    Covers key branches:
        - Line 1278-1281: Column validation in the loop
        - Line 1283-1285: Logging success message
        - Line 1288-1291: Data preparation and dropna (non-empty)
        - Line 1294: Type conversion to category
        - Lines 1297-1337: Plot creation, labeling, legend, layout adjustment,
                          and plot_dir handling
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
                "[[(0.8, 'Model4', [1, 1, 1], 0, 0.92, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92, 0.88],
            "run_time": [120, 180, 90, 240],
            "pop_val": [10, 20, 30, 40],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_performance_tradeoff(
        performance_metric="auc",
        cost_metric="run_time",
        hue_parameter="pop_val",
    )

    # Should return None after plotting
    assert result is None


def test_plot_performance_tradeoff_missing_column():
    """Test plot_performance_tradeoff when a required column is missing.

    This covers lines 1278-1281 where the validation loop checks for required columns.
    When a required column (e.g., cost_metric='run_time') is missing from the DataFrame,
    it should log an error and return early without crashing.

    Covers:
        - Line 1279: col not in self.df.columns check
        - Line 1280: logger.error call
        - Line 1281: early return None
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            # Note:缺少 run_time column, which is the default cost_metric
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_performance_tradeoff()

    # Should return None due to missing column
    assert result is None


def test_plot_performance_tradeoff_empty_after_dropna():
    """Test plot_performance_tradeoff when all data gets dropped due to NaN values.

    This covers lines 1289-1291 where if plot_df.empty after dropna(), the method
    returns early with a warning without attempting to plot.

    Covers:
        - Line 1288: plot_df = self.df[required_cols].dropna()
        - Line 1289: if plot_df.empty check
        - Line 1290-1291: Warning log and early return
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            # All NaN values in required columns
            "auc": [float("nan")],
            "run_time": [float("nan")],
            "pop_val": [float("nan")],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_performance_tradeoff()

    # Should return None due to empty plot_df after dropna
    assert result is None


def test_plot_performance_tradeoff_with_plot_dir():
    """Test plot_performance_tradeoff with plot_dir provided to cover save path.

    This covers lines 1331-1334 where if plot_dir is not None, the plot is saved
    to a file and logged.

    Covers:
        - Line 1331: if plot_dir is not None check
        - Line 1332-1333: plot_path construction and plt.savefig()
        - Line 1334: Success log message
    """
    import os
    import tempfile

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
            "run_time": [120, 180, 90],
            "pop_val": [10, 20, 30],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_performance_tradeoff(
            performance_metric="auc",
            cost_metric="run_time",
            hue_parameter="pop_val",
            plot_dir=tmpdir,
        )

        assert result is None

        expected_file = os.path.join(tmpdir, "performance_tradeoff.png")
        assert os.path.exists(expected_file), f"Plot file {expected_file} not created"
