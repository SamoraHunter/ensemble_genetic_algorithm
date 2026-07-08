"""Tests for main.py including initialize_logger, main() function, and argument parsing."""

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_initialize_logger_creates_experiment_directory_with_timestamp():
    """Test that initialize_logger creates the experiment directory with timestamp."""
    import tempfile

    from main import initialize_logger

    tmp_dir = tempfile.mkdtemp()
    config_path = os.path.join(tmp_dir, "config.yml")

    with open(config_path, "w") as f:
        f.write("global_params:\n  testing: True\ngrid_params:\n  outcome_var_n: [1]\n")

    # Initialize logger
    logger = initialize_logger(config_path)

    # Verify logger is configured correctly
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ensemble_ga"

    try:
        # Verify experiment directory was created with timestamp pattern
        base_log_dir = logger.handlers[0].baseFilename

        # Extract the directory path from the log file path
        run_specific_dir = os.path.dirname(base_log_dir)

        # Directory should be in experiments folder
        assert "HFE_GA_experiments" in run_specific_dir

        # Directory name should contain timestamp pattern (YYYY-MM-DD_HH-MM-SS)
        dir_name = os.path.basename(run_specific_dir)
        import re

        timestamp_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
        assert re.match(
            timestamp_pattern, dir_name
        ), f"Expected timestamp pattern in {dir_name}"
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_main_evaluate_flag_functionality(caplog):
    """Test the evaluate flag functionality in main() - tests the evaluate code path."""
    from main import main

    tmp_dir = tempfile.mkdtemp()
    try:
        config_path = os.path.join(tmp_dir, "test_config.yml")

        # Write minimal config
        config_content = """
global_params:
  input_csv_path: "test_data.csv"
  n_iter: 1
  testing: True
grid_params:
  outcome_var_n: [1]
"""
        with open(config_path, "w") as f:
            f.write(config_content)

        # Create a mock results CSV file with the data structure needed for evaluate
        results_csv = os.path.join(tmp_dir, "final_grid_score_log.csv")
        df = pd.DataFrame(
            {
                "auc": [0.92],  # Pick highest AUC
                "best_ensemble": ['[(1.0, "LogisticRegression", [1, 0, 1])]'],
                "X_train_size": [100],
                "X_test_orig_size": [50],
                "X_test_size": [50],
            }
        )
        df.to_csv(results_csv, index=False)

        # Create initial test data file (needed for evaluating on validation set)
        test_data = os.path.join(tmp_dir, "test_data.csv")
        pd.DataFrame({"age": [25.0], "male": [1.0], "outcome_var_1": [1.0]}).to_csv(
            test_data, index=False
        )

        # Set up proper return values for the evaluation row
        df_eval = pd.DataFrame(
            {
                "auc": [0.92],
                "best_ensemble": ['[(1.0, "LogisticRegression", [1, 0, 1])]'],
            }
        )

        def get_best_row():
            return df_eval.loc[df_eval["auc"].idxmax()]

        # Mock the data.pipe class with a proper ml_grid_object that has all needed attributes
        def mock_data_pipe(*args, **kwargs):
            mock_obj = MagicMock()
            mock_obj.X_train = MagicMock()
            mock_obj.y_train = MagicMock()
            mock_obj.X_test = MagicMock()
            mock_obj.y_test = MagicMock()
            mock_obj.X_test_orig = pd.DataFrame({"age": [30.0], "male": [1.0]})
            mock_obj.y_test_orig = pd.Series([1.0])
            mock_obj.original_feature_names = ["age", "male", "outcome_var_1"]
            mock_obj.local_param_dict = {"weighted": "unweighted"}
            mock_obj.verbose = 0
            mock_obj.base_project_dir = tmp_dir
            return mock_obj

        # Mock all data.pipe instances with our mock that has proper attributes
        with patch("main.data.pipe", side_effect=mock_data_pipe):

            # Mock main_ga.run and its execute method
            def mock_main_ga_run(*args, **kwargs):
                mock_obj = MagicMock()
                mock_obj.execute.return_value = None
                return mock_obj

            with patch("ml_grid.pipeline.main_ga.run", side_effect=mock_main_ga_run):

                # Mock global_parameters to control the experiment
                def mock_global_parameters(config_path=None, **kwargs):
                    gp_mock = MagicMock()
                    gp_mock.n_iter = 1
                    gp_mock.testing = True
                    gp_mock.input_csv_path = test_data
                    gp_mock.base_project_dir = tmp_dir
                    gp_mock.verbose = 0
                    gp_mock.model_list = []
                    return gp_mock

                with patch(
                    "main.global_parameters", side_effect=mock_global_parameters
                ):

                    # Patch read_csv for the CSV reading in evaluate section and plot section
                    def read_csv_side_effect(filepath, *args, **kwargs):
                        if filepath == results_csv:
                            return df_eval
                        elif "test_data.csv" in str(filepath):
                            return pd.DataFrame(
                                {"age": [25.0], "male": [1.0], "outcome_var_1": [1.0]}
                            )
                        # Make the first row accessible for idxmax()
                        result = df_eval.copy()
                        result.index = pd.RangeIndex(len(result))
                        return result

                    with patch("pandas.read_csv", side_effect=read_csv_side_effect):

                        # Execute main with evaluate=True
                        main(config_path=config_path, evaluate=True)

                        # Verify logger.info was called for evaluation start using caplog
                        assert any(
                            "Starting evaluation" in record.message
                            for record in caplog.records
                        ), f"Expected 'Starting evaluation' in logs. Got: {[r.message for r in caplog.records]}"

                        # Verify logger.info was called for Best Ensemble
                        assert any(
                            "Best Ensemble on Validation Set" in str(record.message)
                            for record in caplog.records
                        ), "Expected 'Best Ensemble on Validation Set' in logs"

    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_main_plot_flag_functionality(caplog):
    """Test the plot flag functionality in main() - tests the plot code path."""
    from main import main

    tmp_dir = tempfile.mkdtemp()
    try:
        config_path = os.path.join(tmp_dir, "test_config.yml")

        # Write minimal config
        config_content = """
global_params:
  input_csv_path: "test_data.csv"
  n_iter: 1
  testing: True
grid_params:
  outcome_var_n: [1]
"""
        with open(config_path, "w") as f:
            f.write(config_content)

        # Create a mock results CSV file
        results_csv = os.path.join(tmp_dir, "final_grid_score_log.csv")
        df = pd.DataFrame(
            {
                "auc": [0.85],
                "best_ensemble": ['[(1.0, "LogisticRegression", [1, 0, 1])]'],
                "X_train_size": [100],
                "X_test_orig_size": [50],
                "X_test_size": [50],
            }
        )
        df.to_csv(results_csv, index=False)

        # Create initial test data file
        test_data = os.path.join(tmp_dir, "test_data.csv")
        pd.DataFrame({"age": [25.0], "male": [1.0], "outcome_var_1": [1.0]}).to_csv(
            test_data, index=False
        )

        # Mock data.pipe
        def mock_data_pipe(*args, **kwargs):
            mock_obj = MagicMock()
            mock_obj.X_train = MagicMock()
            mock_obj.y_train = MagicMock()
            mock_obj.X_test = MagicMock()
            mock_obj.y_test = MagicMock()
            mock_obj.X_test_orig = pd.DataFrame({"age": [30.0], "male": [1.0]})
            mock_obj.y_test_orig = pd.Series([1.0])
            mock_obj.original_feature_names = ["age", "male", "outcome_var_1"]
            mock_obj.local_param_dict = {"weighted": "unweighted"}
            mock_obj.verbose = 0
            mock_obj.base_project_dir = tmp_dir
            return mock_obj

        with patch("main.data.pipe", side_effect=mock_data_pipe):

            # Mock main_ga.run
            def mock_main_ga_run(*args, **kwargs):
                mock_obj = MagicMock()
                mock_obj.execute.return_value = None
                return mock_obj

            with patch("ml_grid.pipeline.main_ga.run", side_effect=mock_main_ga_run):

                # Mock global_parameters
                def mock_global_parameters(config_path=None, **kwargs):
                    gp_mock = MagicMock()
                    gp_mock.n_iter = 1
                    gp_mock.testing = True
                    gp_mock.input_csv_path = test_data
                    gp_mock.base_project_dir = tmp_dir
                    gp_mock.verbose = 0
                    gp_mock.model_list = []
                    return gp_mock

                with patch(
                    "main.global_parameters", side_effect=mock_global_parameters
                ):

                    # Patch read_csv for CSV reading in plot section
                    def read_csv_side_effect(filepath, *args, **kwargs):
                        if filepath == results_csv:
                            return pd.DataFrame(
                                {
                                    "auc": [0.85],
                                    "best_ensemble": [
                                        '[(1.0, "LogisticRegression", [1, 0, 1])]'
                                    ],
                                    "X_train_size": [100],
                                    "X_test_orig_size": [50],
                                    "X_test_size": [50],
                                }
                            )
                        elif "test_data.csv" in str(filepath):
                            return pd.DataFrame(
                                {"age": [25.0], "male": [1.0], "outcome_var_1": [1.0]}
                            )
                        result = pd.DataFrame(
                            {
                                "auc": [0.85],
                            }
                        )
                        result.index = pd.RangeIndex(len(result))
                        return result

                    with patch("pandas.read_csv", side_effect=read_csv_side_effect):

                        # Mock GA_results_explorer
                        mock_explorer_instance = MagicMock()
                        mock_explorer_instance.run_all_plots.return_value = None

                        with patch("main.GA_results_explorer") as mock_explorer_class:
                            mock_explorer_class.return_value = mock_explorer_instance

                            # Execute main with plot=True
                            main(config_path=config_path, plot=True)

                            # Verify logger.info was called for plot generation start
                            assert any(
                                "Starting plot generation" in record.message
                                for record in caplog.records
                            ), f"Expected 'Starting plot generation' in logs. Got: {[r.message for r in caplog.records]}"

                            # Verify GA_results_explorer was instantiated and run
                            assert (
                                mock_explorer_class.called
                            ), "GA_results_explorer should be called"
                            mock_explorer_instance.run_all_plots.assert_called_once()

    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_main_full_execution_with_mock():
    """Test full main() execution flow including grid search iterations."""
    from main import main

    tmp_dir = tempfile.mkdtemp()
    try:
        config_path = os.path.join(tmp_dir, "test_config.yml")

        # Write minimal config
        config_content = """
global_params:
  input_csv_path: "test_data.csv"
  n_iter: 2
  testing: True
grid_params:
  outcome_var_n: [1]
"""
        with open(config_path, "w") as f:
            f.write(config_content)

        iteration_count = {"count": 0}
        iterations_executed = []

        def mock_global_parameters(config_path=None, **kwargs):
            gp_mock = MagicMock()
            gp_mock.n_iter = 2
            gp_mock.testing = True
            gp_mock.input_csv_path = os.path.join(tmp_dir, "test_data.csv")
            gp_mock.base_project_dir = tmp_dir
            gp_mock.verbose = 0
            gp_mock.model_list = []
            return gp_mock

        def mock_data_pipe(*args, **kwargs):
            iteration_count["count"] += 1
            iterations_executed.append(kwargs.get("param_space_index", -1))

            mock_obj = MagicMock()
            mock_obj.X_train = MagicMock()
            mock_obj.y_train = MagicMock()
            mock_obj.X_test = MagicMock()
            mock_obj.y_test = MagicMock()
            mock_obj.X_test_orig = pd.DataFrame({"age": [30.0], "male": [1.0]})
            mock_obj.y_test_orig = pd.Series([1.0])
            mock_obj.original_feature_names = ["age", "male", "outcome_var_1"]
            mock_obj.local_param_dict = {"weighted": "unweighted"}
            mock_obj.verbose = 0
            mock_obj.base_project_dir = tmp_dir
            return mock_obj

        def mock_main_ga_run(*args, **kwargs):
            mock_obj = MagicMock()
            mock_obj.execute.return_value = None
            return mock_obj

        with (
            patch("main.global_parameters", side_effect=mock_global_parameters),
            patch("main.data.pipe", side_effect=mock_data_pipe),
            patch("ml_grid.pipeline.main_ga.run", side_effect=mock_main_ga_run),
        ):

            # Execute main function
            main(config_path=config_path)

            # Verify exact number of iterations occurred
            assert (
                iteration_count["count"] == 2
            ), f"Expected 2 iterations but got {iteration_count['count']}"

    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
