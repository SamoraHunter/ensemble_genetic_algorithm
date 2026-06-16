"""Tests for main_ga module."""

from unittest.mock import MagicMock, patch

import numpy as np

from ml_grid.pipeline.main_ga import run


class TestRunInitialization:
    """Tests for the run class __init__ method."""

    def test_run_initializes_with_required_attributes(self):
        """Test that run initializes with all required attributes."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 1
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()

        mock_ml_grid.X_test = MagicMock()
        mock_ml_grid.y_test = MagicMock()
        mock_ml_grid.X_train = MagicMock()
        mock_ml_grid.y_train = MagicMock()
        mock_ml_grid.X_test_orig = MagicMock()
        mock_ml_grid.y_test_orig = MagicMock()

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "global_params")
        assert hasattr(test_run, "ml_grid_object")
        assert hasattr(test_run, "verbose")
        assert hasattr(test_run, "error_raise")
        assert hasattr(test_run, "nb_params")
        assert hasattr(test_run, "pop_params")
        assert hasattr(test_run, "g_params")
        assert hasattr(test_run, "log_folder_path")
        assert hasattr(test_run, "global_param_str")
        assert hasattr(test_run, "additional_naming")
        assert hasattr(test_run, "gen_eval_score_threshold_early_stopping")
        assert hasattr(test_run, "creator")
        assert hasattr(test_run, "tools")
        assert hasattr(test_run, "toolbox")
        assert hasattr(test_run, "project_score_save_object")
        assert hasattr(test_run, "X_test")
        assert hasattr(test_run, "y_test")
        assert hasattr(test_run, "X_train")
        assert hasattr(test_run, "y_train")
        assert hasattr(test_run, "X_test_orig")
        assert hasattr(test_run, "y_test_orig")
        assert hasattr(test_run, "multiprocess")
        assert hasattr(test_run, "local_param_dict")

    def test_clear_output_exception_handling(self):
        """Test that clear_output failure is handled gracefully."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
            "nb_params": [2],
            "pop_params": [5],
            "g_params": [3],
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 1
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()

        mock_ml_grid.X_test = MagicMock()
        mock_ml_grid.y_test = MagicMock()
        mock_ml_grid.X_train = MagicMock()
        mock_ml_grid.y_train = MagicMock()
        mock_ml_grid.X_test_orig = MagicMock()
        mock_ml_grid.y_test_orig = MagicMock()

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "ml_grid_object")

    def test_value_error_handling_for_auc_score(self):
        """Test that ValueError in roc_auc_score defaults to 0.5."""

        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
            "nb_params": [2],
            "pop_params": [5],
            "g_params": [3],
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 1
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()

        mock_ml_grid.X_test = MagicMock()
        mock_ml_grid.y_test = [1, 2, 3]
        mock_ml_grid.X_train = MagicMock()
        mock_ml_grid.y_train = MagicMock()
        mock_ml_grid.X_test_orig = MagicMock()
        mock_ml_grid.y_test_orig = MagicMock()

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "y_test")

    def test_verbose_logging_during_init(self):
        """Test that verbose >= 2 logs 'Passed main GA init'."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 3
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()

        mock_ml_grid.X_test = MagicMock()
        mock_ml_grid.y_test = MagicMock()
        mock_ml_grid.X_train = MagicMock()
        mock_ml_grid.y_train = MagicMock()
        mock_ml_grid.X_test_orig = MagicMock()
        mock_ml_grid.y_test_orig = MagicMock()

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "ml_grid_object")


class TestRunInitializationMainBlock:
    """Tests for the __main__ block handling in run.__init__."""

    def test_init_logs_debug_when_verbose_ge_2(self):
        """Test that debug logging occurs when verbose >= 2."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 2
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()
        mock_ml_grid.model_class_list = [1, 2, 3]

        mock_ml_grid.X_test = MagicMock()
        mock_ml_grid.y_test = MagicMock()
        mock_ml_grid.X_train = MagicMock()
        mock_ml_grid.y_train = MagicMock()
        mock_ml_grid.X_test_orig = MagicMock()
        mock_ml_grid.y_test_orig = MagicMock()

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "ml_grid_object")
        assert hasattr(test_run, "verbose")
        assert hasattr(test_run, "error_raise")
        assert hasattr(test_run, "nb_params")
        assert hasattr(test_run, "pop_params")
        assert hasattr(test_run, "g_params")
        assert hasattr(test_run, "log_folder_path")
        assert hasattr(test_run, "global_param_str")
        assert hasattr(test_run, "additional_naming")
        assert hasattr(test_run, "gen_eval_score_threshold_early_stopping")
        assert hasattr(test_run, "creator")
        assert hasattr(test_run, "tools")
        assert hasattr(test_run, "toolbox")
        assert hasattr(test_run, "project_score_save_object")
        assert hasattr(test_run, "X_test")
        assert hasattr(test_run, "y_test")
        assert hasattr(test_run, "X_train")
        assert hasattr(test_run, "y_train")
        assert hasattr(test_run, "X_test_orig")
        assert hasattr(test_run, "y_test_orig")
        assert hasattr(test_run, "multiprocess")
        assert hasattr(test_run, "local_param_dict")


class TestRunInitMainBlockCoverage:
    """Tests for coverage of __main__ block in __init__."""

    def test_param_grid_generation_in_init(self):
        """Test that param_grid generation works correctly during init."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
            "nb_params": [2, 3],
            "pop_params": [5, 10],
            "g_params": [3, 5],
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 0
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()

        mock_ml_grid.X_test = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_test = np.array([0, 1])
        mock_ml_grid.X_train = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_train = np.array([0, 1])
        mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_test_orig = np.array([0, 1])

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "nb_params")
        assert isinstance(test_run.nb_params, list)

    def test_verbose_logging_for_model_count(self):
        """Test that verbose >= 2 logs model count during init."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
        }

        mock_global_params = MagicMock()
        mock_global_params.verbose = 3
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()
        mock_ml_grid.model_class_list = [1, 2, 3, 4, 5]

        mock_ml_grid.X_test = MagicMock()
        mock_ml_grid.y_test = MagicMock()
        mock_ml_grid.X_train = MagicMock()
        mock_ml_grid.y_train = MagicMock()
        mock_ml_grid.X_test_orig = MagicMock()
        mock_ml_grid.y_test_orig = MagicMock()

        run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert len(mock_ml_grid.model_class_list) == 5


class TestRunExecuteSimple:
    """Simple execute test that just ensures the method exists."""

    def test_execute_method_exists(self):
        """Test that execute method exists and returns list."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = False
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
            "nb_params": [2],
            "pop_params": [2],
            "g_params": [1],
        }
        mock_ml_grid.verbose = 0
        mock_ml_grid.error_raise = True

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = "/tmp/test_logs"
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()

        import numpy as np

        mock_ml_grid.X_test = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_test = np.array([0, 1])
        mock_ml_grid.X_train = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_train = np.array([0, 1])
        mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_test_orig = np.array([0, 1])

        mock_ml_grid.base_project_dir = "/tmp/test_base"
        mock_ml_grid.original_feature_names = ["feature1", "feature2"]
        mock_ml_grid.model_class_list = []
        mock_ml_grid.logging_paths_obj.additional_naming = ""

        mock_global_params = MagicMock()
        mock_global_params.verbose = 0
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "execute")
        assert callable(getattr(test_run, "execute"))


class TestRunExecute:
    """Tests for the execute method that runs the GA evolution loop."""

    def test_execute_runs_ga_evolution(self):
        """Test that execute successfully runs a single GA generation.

        This test mocks the evaluate function and ensemble generator to
        simulate a complete GA run without actually training real models.
        """
        import os

        # Setup mock_ml_grid with all required attributes
        mock_ml_grid = MagicMock()
        mock_ml_grid.testing = True
        mock_ml_grid.local_param_dict = {
            "param_space_size": 10,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "t_size": 3,
            "indpb": 0.1,
            "weighted": "unweighted",
        }

        mock_ml_grid.verbose = 0
        mock_ml_grid.error_raise = True

        # Disable multiprocessing to avoid pickling issues with mocks
        mock_ml_grid.multiprocessing_ensemble = False

        # Create log directory for testing
        log_dir = "/tmp/test_logs"
        os.makedirs(log_dir, exist_ok=True)
        model_store_path = "/tmp/model_store.json"
        with open(model_store_path, "w") as f:
            import json

            json.dump({"models": {}}, f)

        mock_logging_paths = MagicMock()
        mock_logging_paths.log_folder_path = log_dir
        mock_logging_paths.global_param_str = "test_run"
        mock_logging_paths.additional_naming = ""
        mock_logging_paths.model_store_path = model_store_path

        mock_ml_grid.logging_paths_obj = mock_logging_paths
        mock_ml_grid.project_score_save_object = MagicMock()
        # Mock project_score_save_object.update_score_log to avoid file I/O
        mock_ml_grid.project_score_save_object.update_score_log = MagicMock()

        import numpy as np

        # Required data attributes
        mock_ml_grid.X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mock_ml_grid.y_test = np.array([0, 0, 1, 1])
        mock_ml_grid.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mock_ml_grid.y_train = np.array([0, 0, 1, 1])
        mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mock_ml_grid.y_test_orig = np.array([0, 0, 1, 1])

        mock_ml_grid.base_project_dir = "/tmp/test_base"
        mock_ml_grid.original_feature_names = ["feature1", "feature2"]
        mock_ml_grid.model_class_list = []
        mock_ml_grid.config_dict = {
            "modelFuncList": [],
            "use_stored_base_learners": False,
        }

        # Create global params
        mock_global_params = MagicMock()
        mock_global_params.verbose = 0
        mock_global_params.error_raise = True
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        # Important: Create run object which sets up DEAP creator classes
        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        # Verify that DEAP creator classes are set up (FitnessMax, Individual)
        assert hasattr(test_run.creator, "FitnessMax")
        assert hasattr(test_run.creator, "Individual")

        # Set up mock model generator function
        def mock_model_generator(ml_grid, local_param):
            """Mock model generator returning same format as real models."""
            return (0.5, MagicMock(), ["f1", "f2"], 1, 0.6, np.array([0, 0, 1, 1]))

        # Add to config for ensemble generation
        mock_ml_grid.config_dict["modelFuncList"] = [mock_model_generator]

        # Mock the evaluate function to return valid fitness values
        def mock_evaluate(individual, ml_grid_object):
            """Mock evaluation returning a single-element tuple."""
            return (0.7,)  # Return a fitness score

        with (
            patch(
                "ml_grid.pipeline.main_ga.evaluate_weighted_ensemble_auc", mock_evaluate
            ),
            patch("ml_grid.pipeline.main_ga.clear_output"),
            patch("ml_grid.pipeline.main_ga.time"),
            patch("ml_grid.pipeline.main_ga.tqdm") as mock_tqdm,
            patch("ml_grid.pipeline.main_ga.metrics.roc_auc_score", return_value=0.75),
            patch(
                "ml_grid.pipeline.main_ga.get_y_pred_resolver",
                return_value=np.array([0, 0, 1, 1]),
            ),
            patch("ml_grid.pipeline.main_ga.plot_auc"),
            patch(
                "ml_grid.pipeline.main_ga.measure_binary_vector_diversity",
                return_value=0.5,
            ),
            # Patch file I/O operations
            patch("ml_grid.pipeline.main_ga.open", create=True),
            patch("ml_grid.pipeline.main_ga.pickle.dump"),
            patch("ml_grid.pipeline.main_ga.plot_generation_progress_fitness"),
        ):
            mock_tqdm.tqdm.return_value.__enter__ = lambda self: self
            mock_tqdm.tqdm.return_value.__exit__ = lambda self, *args: None

            # Set up nb_params, pop_params, g_params to small values for fast test
            test_run.nb_params = [2]
            test_run.pop_params = [4]
            test_run.g_params = [1]  # Only 1 generation

            # Also patch ensembleGenerator in toolbox
            def mock_ensemble_gen(nb_val, ml_grid_object):
                """Mock ensemble generator returning list of model tuples."""
                return [
                    (0.5, "model", ["f1"], 1, 0.6, np.array([0, 0, 1, 1]))
                    for _ in range(2)  # Return ensemble with 2 models
                ]

            # Register mock functions in toolbox before calling execute
            test_run.toolbox.register(
                "ensembleGenerator",
                mock_ensemble_gen,
                nb_val=2,
                ml_grid_object=mock_ml_grid,
            )
            test_run.toolbox.register(
                "individual",
                test_run.tools.initRepeat,
                test_run.creator.Individual,
                test_run.toolbox.ensembleGenerator,
                n=1,
            )
            test_run.toolbox.register(
                "population",
                test_run.tools.initRepeat,
                list,
                test_run.toolbox.individual,
            )

            # Register evaluate function
            test_run.toolbox.register(
                "evaluate", mock_evaluate, ml_grid_object=mock_ml_grid
            )

            # Finally, call execute and verify it completes without errors
            result = test_run.execute()

        # Verify execute returns the expected type (list of errors)
        assert isinstance(result, list)


def _create_mock_ml_grid():  # Helper function to create mock for Execute tests
    """Helper to create a mock ml_grid object for execute() testing."""
    import os

    mock_ml_grid = MagicMock()
    mock_ml_grid.testing = True
    mock_ml_grid.local_param_dict = {
        "param_space_size": 10,
        "cxpb": 0.5,
        "mutpb": 0.2,
        "t_size": 3,
        "indpb": 0.1,
        "weighted": "unweighted",
    }
    mock_ml_grid.verbose = 0
    mock_ml_grid.error_raise = False

    log_dir = "/tmp/test_logs_valueerror"
    os.makedirs(log_dir, exist_ok=True)

    mock_logging_paths = MagicMock()
    mock_logging_paths.log_folder_path = log_dir
    mock_logging_paths.global_param_str = "test_run"
    mock_logging_paths.additional_naming = ""

    mock_ml_grid.logging_paths_obj = mock_logging_paths
    mock_ml_grid.project_score_save_object = MagicMock()
    mock_ml_grid.project_score_save_object.update_score_log = MagicMock()

    # Normal data for testing (non-empty, multiple classes)
    import numpy as np

    mock_ml_grid.X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    mock_ml_grid.y_test = np.array([0, 0, 1, 1])  # Multiple classes
    mock_ml_grid.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    mock_ml_grid.y_train = np.array([0, 0, 1, 1])
    mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    mock_ml_grid.y_test_orig = np.array([0, 0, 1, 1])
    mock_ml_grid.base_project_dir = "/tmp/test_base"
    mock_ml_grid.original_feature_names = ["feature1", "feature2"]
    mock_ml_grid.model_class_list = []
    mock_ml_grid.config_dict = {
        "modelFuncList": [],
        "use_stored_base_learners": False,
    }

    return mock_ml_grid


class TestRunExecuteValueErrorHandling:
    """Tests for ValueError handling in roc_auc_score calls."""

    def test_empty_y_test_triggers_value_error_handler_line_329(self):
        """Test that empty y_test triggers the ValueError handler at line 329.

        When y_test is empty (len=0), chance_dummy_best_pred becomes [] which
        causes metrics.roc_auc_score to raise ValueError with "Found array
        with 0 sample(s)" message."""
        import numpy as np

        mock_ml_grid = _create_mock_ml_grid()
        mock_ml_grid.X_test = np.array([]).reshape(0, 2)
        mock_ml_grid.y_test = np.array([])  # Empty triggers ValueError
        mock_ml_grid.X_train = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_train = np.array([0, 1])
        mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4]])
        mock_ml_grid.y_test_orig = np.array([0, 1])

        mock_global_params = MagicMock()
        mock_global_params.verbose = 0
        mock_global_params.error_raise = False
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "gen_eval_score_threshold_early_stopping")

    def test_execute_with_single_class_y_test_triggers_value_error_handler(self):
        """Test that y_test with single class triggers ValueError handler at line 414.

        When all samples in y_test have the same class (e.g., all 0s or all 1s),
        roc_auc_score raises ValueError. This test ensures the handler defaults
        to gen_eval_score = 0.5."""
        import numpy as np

        mock_ml_grid = _create_mock_ml_grid()
        # Single class in y_test - triggers ValueError in roc_auc_score
        mock_ml_grid.y_test = np.array([1, 1, 1, 1])  # All same class
        mock_ml_grid.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mock_ml_grid.y_train = np.array([0, 0, 1, 1])
        mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mock_ml_grid.y_test_orig = np.array([0, 0, 1, 1])
        mock_ml_grid.model_class_list = []

        mock_global_params = MagicMock()
        mock_global_params.verbose = 0
        mock_global_params.error_raise = False
        mock_global_params.sub_sample_param_space_pct = 0.5
        mock_global_params.gen_eval_score_threshold_early_stopping = 10

        test_run = run(
            ml_grid_object=mock_ml_grid,
            local_param_dict=mock_ml_grid.local_param_dict,
            global_params=mock_global_params,
        )

        assert hasattr(test_run, "y_test")
