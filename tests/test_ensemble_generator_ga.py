# ruff: noqa: PLC2401
"""Tests for ensemble_generator_ga module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_grid.pipeline.ensemble_generator_ga import (
    do_work,
    ensembleGenerator,
    multi_run_wrapper,
)


class TestMultiRunWrapper:
    """Tests for the multi_run_wrapper function."""

    def test_multi_run_wrapper_calls_do_work(self):
        """Test that multi_run_wrapper correctly unpacks args and calls do_work."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 1

        # Create a mock model function that returns a known tuple
        def mock_model_func(ml_grid, local_param):
            return (0.8, "model_obj", ["feat1"], 10, 0.9, None)

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
        }
        mock_ml_grid.local_param_dict = {}

        # Mock random module to control which index is chosen
        with patch("ml_grid.pipeline.ensemble_generator_ga.random") as mock_random:
            mock_random.randint.return_value = 0

            result = multi_run_wrapper((5, mock_ml_grid))

            assert isinstance(result, tuple)
            assert result[0] == 0.8


class TestEnsembleGenerator:
    """Tests for the ensembleGenerator function."""

    def test_ensemble_generator_with_nb_val_less_than_or_equal_to_1(self):
        """Test that ensembleGenerator calls baseLearnerGenerator when nb_val <= 1."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 1
        mock_ml_grid.multiprocessing_ensemble = False

        # Create a mock model function that returns a known tuple
        def mock_model_func(ml_grid, local_param):
            return (0.8, "model_obj", ["feat1"], 10, 0.9, None)

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
        }
        mock_ml_grid.local_param_dict = {}

        # Test with nb_val <= 1 - should call baseLearnerGenerator
        # Force random choices to get predictable results
        with (
            patch("ml_grid.pipeline.ensemble_generator_ga.skewnorm") as mock_skew,
            patch("ml_grid.pipeline.ensemble_generator_ga.np.random") as mock_np_random,
        ):
            # Make skewnorm return consistent values
            mock_skew.rvs.return_value = np.array([10] * 10000)
            # Force random.choice to select a specific value
            mock_np_random.choice.side_effect = [2, 3]

            result = ensembleGenerator(nb_val=1, ml_grid_object=mock_ml_grid)

        assert isinstance(result, list)

    def test_ensemble_generator_with_nb_val_greater_than_1(self):
        """Test that ensembleGenerator generates multiple models when nb_val > 1."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 1
        mock_ml_grid.multiprocessing_ensemble = False

        # Create a mock model function that returns a known tuple
        def mock_model_func(ml_grid, local_param):
            return (0.8, "model_obj", ["feat1"], 10, 0.9, None)

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
        }
        mock_ml_grid.local_param_dict = {}

        # Test with nb_val > 1
        with (
            patch("ml_grid.pipeline.ensemble_generator_ga.skewnorm") as mock_skew,
            patch("ml_grid.pipeline.ensemble_generator_ga.np.random") as mock_np_random,
        ):
            mock_skew.rvs.return_value = np.array([10] * 10000)
            mock_np_random.choice.side_effect = [3]

            result = ensembleGenerator(nb_val=5, ml_grid_object=mock_ml_grid)

        assert isinstance(result, list)
        # Should have generated a few models
        assert len(result) >= 2


class TestDoWork:
    """Tests for the do_work function."""

    def test_do_work_with_use_stored_base_learners_true(self):
        """Test that do_work takes the stored model branch when use_stored_base_learners=True and random.random() > 0.5."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 2

        # Mock model function for the else branch
        def mock_model_func(ml_grid, local_param):
            return (0.8, "model_obj", ["feat1"], 10, 0.9, None)

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
            "use_stored_base_learners": True,
        }
        mock_ml_grid.local_param_dict = {}

        # Patch both random and get_stored_model
        with (
            patch("ml_grid.pipeline.ensemble_generator_ga.random") as mock_random,
            patch(
                "ml_grid.pipeline.ensemble_generator_ga.get_stored_model"
            ) as mock_get_stored,
        ):
            # Make random.random() return > 0.5 to trigger stored model branch
            mock_random.random.return_value = 0.7
            # Configure get_stored_model to return a known value
            mock_get_stored.return_value = (
                0.95,
                "stored_model",
                ["feat1"],
                10,
                0.95,
                None,
            )

            result = do_work(ml_grid_object=mock_ml_grid)

            assert result == (0.95, "stored_model", ["feat1"], 10, 0.95, None)
            mock_get_stored.assert_called_once_with(mock_ml_grid)

    def test_do_work_raises_exception_get_stored_model(self):
        """Test that do_work raises exception when get_stored_model fails."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 11

        # Mock model function for the else branch
        def mock_model_func(ml_grid, local_param):
            return (0.8, "model_obj", ["feat1"], 10, 0.9, None)

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
            "use_stored_base_learners": True,
        }
        mock_ml_grid.local_param_dict = {}

        # Patch random and get_stored_model to raise exception
        with (
            patch("ml_grid.pipeline.ensemble_generator_ga.random") as mock_random,
            patch(
                "ml_grid.pipeline.ensemble_generator_ga.get_stored_model"
            ) as mock_get_stored,
            patch("ml_grid.pipeline.ensemble_generator_ga.logger") as mock_logger,
        ):
            mock_random.random.return_value = 0.7
            mock_get_stored.side_effect = ValueError("Test error")

            with pytest.raises(ValueError):
                do_work(ml_grid_object=mock_ml_grid)

            assert len(mock_logger.error.call_args_list) >= 2

    def test_do_work_verbose_logging(self):
        """Test that verbose >= 11 triggers debug logging for do_work."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 11

        def mock_model_func(ml_grid, local_param):
            return (0.8, "model_obj", ["feat1"], 10, 0.9, None)

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
            "use_stored_base_learners": False,
        }
        mock_ml_grid.local_param_dict = {}

        with (
            patch("ml_grid.pipeline.ensemble_generator_ga.random") as mock_random,
            patch("ml_grid.pipeline.ensemble_generator_ga.logger") as mock_logger,
        ):
            mock_random.randint.return_value = 0
            mock_random.random.return_value = 0

            do_work(ml_grid_object=mock_ml_grid)

            mock_logger.debug.assert_any_call("do_work")

    def test_do_work_raises_exception_new_model_generation(self):
        """Test that do_work raises exception when new model generation fails."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.verbose = 11

        def mock_model_func(ml_grid, local_param):
            raise ValueError("Model creation failed")

        mock_ml_grid.config_dict = {
            "modelFuncList": [mock_model_func],
            "use_stored_base_learners": False,
        }
        mock_ml_grid.local_param_dict = {}

        with (
            patch("ml_grid.pipeline.ensemble_generator_ga.random") as mock_random,
            patch("ml_grid.pipeline.ensemble_generator_ga.logger") as mock_logger,
        ):
            mock_random.randint.return_value = 0
            mock_random.random.return_value = 0

            with pytest.raises(ValueError):
                do_work(ml_grid_object=mock_ml_grid)

            assert len(mock_logger.error.call_args_list) >= 2
