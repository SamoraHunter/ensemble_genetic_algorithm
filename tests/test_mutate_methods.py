"""Tests for mutate_methods module."""

import unittest
from unittest.mock import MagicMock

from ml_grid.pipeline.mutate_methods import baseLearnerGenerator, mutateEnsemble


class TestBaseLearnerGenerator:
    """Tests for the baseLearnerGenerator function."""

    def test_base_learner_generator_calls_model_func(self):
        """Test that baseLearnerGenerator calls a model generator function."""
        mock_ml_grid = MagicMock()
        mock_ml_grid.config_dict = {"modelFuncList": [MagicMock()]}
        mock_ml_grid.local_param_dict = {}

        expected_result = (0.8, "model_obj", ["feat1"], 10, 0.9, None)
        mock_ml_grid.config_dict["modelFuncList"][0].return_value = expected_result

        result = baseLearnerGenerator(mock_ml_grid)

        assert result == expected_result
        mock_ml_grid.config_dict["modelFuncList"][0].assert_called_once_with(
            mock_ml_grid, mock_ml_grid.local_param_dict
        )


class TestMutateEnsemble(unittest.TestCase):
    """Tests for the mutateEnsemble function."""

    def test_mutate_ensemble_normal_operation(self):
        """Test mutateEnsemble with a normal individual."""
        mock_ml_grid = MagicMock()

        original_model = (0.8, "model_obj", ["feat1"], 10, 0.9, None)
        individual = [[original_model]]

        new_model = (0.75, "new_model", ["feat2"], 5, 0.85, None)

        import ml_grid.pipeline.mutate_methods as mut_mod

        original_gen = mut_mod.baseLearnerGenerator
        mut_mod.baseLearnerGenerator = MagicMock(return_value=new_model)

        try:
            result = mutateEnsemble(individual, mock_ml_grid)

            assert len(result[0]) == 1
            assert result[0][0] == new_model
        finally:
            mut_mod.baseLearnerGenerator = original_gen

    def test_mutate_ensemble_with_single_model(self):
        """Test mutateEnsemble handles an individual with a single model."""
        mock_ml_grid = MagicMock()

        # Individual with one model - should still work (pops it, adds new)
        original_model = (0.8, "model_obj", ["feat1"], 10, 0.9, None)
        individual = [[original_model]]

        new_model = (0.75, "new_model", ["feat2"], 5, 0.85, None)

        import ml_grid.pipeline.mutate_methods as mut_mod

        original_gen = mut_mod.baseLearnerGenerator
        mut_mod.baseLearnerGenerator = MagicMock(return_value=new_model)

        try:
            result = mutateEnsemble(individual, mock_ml_grid)

            assert len(result[0]) == 1
            assert result[0][0] == new_model
        finally:
            mut_mod.baseLearnerGenerator = original_gen

    def test_mutate_ensemble_with_multiple_models(self):
        """Test mutateEnsemble with an individual containing multiple models."""
        mock_ml_grid = MagicMock()

        model1 = (0.8, "model1", ["feat1"], 10, 0.9, None)
        model2 = (0.75, "model2", ["feat2"], 5, 0.85, None)
        individual = [[model1, model2]]

        new_model = (0.7, "new_model", ["feat3"], 5, 0.8, None)

        import ml_grid.pipeline.mutate_methods as mut_mod

        original_gen = mut_mod.baseLearnerGenerator
        mut_mod.baseLearnerGenerator = MagicMock(return_value=new_model)

        try:
            result = mutateEnsemble(individual, mock_ml_grid)

            assert len(result[0]) == 2
            assert new_model in result[0]
            assert model1 not in result[0] or model2 not in result[0]
        finally:
            mut_mod.baseLearnerGenerator = original_gen

    def test_mutate_ensemble_with_empty_models(self):
        """Test mutateEnsemble handles an individual with empty model list."""
        mock_ml_grid = MagicMock()

        # Individual with empty model list - should pop 0 (fallback)
        individual = [[]]

        new_model = (0.75, "new_model", ["feat2"], 5, 0.85, None)

        import ml_grid.pipeline.mutate_methods as mut_mod

        original_gen = mut_mod.baseLearnerGenerator
        mut_mod.baseLearnerGenerator = MagicMock(return_value=new_model)

        try:
            result = mutateEnsemble(individual, mock_ml_grid)

            assert len(result[0]) == 1
            assert result[0][0] == new_model
        finally:
            mut_mod.baseLearnerGenerator = original_gen

    def test_mutate_ensemble_exception_handling(self):
        """Test mutateEnsemble exception handler when baseLearnerGenerator fails."""
        mock_ml_grid = MagicMock()

        # Individual with one model that will be popped, then generator fails
        original_model = (0.8, "model_obj", ["feat1"], 10, 0.9, None)
        individual = [[original_model]]

        import ml_grid.pipeline.mutate_methods as mut_mod

        original_gen = mut_mod.baseLearnerGenerator
        mut_mod.baseLearnerGenerator = MagicMock(side_effect=ValueError("Test error"))

        try:
            with self.assertRaises(ValueError):
                mutateEnsemble(individual, mock_ml_grid)
        finally:
            mut_mod.baseLearnerGenerator = original_gen
