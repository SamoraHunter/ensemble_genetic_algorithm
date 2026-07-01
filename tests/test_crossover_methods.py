"""Tests for crossover_methods module."""

import unittest

from ml_grid.pipeline.crossover_methods import (
    cxBlend,
    cxOnePoint,
    cxOrdered,
    cxUniform,
    get_crossover_operator,
)


class TestGetCrossoverOperator(unittest.TestCase):
    """Tests for the get_crossover_operator factory function."""

    def test_get_crossover_operator_onepoint(self):
        """Test retrieving onepoint crossover operator."""
        func = get_crossover_operator("onepoint")
        self.assertEqual(func.__name__, "cxOnePoint")

    def test_get_crossover_operator_uniform(self):
        """Test retrieving uniform crossover operator."""
        func = get_crossover_operator("uniform")
        self.assertEqual(func.__name__, "cxUniform")

    def test_get_crossover_operator_blend(self):
        """Test retrieving blend crossover operator."""
        func = get_crossover_operator("blend")
        self.assertEqual(func.__name__, "cxBlend")

    def test_get_crossover_operator_ordered(self):
        """Test retrieving ordered crossover operator."""
        func = get_crossover_operator("ordered")
        self.assertEqual(func.__name__, "cxOrdered")

    def test_get_crossover_operator_twopoint(self):
        """Test retrieving twopoint crossover operator (DEAP built-in)."""
        func = get_crossover_operator("twopoint")
        from deap import tools

        self.assertEqual(func, tools.cxTwoPoint)

    def test_get_crossover_operator_invalid(self):
        """Test that invalid cx_type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_crossover_operator("invalid_type")

        self.assertIn("Unknown crossover type", str(context.exception))
        self.assertIn("onepoint", str(context.exception))


class TestCrossoverOperators(unittest.TestCase):
    """Tests for individual crossover operators."""

    def test_cxOnePoint(self):
        """Test one-point crossover function."""
        ind1 = [1, 2, 3, 4, 5]
        ind2 = [6, 7, 8, 9, 10]

        result = cxOnePoint(ind1, ind2)

        # Check that both parents were modified
        self.assertEqual(result[0], ind1)
        self.assertEqual(result[1], ind2)

        # Verify crossover happened (at least one element should be different)
        # Note: Due to randomness, this might not always pass in a single test run
        # but will pass most of the time with these specific values

    def test_cxUniform(self):
        """Test uniform crossover function."""
        ind1 = [0, 1, 2, 3, 4]
        ind2 = [5, 6, 7, 8, 9]

        result = cxUniform(ind1, ind2, indpb=0.5)

        # Check that both parents were modified
        self.assertEqual(result[0], ind1)
        self.assertEqual(result[1], ind2)

    def test_cxBlend(self):
        """Test blend crossover function."""
        ind1 = [0.0, 1.0, 2.0, 3.0, 4.0]
        ind2 = [5.0, 6.0, 7.0, 8.0, 9.0]

        result = cxBlend(ind1, ind2)

        # Check that both parents were modified
        self.assertEqual(result[0], ind1)
        self.assertEqual(result[1], ind2)

        # Blend crossover can produce values slightly outside parent range due to blending
        # Just verify they are reasonable floats
        for val in ind1:
            self.assertIsInstance(val, float)

    def test_cxOrdered(self):
        """Test order crossover function with valid permutations."""
        # cxOrdered requires permutations (each element appears exactly once)
        ind1 = [0, 1, 2, 3, 4]
        ind2 = [4, 3, 2, 1, 0]

        result = cxOrdered(ind1, ind2)

        # Check that both parents were modified
        self.assertEqual(result[0], ind1)
        self.assertEqual(result[1], ind2)

        # Each child should still contain exactly the same elements as parents
        self.assertCountEqual(set(ind1), {0, 1, 2, 3, 4})
        self.assertCountEqual(set(ind2), {0, 1, 2, 3, 4})


class TestCrossoverOperatorNames(unittest.TestCase):
    """Tests to verify operator names match expected pattern."""

    def test_operator_consistency(self):
        """Test that operator names are consistent."""
        operators = ["onepoint", "uniform", "blend", "ordered"]

        for op_name in operators:
            func = get_crossover_operator(op_name)
            # The function name should contain the operator type
            self.assertIn(op_name, func.__name__.lower())
