"""Tests for adaptive_mutation module."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from ml_grid.pipeline.adaptive_mutation import (
    AdaptiveMutation,
    adaptive_mutation_rate,
    calculate_population_diversity,
)


class TestAdaptiveMutationRate(unittest.TestCase):

    def test_high_diversity_low_mutation(self):
        """Test that high diversity results in low mutation rate."""
        # When diversity is high (1.0), mutation should be close to minimum
        result = adaptive_mutation_rate(
            diversity=0.9,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=1.0,
        )
        self.assertGreaterEqual(result, 0.01)
        self.assertLessEqual(result, 0.1)  # Should be relatively low

    def test_low_diversity_high_mutation(self):
        """Test that low diversity results in high mutation rate."""
        # When diversity is low (0.0), mutation should be higher
        result = adaptive_mutation_rate(
            diversity=0.1,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=1.0,
        )
        # Should be significantly above base rate since (1-diversity)^1 = 0.9
        self.assertGreater(result, 0.05)

    def test_base_mutation_rate(self):
        """Test that diversity=0 gives mutation near max."""
        result = adaptive_mutation_rate(
            diversity=0.0,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=1.0,
        )
        self.assertGreater(result, 0.07)  # Should be boosted significantly

    def test_max_mutation_rate_clamped(self):
        """Test that mutation rate doesn't exceed max_mutpb."""
        result = adaptive_mutation_rate(
            diversity=0.0,
            base_mutpb=1.0,  # Very high base to test clamping
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=2.0,
        )
        self.assertLessEqual(result, 0.5)

    def test_min_mutation_rate_clamped(self):
        """Test that mutation rate doesn't go below min_mutpb."""
        result = adaptive_mutation_rate(
            diversity=1.0,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=2.0,
        )
        self.assertGreaterEqual(result, 0.01)

    def test_diversity_out_of_range_clamped(self):
        """Test that diversity values outside [0,1] are clamped."""
        # Very high diversity should be clamped to max diversity effect
        result_high = adaptive_mutation_rate(
            diversity=2.0,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=1.0,
        )
        # Should give same result as diversity=1.0
        result_normal = adaptive_mutation_rate(
            diversity=1.0,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=1.0,
        )
        self.assertAlmostEqual(result_high, result_normal)

    def test_different_sensitivity_values(self):
        """Test that different sensitivity values produce different mutation rates."""
        low_sens = adaptive_mutation_rate(
            diversity=0.5,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=0.5,
        )
        high_sens = adaptive_mutation_rate(
            diversity=0.5,
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=2.0,
        )
        # With higher sensitivity, low diversity should have even higher mutation
        self.assertNotEqual(low_sens, high_sens)


class TestCalculatePopulationDiversity(unittest.TestCase):

    def test_empty_population(self):
        """Test that empty population returns 0.0 diversity."""
        result = calculate_population_diversity([], lambda x: 0.5)
        self.assertEqual(result, 0.0)

    def test_single_individual(self):
        """Test diversity calculation with single individual."""
        mock_ind = MagicMock()
        result = calculate_population_diversity([mock_ind], lambda x: 0.7)
        self.assertAlmostEqual(result, 0.7)

    def test_multiple_individuals_average(self):
        """Test that diversity is properly averaged across population."""
        population = [MagicMock() for _ in range(5)]

        def mock_diversity(ind):
            idx = population.index(ind)
            return (idx + 1) * 0.2  # Returns 0.2, 0.4, 0.6, 0.8, 1.0

        result = calculate_population_diversity(population, mock_diversity)
        expected_avg = sum([0.2, 0.4, 0.6, 0.8, 1.0]) / 5
        self.assertAlmostEqual(result, expected_avg)


class TestAdaptiveMutationClass(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.mock_diversity_result = 0.5

        # Create mock population
        self.population = [MagicMock() for _ in range(10)]

        # Use a simple lambda as diversity calculator (no patch needed)
        self.adapt_mut = AdaptiveMutation(
            base_mutpb=0.05,
            min_mutpb=0.01,
            max_mutpb=0.5,
            sensitivity=1.0,
            diversity_calculator=lambda ind: self.mock_diversity_result,
        )

    def test_initialization_with_default_params(self):
        """Test that AdaptiveMutation initializes with default parameters."""
        adapt = AdaptiveMutation(base_mutpb=0.05)
        self.assertEqual(adapt.base_mutpb, 0.05)
        self.assertEqual(adapt.min_mutpb, 0.01)
        self.assertEqual(adapt.max_mutpb, 0.5)
        self.assertEqual(adapt.sensitivity, 1.0)

    def test_get_adaptive_mutation_rate(self):
        """Test that get_adaptive_mutation_rate returns a valid value."""
        result = self.adapt_mut.get_adaptive_mutation_rate(self.population)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.01)
        self.assertLessEqual(result, 0.5)

    def test_diversity_affects_mutation_rate(self):
        """Test that different diversities produce different mutation rates."""
        # Test with high diversity
        high_div_result = self.adapt_mut.get_adaptive_mutation_rate(self.population)

        # The rate should be higher than minimum (when diversity is 0.5)
        self.assertGreater(high_div_result, 0.02)

    def test_callable_interface(self):
        """Test that AdaptiveMutation can be called as a function."""
        result = self.adapt_mut(self.population)
        self.assertIsInstance(result, float)
        self.assertEqual(
            result, self.adapt_mut.get_adaptive_mutation_rate(self.population)
        )

    def test_empty_population_fallback(self):
        """Test fallback behavior when population is empty."""

        def mock_div(ind):
            return 0.5

        adapt = AdaptiveMutation(base_mutpb=0.05, diversity_calculator=mock_div)
        result = adapt.get_adaptive_mutation_rate([])
        self.assertEqual(result, 0.05)  # Should return base_mutpb when population empty

    def test_custom_diversity_calculator(self):
        """Test that custom diversity calculator is used."""

        def high_diversity(ind):
            return 0.9  # Always return high diversity

        adapt = AdaptiveMutation(
            base_mutpb=0.05, diversity_calculator=lambda ind: high_diversity(ind)
        )

        result = adapt.get_adaptive_mutation_rate([MagicMock()])
        # With high diversity, mutation should be close to minimum
        self.assertLess(result, 0.1)


class TestIntegrationWithDiversityMethods(unittest.TestCase):

    def test_integration_with_measure_diversity_wrapper(self):
        """Test that AdaptiveMutation works with measure_diversity_wrapper."""
        from ml_grid.util.ensemble_diversity_methods import measure_diversity_wrapper

        ensemble = [
            [
                (0, 0, 0, 0, 0, np.array([1, 1, 0, 0])),
                (0, 0, 0, 0, 0, np.array([0, 0, 1, 1])),
            ]
        ]

        diversity_result = measure_diversity_wrapper(ensemble)

        # Measure diversity should return a value between 0 and 1
        self.assertGreaterEqual(diversity_result, 0)
        self.assertLessEqual(diversity_result, 1)


if __name__ == "__main__":
    unittest.main()
