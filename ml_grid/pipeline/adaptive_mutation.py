"""
Adaptive mutation rate based on population diversity.

This module provides functions for dynamically adjusting mutation rates
based on current population diversity to prevent premature convergence
and maintain genetic diversity throughout evolution.
"""

import logging
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger("ensemble_ga")


def calculate_population_diversity(
    population: List, diversity_calculator: Callable
) -> float:
    """Calculate average diversity across all individuals in the population.

    Args:
        population: A list of DEAP individual objects.
        diversity_calculator: A callable that takes an individual and returns a diversity score.

    Returns:
        The average diversity score across all individuals in the population.
    """
    if not population:
        return 0.0

    diversity_scores = [diversity_calculator(ind) for ind in population]
    return np.mean(diversity_scores)


def adaptive_mutation_rate(
    diversity: float,
    base_mutpb: float,
    min_mutpb: float = 0.01,
    max_mutpb: float = 0.5,
    sensitivity: float = 1.0,
) -> float:
    """Calculate an adaptive mutation rate based on population diversity.

    Uses an inverse relationship between diversity and mutation rate:
    - High diversity → low mutation rate (stabilize good solutions)
    - Low diversity → high mutation rate (introduce variety)

    Args:
        diversity: Current population diversity score (0=identical, 1=max diverse).
        base_mutpb: Base mutation probability from config for scaling.
        min_mutpb: Minimum allowed mutation rate (default: 0.01).
        max_mutpb: Maximum allowed mutation rate (default: 0.5).
        sensitivity: Controls the sharpness of the transition (default: 1.0).

    Returns:
        Adaptive mutation rate clipped to valid range [min_mutpb, max_mutpb].
    """
    # Ensure diversity is in valid range
    diversity = max(0.0, min(1.0, float(diversity)))

    # Calculate adaptive factor based on diversity
    # High diversity (near 1.0) → low mutation (factor near 0)
    # Low diversity (near 0.0) → high mutation (factor near 1)
    adaptive_factor = (1 - diversity) ** sensitivity

    # Scale by base mutation rate
    adjusted_mutpb = base_mutpb * (1 + adaptive_factor)

    # Clip to valid range
    return max(min_mutpb, min(max_mutpb, adjusted_mutpb))


class AdaptiveMutation:
    """Class for managing adaptive mutation rates based on population diversity.

    This class encapsulates the logic for dynamically adjusting mutation rates
    throughout the evolutionary process. It provides both the core calculation
    and a callable interface for integration with DEAP's evolution loop.
    """

    base_mutpb: float
    min_mutpb: float
    max_mutpb: float
    sensitivity: float
    diversity_calculator: Callable

    def __init__(
        self,
        base_mutpb: float,
        min_mutpb: float = 0.01,
        max_mutpb: float = 0.5,
        sensitivity: float = 1.0,
        diversity_calculator: Optional[Callable] = None,
    ):
        """Initialize the adaptive mutation controller.

        Args:
            base_mutpb: Base mutation probability from configuration.
            min_mutpb: Minimum mutation rate (default: 0.01).
            max_mutpb: Maximum mutation rate (default: 0.5).
            sensitivity: Sensitivity parameter for diversity mapping (default: 1.0).
                Higher values make mutation more responsive to low diversity.
            diversity_calculator: Optional callable that takes an individual
                and returns a diversity score. If None, uses measure_diversity_wrapper.
        """
        self.base_mutpb = base_mutpb
        self.min_mutpb = min_mutpb
        self.max_mutpb = max_mutpb
        self.sensitivity = sensitivity

        if diversity_calculator is not None:
            self.diversity_calculator = diversity_calculator
        else:
            from ml_grid.util.ensemble_diversity_methods import (
                measure_diversity_wrapper,
            )

            self.diversity_calculator = lambda ind: measure_diversity_wrapper(ind)

    def get_adaptive_mutation_rate(self, population: List) -> float:
        """Calculate and return the adaptive mutation rate for current population.

        Args:
            population: Current list of individual objects in the population.

        Returns:
            The calculated adaptive mutation rate.
        """
        if not population:
            logger.warning("Population is empty, using base mutation rate")
            return self.base_mutpb

        # Calculate average diversity across population
        avg_diversity = calculate_population_diversity(
            population, self.diversity_calculator
        )

        # Log diversity for debugging/tuning
        logger.debug(f"Population diversity: {avg_diversity:.4f}")

        # Calculate adaptive mutation rate
        mutpb = adaptive_mutation_rate(
            diversity=avg_diversity,
            base_mutpb=self.base_mutpb,
            min_mutpb=self.min_mutpb,
            max_mutpb=self.max_mutpb,
            sensitivity=self.sensitivity,
        )

        logger.debug(f"Adaptive mutation rate: {mutpb:.4f}")

        return mutpb

    def __call__(self, population: List) -> float:
        """Make class callable for use as a function.

        Args:
            population: Current population of individuals.

        Returns:
            The calculated adaptive mutation rate.
        """
        return self.get_adaptive_mutation_rate(population)
