"""
Niching methods for diversity preservation in ensemble genetic algorithms.

This module implements niche formation and fitness sharing mechanisms to maintain
genetically diverse solutions in the evolving population. These techniques encourage
the exploration of different regions of the search space by creating subpopulations
(niches) based on similarity between individuals.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ensemble_ga")


def calculate_ensemble_similarity(ind1: Any, ind2: Any) -> float:
    """Calculate similarity between two ensemble individuals.

    Uses a combined metric based on:
    1. Base learner type overlap (Jaccard index)
    2. Feature set overlap
    3. Hyperparameter similarity

    Args:
        ind1: First individual (ensemble)
        ind2: Second individual (ensemble)

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    try:
        # Extract base learners from ensemble structure
        # Ensemble format: [[(model_class, model_params)], ...]
        learners1 = ind1[0] if len(ind1) > 0 else []
        learners2 = ind2[0] if len(ind2) > 0 else []

        if not learners1 or not learners2:
            return 0.5

        # Calculate base learner type overlap
        types1 = set()
        types2 = set()

        for learner in learners1:
            if isinstance(learner, (list, tuple)) and len(learner) >= 2:
                types1.add(str(learner[1]))

        for learner in learners2:
            if isinstance(learner, (list, tuple)) and len(learner) >= 2:
                types2.add(str(learner[1]))

        # Jaccard similarity for model types
        if types1 and types2:
            type_overlap = len(types1 & types2) / len(types1 | types2)
        else:
            type_overlap = 0.5

        # Calculate feature set overlap (if available)
        features1 = []
        features2 = []

        for learner in learners1:
            if isinstance(learner, (list, tuple)) and len(learner) >= 3:
                feat = learner[2]
                if isinstance(feat, list):
                    features1.extend(feat)

        for learner in learners2:
            if isinstance(learner, (list, tuple)) and len(learner) >= 3:
                feat = learner[2]
                if isinstance(feat, list):
                    features2.extend(feat)

        # Jaccard similarity for feature sets
        set1 = set(features1)
        set2 = set(features2)

        if set1 and set2:
            feature_overlap = len(set1 & set2) / len(set1 | set2)
        else:
            feature_overlap = 0.5

        # Calculate size similarity (normalized)
        size1 = len(learners1)
        size2 = len(learners2)

        if max(size1, size2) > 0:
            size_similarity = min(size1, size2) / max(size1, size2)
        else:
            size_similarity = 1.0

        # Combined similarity (weighted average)
        combined = 0.4 * type_overlap + 0.4 * feature_overlap + 0.2 * size_similarity

        return min(1.0, max(0.0, combined))

    except Exception:
        logger.debug("Error calculating ensemble similarity")
        return 0.5


def get_niches(
    population: List,
    niche_threshold: float = 0.3,
    similarity_func: Optional[Callable] = None,
) -> Dict[int, List]:
    """Assign individuals to niches based on similarity.

    Uses a greedy clustering algorithm:
    1. Start with first individual as new niche
    2. For each remaining individual, find closest existing niche
    3. If closest niche > threshold dissimilarity, create new niche

    Args:
        population: List of individual objects
        niche_threshold: Dissimilarity threshold for niche formation (0-1)
        similarity_func: Function to calculate similarity between individuals

    Returns:
        Dictionary mapping niche_id -> list of individuals in that niche
    """
    if not population:
        return {}

    if similarity_func is None:
        similarity_func = calculate_ensemble_similarity

    # Initialize niches dictionary
    niches: Dict[int, List] = {}

    # Assign first individual to niche 0
    niches[0] = [population[0]]

    # Assign remaining individuals
    for ind in population[1:]:
        best_niche = None
        best_dissimilarity = 0.0

        # Find the niche with highest similarity
        for niche_id, niche_members in niches.items():
            # Calculate average similarity to niche members
            similarities = [similarity_func(ind, member) for member in niche_members]
            avg_similarity = np.mean(similarities)

            dissimilarity = 1.0 - avg_similarity

            if dissimilarity > best_dissimilarity:
                best_dissimilarity = dissimilarity
                best_niche = niche_id

        # Create new niche or add to existing
        if best_niche is None or best_dissimilarity >= niche_threshold:
            # Create new niche
            new_niche_id = max(niches.keys()) + 1 if niches else 0
            niches[new_niche_id] = [ind]
        else:
            # Add to existing niche
            niches[best_niche].append(ind)

    return niches


def calculate_niche_size(population: List, niche_threshold: float = 0.3) -> int:
    """Estimate the number of niches in a population.

    Args:
        population: List of individual objects
        niche_threshold: Dissimilarity threshold for niche formation

    Returns:
        Number of niches identified
    """
    niches = get_niches(population, niche_threshold)
    return len(niches)


def fitness_sharing(
    fitness: float,
    individual: Any,
    population: List,
    niche_threshold: float = 0.3,
    sharing_alpha: float = 1.0,
) -> float:
    """Apply fitness sharing to reduce fitness in crowded niches.

    Fitness sharing reduces the effective fitness of individuals based on
    how many similar individuals are in their niche. This encourages diversity
    by making rare niches more attractive.

    The shared fitness is calculated as:
        f_shared = f_original / (1 + alpha * count_in_niche)

    Args:
        fitness: Original fitness value
        individual: The individual to apply sharing to
        population: Current population for niche calculation
        niche_threshold: Dissimilarity threshold for niche formation
        sharing_alpha: Sharing intensity parameter (>0, higher = stronger sharing)

    Returns:
        Shared fitness value (reduced if in crowded niche)
    """
    niches = get_niches(population, niche_threshold)

    # Find individual's niche
    for niche_id, members in niches.items():
        for member in members:
            if member is individual:
                niche_size = len(members)
                sharing_factor = 1.0 + (sharing_alpha * niche_size)
                return fitness / sharing_factor

    return fitness


def crowding_fitness_sharing(
    fitness: float,
    individual: Any,
    population: List,
    niche_threshold: float = 0.3,
) -> float:
    """Apply crowding-based fitness sharing.

    Uses a quadratic sharing function that drops to zero beyond the niche radius.
    This creates clear niches with sharp boundaries.

    The shared fitness is:
        f_shared = f_original * (1 - d/r)^alpha if d < r
        f_shared = 0 otherwise

    where d is average distance to niche members, r is niche radius

    Args:
        fitness: Original fitness value
        individual: The individual to apply sharing to
        population: Current population for niche calculation
        niche_threshold: Dissimilarity threshold (acts as niche radius)

    Returns:
        Crowdedfitness value
    """
    try:
        niches = get_niches(population, niche_threshold)
        niche_radius = 1.0 - niche_threshold  # Convert dissim to similarity

        for niche_id, members in niches.items():
            if individual in members:
                # Calculate average similarity within niche
                similarities = [
                    calculate_ensemble_similarity(individual, member)
                    for member in members
                ]
                avg_similarity = np.mean(similarities)

                # If within niche radius, apply quadratic sharing
                if avg_similarity >= niche_radius:
                    relative_dist = (avg_similarity - niche_radius) / (
                        1.0 - niche_radius + 1e-10
                    )
                    sharing_factor = max(0.0, 1.0 - relative_dist**2)
                    return fitness * sharing_factor

                return fitness

        return fitness

    except Exception:
        logger.debug("Error in crowding fitness sharing")
        return fitness


class niching_selector:
    """DEAP-compatible selection operator with niching support.

    Applies niche-based selection pressure to maintain population diversity.
    Combines tournament selection with niche formation to prevent dominance
    of similar individuals.

    Args:
        base_selector: Base DEAP selector (e.g., tools.selTournament)
        niche_threshold: Dissimilarity threshold for niche formation
        use_fitness_sharing: Whether to apply fitness sharing
        sharing_alpha: Alpha parameter for fitness sharing
    """

    def __init__(
        self,
        base_selector: Optional[Callable] = None,
        niche_threshold: float = 0.3,
        use_fitness_sharing: bool = False,
        sharing_alpha: float = 1.0,
    ):
        """Initialize niching selector."""
        self.niche_threshold = niche_threshold
        self.use_fitness_sharing = use_fitness_sharing
        self.sharing_alpha = sharing_alpha

        # Default to tournament selection with size 3 if not specified
        if base_selector is None:
            from deap import tools

            def default_tournament(pop, **kwargs):
                return tools.selTournament(pop, tournsize=3, **kwargs)

            self.base_selector = default_tournament
        else:
            self.base_selector = base_selector

    def __call__(self, population: List, *args, **kwargs) -> List:
        """Apply niching selection to the population.

        Args:
            population: Current population of individuals
            *args: Additional args for base selector
            **kwargs: Additional kwargs for base selector

        Returns:
            Selected individuals for next generation
        """
        if not population:
            return []

        # Apply fitness sharing if enabled
        if self.use_fitness_sharing and len(population) > 1:
            for ind in population:
                original_fitness = ind.fitness.values[0]
                shared_fitness = fitness_sharing(
                    original_fitness,
                    ind,
                    population,
                    self.niche_threshold,
                    self.sharing_alpha,
                )
                # Update first fitness value
                ind.fitness.values = (shared_fitness,) + ind.fitness.values[1:]

        # Apply base selector
        selected = self.base_selector(population, *args, **kwargs)

        return selected


def niching_from_config(config_dict: Dict) -> niching_selector:
    """Create a niching selector based on configuration.

    Args:
        config_dict: Dictionary containing niching configuration with keys:
            - 'niching_enabled': bool
            - 'niche_threshold': float (dissimilarity threshold)
            - 'use_fitness_sharing': bool
            - 'sharing_alpha': float
            - 'niching_method': str ('formation' or 'sharing')

    Returns:
        Configured niching_selector instance
    """
    niche_config = config_dict.get("niche_params", {})

    if not niche_config.get("enabled", False):
        return niching_selector(base_selector=None)

    niche_threshold = niche_config.get("niche_threshold", 0.3)
    use_fitness_sharing = niche_config.get("use_fitness_sharing", False)
    sharing_alpha = niche_config.get("sharing_alpha", 1.0)

    return niching_selector(
        niche_threshold=niche_threshold,
        use_fitness_sharing=use_fitness_sharing,
        sharing_alpha=sharing_alpha,
    )


def calculate_diversity_by_niches(
    population: List, niche_threshold: float = 0.3
) -> Dict:
    """Calculate diversity metrics across niches.

    Provides insights into how population diversity is distributed
    across different niches.

    Args:
        population: Current population
        niche_threshold: Dissimilarity threshold for niche formation

    Returns:
        Dictionary with niche statistics including:
            - num_niches
            - niche_sizes
            - average_niche_size
            - niche_diversity_scores
    """
    niches = get_niches(population, niche_threshold)

    if not niches:
        return {
            "num_niches": 0,
            "niche_sizes": [],
            "average_niche_size": 0.0,
            "niche_diversity_scores": {},
        }

    # Calculate niche sizes
    niche_sizes = [len(members) for members in niches.values()]
    avg_niche_size = np.mean(niche_sizes)

    # Calculate diversity within each niche

    niche_diversity_scores = {}

    for niche_id, members in niches.items():
        if len(members) > 1:
            # Use average pairwise diversity
            diversities = []
            for i, ind1 in enumerate(members):
                for ind2 in members[i + 1 :]:
                    div = calculate_ensemble_similarity(ind1, ind2)
                    diversities.append(div)

            niche_diversity_scores[niche_id] = {
                "size": len(members),
                "avg_pairwise_sim": np.mean(diversities) if diversities else 0.0,
                "min_pairwise_sim": min(diversities) if diversities else 0.0,
                "max_pairwise_sim": max(diversities) if diversities else 1.0,
            }
        else:
            niche_diversity_scores[niche_id] = {
                "size": 1,
                "avg_pairwise_sim": 1.0,
                "min_pairwise_sim": 1.0,
                "max_pairwise_sim": 1.0,
            }

    return {
        "num_niches": len(niches),
        "niche_sizes": niche_sizes,
        "average_niche_size": avg_niche_size,
        "niche_diversity_scores": niche_diversity_scores,
    }


def niching_analysis(population: List, niche_threshold: float = 0.3) -> str:
    """Generate a human-readable analysis of populationNiching.

    Args:
        population: Current population
        niche_threshold: Dissimilarity threshold

    Returns:
        Formatted string with niching analysis
    """
    metrics = calculate_diversity_by_niches(population, niche_threshold)

    lines = [
        "\n=== Population Niching Analysis ===",
        f"Number of niches: {metrics['num_niches']}",
        f"Niche sizes: {metrics['niche_sizes']}",
        f"Avg niche size: {metrics['average_niche_size']:.2f}",
    ]

    # Check for potential issues
    if metrics["num_niches"] == 1:
        lines.append(
            "⚠ WARNING: Population is not分化 - all individuals in single niche"
        )

    avg_niche_size = metrics.get("average_niche_size", 0)
    if avg_niche_size > len(population) / max(1, metrics["num_niches"]):
        lines.append(
            f"⚠ Some niches are large (> {avg_niche_size:.1f} individuals), may indicate insufficient diversity"
        )

    return "\n".join(lines)
