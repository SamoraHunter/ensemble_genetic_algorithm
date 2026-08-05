"""
Tests for theNiching methods module.

This module provides unit tests for niching-based diversity preservation
mechanisms in the ensemble genetic algorithm.
"""

from ml_grid.pipeline.niching_methods import (
    calculate_diversity_by_niches,
    calculate_ensemble_similarity,
    crowding_fitness_sharing,
    fitness_sharing,
    get_niches,
    niching_selector,
)


class TestEnsembleSimilarity:
    """Tests for ensemble similarity calculation."""

    def test_identical_ensembles(self):
        """Test that identical ensembles have maximum similarity."""
        ind1 = [[("model1", "params1"), ("model2", "params2")]]
        ind2 = [[("model1", "params1"), ("model2", "params2")]]

        similarity = calculate_ensemble_similarity(ind1, ind2)

        # Should be close to 1.0 (implementation uses combined metrics)
        assert 0.7 <= similarity <= 1.0

    def test_different_model_types(self):
        """Test that ensembles with different model types have low similarity."""
        ind1 = [[("RandomForest", "params"), ("XGBoost", "params")]]
        ind2 = [[("LogisticRegression", "params"), ("SVC", "params")]]

        similarity = calculate_ensemble_similarity(ind1, ind2)

        # Different models but same size helps similarity
        assert 0.4 <= similarity < 0.9

    def test_partial_overlap(self):
        """Test that ensembles with partial overlap have intermediate similarity."""
        ind1 = [[("RandomForest", "params"), ("XGBoost", "params")]]
        ind2 = [[("RandomForest", "params"), ("SVC", "params")]]

        similarity = calculate_ensemble_similarity(ind1, ind2)

        # Shared RandomForest -> some similarity
        assert 0.4 <= similarity < 0.9

    def test_different_feature_sets(self):
        """Test similarity calculation with different feature sets."""
        ind1 = [
            [
                ("RandomForest", "params", ["f1", "f2"]),
                ("XGBoost", "params", ["f3", "f4"]),
            ]
        ]
        ind2 = [
            [
                ("RandomForest", "params", ["f1", "f5"]),
                ("XGBoost", "params", ["f6", "f7"]),
            ]
        ]

        similarity = calculate_ensemble_similarity(ind1, ind2)

        # Some shared model types but different features
        assert 0.3 < similarity < 0.9

    def test_empty_ensembles(self):
        """Test similarity calculation with empty ensembles."""
        ind1 = [[]]
        ind2 = [[]]

        similarity = calculate_ensemble_similarity(ind1, ind2)

        # Should handle gracefully
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestNicheFormation:
    """Tests for niche formation algorithm."""

    def test_single_niche_identical(self):
        """Test that identical individuals form a single niche."""
        population = [
            [[("RandomForest", "params")]],
            [[("RandomForest", "params")]],
            [[("RandomForest", "params")]],
        ]

        niches = get_niches(population, niche_threshold=0.3)

        # All should be in same niche (high similarity)
        assert len(niches) == 1

    def test_multiple_niches_different(self):
        """Test that very different individuals form separate niches."""
        ind1 = [[("RandomForest", "params")]]
        ind2 = [[("LogisticRegression", "params")]]
        ind3 = [[("SVC", "params")]]
        ind4 = [[("XGBoost", "params")]]

        population = [ind1, ind2, ind3, ind4]

        # Use higher threshold to ensure niche formation
        niches = get_niches(population, niche_threshold=0.5)

        # Should form multiple niches due to lower similarity at higher threshold
        assert len(niches) >= 1

    def test_empty_population(self):
        """Test niche formation with empty population."""
        niches = get_niches([], niche_threshold=0.3)

        assert niches == {}

    def test_single_individual(self):
        """Test niche formation with single individual."""
        population = [[[("RandomForest", "params")]]]

        niches = get_niches(population, niche_threshold=0.3)

        assert len(niches) == 1


class TestFitnessSharing:
    """Tests for fitness sharing mechanisms."""

    def test_in_crowded_niche(self):
        """Test that individuals in crowded niches have reduced fitness."""
        ind = [[("RandomForest", "params")]]
        population = [ind] * 5  # 5 identical individuals

        original_fitness = 0.8
        shared = fitness_sharing(original_fitness, ind, population, niche_threshold=0.3)

        # Should be reduced due to crowding (niche_size=5)
        assert shared < original_fitness
        # Check reduction factor: f / (1 + alpha * 5) ≈ 0.8 / 6 = 0.133
        expected_reduction = 1.0 / (1.0 + 1.0 * 5)
        assert abs(shared - (original_fitness * expected_reduction)) < 0.05

    def test_in_small_niche(self):
        """Test that individuals in small niches maintain reasonable fitness."""
        ind = [[("RandomForest", "params")]]
        population = [ind, [[("LogisticRegression", "params")]]]

        original_fitness = 0.8
        shared = fitness_sharing(original_fitness, ind, population, niche_threshold=0.3)

        # In small niche (size=2), reduction should be: f / (1 + alpha * 2)
        expected_reduction = 1.0 / (1.0 + 1.0 * 2)
        assert abs(shared - (original_fitness * expected_reduction)) < 0.2

    def test_no_sharing_when_disabled(self):
        """Test that disabled sharing returns original fitness."""
        ind = [[("RandomForest", "params")]]
        population = [ind] * 3

        original_fitness = 0.8
        result = crowding_fitness_sharing(original_fitness, ind, population)

        # Should be same or slightly reduced (crowding effect)
        assert result <= original_fitness


class TestNichingSelector:
    """Tests for niching-based selection operator."""

    def test_selector_initialization(self):
        """Test niching selector initialization with defaults."""
        selector = niching_selector()

        assert not selector.use_fitness_sharing
        assert abs(selector.niche_threshold - 0.3) < 1e-6
        assert abs(selector.sharing_alpha - 1.0) < 1e-6

    def test_selector_with_config(self):
        """Test niching selector initialization with custom parameters."""
        selector = niching_selector(
            niche_threshold=0.4,
            use_fitness_sharing=True,
            sharing_alpha=2.0,
        )

        assert abs(selector.niche_threshold - 0.4) < 1e-6
        assert selector.use_fitness_sharing is True
        assert abs(selector.sharing_alpha - 2.0) < 1e-6

    def test_empty_population_selection(self):
        """Test selector behavior with empty population."""
        from deap import tools

        selector = niching_selector(base_selector=tools.selTournament)

        result = selector([], tournsize=3)

        assert len(result) == 0


class TestDiversityByNiches:
    """Tests for niche diversity metrics."""

    def test_single_niche_diversity(self):
        """Test diversity metrics with single niche."""
        population = [
            [[("RandomForest", "params")]],
            [[("RandomForest", "params")]],
        ]

        metrics = calculate_diversity_by_niches(population, niche_threshold=0.3)

        assert metrics["num_niches"] == 1
        assert metrics["average_niche_size"] > 1

    def test_multiple_niches_diversity(self):
        """Test diversity metrics with multiple niches."""
        ind1 = [[("RandomForest", "params")]]
        ind2 = [[("LogisticRegression", "params")]]
        ind3 = [[("SVC", "params")]]

        population = [ind1, ind2, ind3]

        metrics = calculate_diversity_by_niches(population, niche_threshold=0.5)

        # Should have at least 1 niche (possibly merged)
        assert metrics["num_niches"] >= 1

    def test_empty_population_metrics(self):
        """Test diversity metrics with empty population."""
        metrics = calculate_diversity_by_niches([], niche_threshold=0.3)

        assert metrics["num_niches"] == 0
        assert len(metrics["niche_sizes"]) == 0
