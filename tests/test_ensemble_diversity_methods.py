import unittest

import numpy as np

from ml_grid.util.ensemble_diversity_methods import (
    EnsembleDiversityMeasurer,
    apply_diversity_penalty,
)


class TestEnsembleDiversity(unittest.TestCase):

    def setUp(self):
        """Set up sample prediction vectors for testing."""
        self.measurer = EnsembleDiversityMeasurer()

        # Ensemble with two identical prediction vectors
        self.identical_ensemble = [
            [
                (0, 0, 0, 0, 0, np.array([1, 1, 0, 0])),
                (0, 0, 0, 0, 0, np.array([1, 1, 0, 0])),
            ]
        ]

        # Ensemble with two completely different prediction vectors
        self.different_ensemble = [
            [
                (0, 0, 0, 0, 0, np.array([1, 1, 0, 0])),
                (0, 0, 0, 0, 0, np.array([0, 0, 1, 1])),
            ]
        ]

        # Ensemble with partially different prediction vectors
        self.partial_ensemble = [
            [
                (0, 0, 0, 0, 0, np.array([1, 1, 0, 0])),
                (0, 0, 0, 0, 0, np.array([1, 0, 1, 0])),
            ]
        ]

        # Ensemble with a single member
        self.single_member_ensemble = [[(0, 0, 0, 0, 0, np.array([1, 1, 0, 0]))]]

        # Empty ensemble
        self.empty_ensemble = [[]]

        # Ensemble with non-binary predictions (should be handled gracefully)
        self.float_ensemble = [[(0, 0, 0, 0, 0, np.array([0.8, 0.2, 0.6, 0.4]))]]

    def test_jaccard_diversity(self):
        """Test Jaccard diversity calculation."""
        self.assertEqual(
            self.measurer.measure_jaccard_diversity(self.identical_ensemble), 0.0
        )
        self.assertEqual(
            self.measurer.measure_jaccard_diversity(self.different_ensemble), 1.0
        )
        self.assertAlmostEqual(
            self.measurer.measure_jaccard_diversity(self.partial_ensemble), 2 / 3
        )

    def test_hamming_diversity(self):
        """Test Hamming diversity calculation."""
        self.assertEqual(
            self.measurer.measure_hamming_diversity(self.identical_ensemble), 0.0
        )
        self.assertEqual(
            self.measurer.measure_hamming_diversity(self.different_ensemble), 1.0
        )
        self.assertEqual(
            self.measurer.measure_hamming_diversity(self.partial_ensemble), 0.5
        )

    def test_disagreement_diversity(self):
        """Test disagreement diversity calculation."""
        # For identical, variance is 0.
        self.assertEqual(
            self.measurer.measure_disagreement_diversity(self.identical_ensemble), 0.0
        )
        # For different, variance is max (0.25) for each instance. Normalized should be 1.0
        self.assertEqual(
            self.measurer.measure_disagreement_diversity(self.different_ensemble), 1.0
        )
        # For partial, variance is 0 for first and last, 0.25 for middle two. Mean is 0.125. Normalized is 0.5
        self.assertEqual(
            self.measurer.measure_disagreement_diversity(self.partial_ensemble), 0.5
        )

    def test_q_statistic_diversity(self):
        """Test Q-statistic diversity calculation."""
        # For identical, n10 and n01 are 0, Q = 1. Diversity = 1 - abs(1) = 0.
        self.assertEqual(
            self.measurer.measure_q_statistic_diversity(self.identical_ensemble), 0.0
        )
        # For different, n11 and n00 are 0, Q = -1. Diversity = 1 - abs(-1) = 0.
        # This shows a limitation of Q-stat for perfectly anti-correlated.
        self.assertEqual(
            self.measurer.measure_q_statistic_diversity(self.different_ensemble), 0.0
        )
        # For partial: n11=1, n10=1, n01=1, n00=1. Q = (1-1)/(1+1) = 0. Diversity = 1 - 0 = 1.
        self.assertEqual(
            self.measurer.measure_q_statistic_diversity(self.partial_ensemble), 1.0
        )

    def test_kappa_diversity(self):
        """Test Kappa diversity calculation."""
        # For identical, kappa is 1. Diversity = 1 - abs(1) = 0.
        self.assertAlmostEqual(
            self.measurer.measure_kappa_diversity(self.identical_ensemble), 0.0
        )
        # For different, agreement is 0. Expected is 0.5. Kappa = (0-0.5)/(1-0.5) = -1. Diversity = 1 - abs(-1) = 0.
        self.assertAlmostEqual(
            self.measurer.measure_kappa_diversity(self.different_ensemble), 0.0
        )
        # For partial, agreement is 0.5. p1_i=0.5, p1_j=0.5. Expected is 0.5. Kappa = 0. Diversity = 1 - 0 = 1.
        self.assertAlmostEqual(
            self.measurer.measure_kappa_diversity(self.partial_ensemble), 1.0
        )

    def test_apply_diversity_penalty(self):
        """Test the application of diversity penalty on scores."""
        auc, mcc = 0.9, 0.8
        params = {"penalty_method": "linear", "penalty_strength": 0.5}

        # High diversity (metric=1.0) -> similarity=0 -> no penalty
        penalized_auc, penalized_mcc = apply_diversity_penalty(auc, mcc, 1.0, params)
        self.assertEqual(penalized_auc, auc)
        self.assertEqual(penalized_mcc, mcc)

        # No diversity (metric=0.0) -> similarity=1 -> max penalty
        penalized_auc, penalized_mcc = apply_diversity_penalty(auc, mcc, 0.0, params)
        self.assertEqual(penalized_auc, auc * 0.5)
        self.assertEqual(penalized_mcc, mcc * 0.5)

        # Medium diversity (metric=0.5) -> similarity=0.5 -> half penalty
        penalized_auc, penalized_mcc = apply_diversity_penalty(auc, mcc, 0.5, params)
        self.assertEqual(penalized_auc, auc * 0.75)
        self.assertEqual(penalized_mcc, mcc * 0.75)

    def test_apply_diversity_penalty_threshold_method(self):
        """Test threshold penalty method."""
        auc, mcc = 0.9, 0.8

        # High similarity below threshold -> no penalty
        below_threshold_params = {
            "penalty_method": "threshold",
            "penalty_strength": 0.5,
            "similarity_threshold": 0.7,
        }
        penalized_auc, penalized_mcc = apply_diversity_penalty(
            auc, mcc, 1.0, below_threshold_params
        )
        self.assertEqual(penalized_auc, auc)
        self.assertEqual(penalized_mcc, mcc)

        # High similarity above threshold -> penalty applied
        params_above_threshold = {
            "penalty_method": "threshold",
            "penalty_strength": 0.5,
            "similarity_threshold": 0.3,
        }
        penalized_auc, penalized_mcc = apply_diversity_penalty(
            auc, mcc, 0.0, params_above_threshold
        )
        # similarity=1.0, threshold=0.3, excess=0.7, factor=1-0.5*0.7=0.65
        self.assertEqual(penalized_auc, auc * 0.65)
        self.assertEqual(penalized_mcc, mcc * 0.65)

    def test_apply_diversity_penalty_unknown_method(self):
        """Test fallback for unknown penalty method."""
        auc, mcc = 0.9, 0.8
        params = {"penalty_method": "unknown"}
        penalized_auc, penalized_mcc = apply_diversity_penalty(auc, mcc, 0.5, params)
        self.assertEqual(penalized_auc, auc)
        self.assertEqual(penalized_mcc, mcc)

    def test_apply_diversity_penalty_multiple_methods(self):
        """Test quadratic and exponential penalty methods."""
        auc, mcc = 0.9, 0.8

        # Quadratic method
        params_quadratic = {"penalty_method": "quadratic", "penalty_strength": 0.5}
        penalized_auc, penalized_mcc = apply_diversity_penalty(
            auc, mcc, 0.0, params_quadratic
        )
        # similarity=1.0, factor=1-0.5*1^2=0.5
        self.assertEqual(penalized_auc, auc * 0.5)
        self.assertEqual(penalized_mcc, mcc * 0.5)

        # Exponential method
        import numpy as np

        params_exponential = {"penalty_method": "exponential", "penalty_strength": 0.5}
        penalized_auc, penalized_mcc = apply_diversity_penalty(
            auc, mcc, 0.0, params_exponential
        )
        expected_factor = np.exp(-0.5 * 1.0)
        self.assertAlmostEqual(penalized_auc, auc * expected_factor)
        self.assertAlmostEqual(penalized_mcc, mcc * expected_factor)

    def test_measure_diversity_wrapper(self):
        """Test the measure_diversity_wrapper function."""
        from ml_grid.util.ensemble_diversity_methods import (
            measure_diversity_wrapper,
        )

        result = measure_diversity_wrapper(self.partial_ensemble, method="jaccard")
        self.assertIsInstance(result, float)

    def test_measure_diversity_unknown_method(self):
        """Test EnsembleDiversityMeasurer with unknown method raises ValueError."""
        measurer = EnsembleDiversityMeasurer(method="unknown_method")
        with self.assertRaises(ValueError) as context:
            measurer.measure_binary_vector_diversity(self.partial_ensemble)
        self.assertIn("Unknown method", str(context.exception))

    def test_measure_diversity_comprehensive_method(self):
        """Test comprehensive method weight combination."""
        from ml_grid.util.ensemble_diversity_methods import (
            EnsembleDiversityMeasurer,
        )

        # Test with custom weights
        custom_weights = [0.4, 0.3, 0.2, 0.1]
        measurer = EnsembleDiversityMeasurer(
            method="comprehensive", weights=custom_weights
        )
        self.assertEqual(measurer.weights, custom_weights)

    def test_apply_diversity_penalty_custom_params(self):
        """Test apply_diversity_penalty with custom min_score_factor."""
        auc, mcc = 0.9, 0.8
        params = {
            "penalty_method": "linear",
            "penalty_strength": 0.5,
            "min_score_factor": 0.2,
        }
        # When diversity=0, similarity=1, factor would be 0.5 but min is 0.2
        penalized_auc, penalized_mcc = apply_diversity_penalty(auc, mcc, 0.0, params)
        self.assertEqual(penalized_auc, auc * 0.5)
        self.assertEqual(penalized_mcc, mcc * 0.5)

    def test_apply_diversity_penalty_low_min_score_factor(self):
        """Test that min_score_factor can be exceeded with higher penalty."""
        auc, mcc = 0.9, 0.8
        params = {
            "penalty_method": "linear",
            "penalty_strength": 3.0,
            "min_score_factor": 0.1,
        }
        # With strength=3 and similarity=1, factor would be 1-3*1=-2 but min is 0.1
        penalized_auc, penalized_mcc = apply_diversity_penalty(auc, mcc, 0.0, params)
        self.assertEqual(penalized_auc, auc * 0.1)
        self.assertEqual(penalized_mcc, mcc * 0.1)

    def test_measure_diversity_wrapper_comprehensive(self):
        """Test measure_diversity_wrapper with comprehensive method."""
        from ml_grid.util.ensemble_diversity_methods import (
            measure_diversity_wrapper,
        )

        result = measure_diversity_wrapper(
            self.partial_ensemble, method="comprehensive"
        )
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_measure_diversity_all_methods(self):
        """Test all diversity measurement methods."""
        from ml_grid.util.ensemble_diversity_methods import (
            EnsembleDiversityMeasurer,
        )

        methods = ["jaccard", "hamming", "disagreement", "q_statistic", "kappa"]
        for method in methods:
            measurer = EnsembleDiversityMeasurer(method=method)
            result = measurer.measure_binary_vector_diversity(self.partial_ensemble)
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)

    def test_diversity_with_single_member(self):
        """Test that diversity is 0 for an ensemble with one member."""
        self.assertEqual(
            self.measurer.measure_jaccard_diversity(self.single_member_ensemble), 0.0
        )
        self.assertEqual(
            self.measurer.measure_hamming_diversity(self.single_member_ensemble), 0.0
        )
        self.assertEqual(
            self.measurer.measure_disagreement_diversity(self.single_member_ensemble),
            0.0,
        )

    def test_diversity_with_empty_ensemble(self):
        """Test that diversity is 0 for an empty ensemble."""
        self.assertEqual(
            self.measurer.measure_jaccard_diversity(self.empty_ensemble), 0.0
        )
        self.assertEqual(
            self.measurer.measure_hamming_diversity(self.empty_ensemble), 0.0
        )
        self.assertEqual(
            self.measurer.measure_disagreement_diversity(self.empty_ensemble), 0.0
        )


if __name__ == "__main__":
    unittest.main()
