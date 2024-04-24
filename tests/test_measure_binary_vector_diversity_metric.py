import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from ml_grid.pipeline.evaluate_methods_ga import measure_binary_vector_diversity
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import scipy.spatial.distance


class TestBinaryVectorDiversity(unittest.TestCase):
    def setUp(self):
        # Mock DecisionTreeClassifier and LogisticRegression
        self.mock_decision_tree = MagicMock()
        self.mock_logistic_regression = MagicMock()
        self.mock_decision_tree.return_value.predict.return_value = np.zeros(
            20
        )  # Assuming 20 predictions for simplicity
        self.mock_logistic_regression.return_value.predict.return_value = np.ones(
            20
        )  # Assuming 20 predictions for simplicity

    @patch.object(DecisionTreeClassifier, "__init__", return_value=None)
    @patch.object(LogisticRegression, "__init__", return_value=None)
    def test_measure_binary_vector_diversity(
        self, mock_logistic_regression_init, mock_decision_tree_init
    ):
        mock_decision_tree_init.return_value = None

        ensemble = [
            [
                (
                    0.0,
                    self.mock_decision_tree,
                    ["feature_1", "feature_2"],
                    0,
                    0.5,
                    np.zeros(20),
                ),
                (
                    0.0,
                    self.mock_logistic_regression,
                    ["feature_1", "feature_2", "feature_3"],
                    0,
                    0.5,
                    np.ones(20),
                ),
            ]
        ]

        # Manually compute Jaccard distance
        vec1 = np.zeros(20)
        vec2 = np.ones(20)
        intersection = np.sum(vec1 * vec2)
        union = np.sum(vec1 + vec2) - intersection
        expected_distance = 1.0 - (intersection / union)

        calculated_distance = measure_binary_vector_diversity(
            ensemble, metric="jaccard"
        )
        print("Expected Distance:", expected_distance)
        print("Calculated Distance:", calculated_distance)
        print(
            "Decision Tree Predictions:",
            self.mock_decision_tree.return_value.predict.return_value,
        )
        print(
            "Logistic Regression Predictions:",
            self.mock_logistic_regression.return_value.predict.return_value,
        )
        self.assertAlmostEqual(
            calculated_distance, expected_distance, places=5
        )  # Check with a tolerance


if __name__ == "__main__":
    unittest.main()
