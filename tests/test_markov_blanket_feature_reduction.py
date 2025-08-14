import unittest
from ml_grid.pipeline.data_feature_methods import (
    feature_methods,
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class TestGetNFeaturesMarkovBlanket(unittest.TestCase):
    def test_number_of_features(self):
        # Generate synthetic data for binary classification
        X, y = make_classification(
            n_samples=150, n_features=10, n_informative=3, n_classes=2, random_state=42
        )

        # Split the data into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=42)

        # Create an instance of MyClass
        my_instance = feature_methods()

        # Call the function to get the top 3 features
        top_features = my_instance.getNFeaturesMarkovBlanket(3, X_train, y_train)

        # Assert that the number of features returned is 3
        self.assertEqual(len(top_features), 3)


if __name__ == "__main__":
    unittest.main()
