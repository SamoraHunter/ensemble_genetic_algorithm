import logging
from typing import Any, Tuple

import pandas as pd

from ml_grid.pipeline.data_feature_methods import feature_methods

# rename this class


class feature_importance_methods:
    """A class to handle feature selection using different importance methods.

    This class acts as a wrapper to call various feature selection algorithms
    defined in `feature_methods`. It selects features based on the method
    specified in the `ml_grid_object`'s configuration.
    """

    logger = logging.getLogger("ensemble_ga")

    def __init__(self):
        """Initializes the feature_importance_methods class."""

    def handle_feature_importance_methods(
        self,
        target_n_features: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        X_test_orig: pd.DataFrame,
        ml_grid_object: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Selects top features and reduces the DataFrames to those features.

        This method applies a feature selection algorithm (e.g., ANOVA F-test or
        Markov Blanket) to identify the most important features from the training
        data. It then filters `X_train`, `X_test`, and `X_test_orig` to include
        only these selected features.

        Args:
            target_n_features: The number of top features to select.
            X_train: The training features DataFrame.
            X_test: The testing features DataFrame.
            y_train: The training target Series.
            X_test_orig: The original (validation) testing features DataFrame.
            ml_grid_object: An object containing the configuration, including
                the `feature_selection_method` to use.

        Returns:
            A tuple containing the three input DataFrames (`X_train`, `X_test`,
            `X_test_orig`) filtered to include only the selected features.
        """

        feature_method = ml_grid_object.local_param_dict.get("feature_selection_method")

        if feature_method == "anova" or feature_method == None:

            feature_importance_methods.logger.info("feature_method ANOVA")

            features = feature_methods.getNfeaturesANOVAF(
                self, n=target_n_features, X_train=X_train, y_train=y_train
            )

        elif feature_method == "markov_blanket":

            feature_importance_methods.logger.info("feature method Markov")

            features = feature_methods.getNFeaturesMarkovBlanket(
                self, n=target_n_features, X_train=X_train, y_train=y_train
            )

        feature_importance_methods.logger.info(
            "target_n_features: %s", target_n_features
        )

        X_train = X_train[features]

        X_test = X_test[features]

        X_test_orig = X_test_orig[features]

        return X_train, X_test, X_test_orig
