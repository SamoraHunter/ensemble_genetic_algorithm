import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_selection
from PyImpetus import PPIMBC
from sklearn.svm import SVC
from typing import List, Union


class feature_methods:
    def __init__(self):
        """Initializes the feature_methods class.

        This class provides different algorithms for feature selection.
        """
        pass

    def getNfeaturesANOVAF(
        self, n: int, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[str]:
        """Gets the top n features based on the ANOVA F-value.

        This method calculates the ANOVA F-value for each feature in `X_train`
        against the target `y_train`. It then returns the names of the top `n`
        features with the highest F-values.

        Note:
            The current implementation calculates the F-value for each column
            individually in a loop, which can be inefficient for datasets with
            many features. A more performant approach would be to call
            `sklearn.feature_selection.f_classif(X_train, y_train)` once.

        Args:
            n: The number of top features to return.
            X_train: The training features DataFrame.
            y_train: The training target Series.

        Returns:
            A list of column names for the top `n` features.

        Raises:
            ValueError: If `X_train` is not a pandas DataFrame or numpy array.
        """
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
            X_train_np = X_train.values
        elif isinstance(X_train, np.ndarray):
            feature_names = np.arange(X_train.shape[1])
            X_train_np = X_train
        else:
            raise ValueError("X_train must be a pandas DataFrame or numpy array")

        res = []
        for i, col in enumerate(X_train_np.T):
            res.append(
                (
                    feature_names[i],
                    sklearn.feature_selection.f_classif(col.reshape(-1, 1), y_train)[0],
                )
            )

        sortedList = sorted(res, key=lambda X: X[1], reverse=True)
        print(sortedList)
        nFeatures = sortedList[:n]
        finalColNames = [elem[0] for elem in nFeatures]

        return finalColNames

    def getNFeaturesMarkovBlanket(
        self,
        n: int,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> List[str]:
        """Gets top n features from the Markov Blanket using PyImpetus.

        Args:
            n: The number of top features to retrieve.
            X_train: The training input samples.
            y_train: The target values.

        Returns:
            A list containing the names of the top `n` features from the
            Markov Blanket.
        """
        model = PPIMBC(
            model=SVC(random_state=27, class_weight="balanced"),
            p_val_thresh=0.05,
            num_simul=30,
            simul_size=0.2,
            simul_type=0,
            sig_test_type="non-parametric",
            cv=5,
            random_state=27,
            n_jobs=-1,
            verbose=2,
        )

        df_train_transformed = model.fit_transform(X_train, y_train)

        feature_names = model.MB[:n]

        return feature_names
