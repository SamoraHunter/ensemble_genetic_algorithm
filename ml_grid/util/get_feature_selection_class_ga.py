import random
from operator import itemgetter
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier


class feature_selection_methods_class:
    """A class for applying various feature selection methods.

    This class holds different feature selection algorithms (ANOVA F-test,
    Random Forest, XGBoost, Extra Trees) and applies one of them to reduce
    the dimensionality of the training and testing data.

    Attributes:
        X_train (pd.DataFrame): The training features DataFrame.
        y_train (pd.Series): The training target Series.
        X_test (pd.DataFrame): The testing features DataFrame.
        feature_parameter_vector (np.ndarray): An array of possible numbers
            of features to select, ranging from 2 to the total number of
            initial features.
    """

    def __init__(self, ml_grid_object: Any):
        """Initializes the feature_selection_methods_class.

        Args:
            ml_grid_object (Any): An object containing the data splits
                (X_train, y_train, X_test).
        """

        self.X_train = ml_grid_object.X_train

        self.y_train = ml_grid_object.y_train

        self.X_test = ml_grid_object.X_test

        start_val = 2

        self.feature_parameter_vector = np.linspace(
            2, len(self.X_train.columns), int(len(self.X_train.columns) + 1 - start_val)
        ).astype(int)

    def getNfeaturesANOVAF(self, n: int) -> List[str]:
        """Selects the top n features using the ANOVA F-test.

        Note:
            The current implementation calculates the F-value for each column
            individually in a loop, which can be inefficient. A more performant
            approach would be to call `f_classif(X_train, y_train)` once for
            all features.

        Args:
            n: The number of top features to select.

        Returns:
            A list of the names of the top n features.
        """
        res = []
        for colName in self.X_train.columns:
            if colName != "intercept":
                # Compute F-score
                f_score = f_classif(
                    np.array(self.X_train[colName]).reshape(-1, 1), self.y_train
                )[0][0]

                # Skip if F-score is NaN
                if not np.isnan(f_score):
                    res.append((colName, f_score))

        # Sort by F-score descending
        sortedList = sorted(res, key=lambda x: x[1], reverse=True)

        # Take top n
        nFeatures = sortedList[:n]

        # Extract column names
        finalColNames = [elem[0] for elem in nFeatures]

        return finalColNames

    def getRandomForestFeatureColumns(
        self, X: pd.DataFrame, y: pd.Series, n: int
    ) -> List[str]:
        """Selects the top n features using Random Forest feature importances.

        Note:
            This method has a side effect of re-assigning `self.X_train` and
            `self.y_train`. This is not ideal and could lead to unexpected
            behavior if the class is used in other contexts.

        Args:
            X: The training features DataFrame.
            y: The training target Series.
            n: The number of top features to select.

        Returns:
            A list of the names of the top n features.
        """
        try:
            X = X.drop("index", axis=1)
        except:
            pass
        independentVariables = X
        self.X_train = X
        self.y_train = y
        forest = RandomForestClassifier(random_state=1)
        forest.fit(self.X_train, self.y_train)
        importances = forest.feature_importances_
        namedFeatures = []
        for i in range(0, len(self.X_train.columns)):
            namedFeatures.append((list(self.X_train.columns)[i], importances[i]))
        sortedNamedFeatures = sorted(namedFeatures, key=itemgetter(1))
        sortedNamedFeatures.reverse()
        nFeatures = sortedNamedFeatures[:n]
        finalColNames = []
        for elem in nFeatures:
            finalColNames.append(elem[0])
        return finalColNames

    def getXGBoostFeatureColumns(
        self, X: pd.DataFrame, y: pd.Series, n: int
    ) -> List[str]:
        """Selects the top n features using XGBoost feature importances.

        Args:
            X: The training features DataFrame.
            y: The training target Series.
            n: The number of top features to select.

        Returns:
            A list of the names of the top n features.
        """
        try:
            X = X.drop("index", axis=1)
        except:
            pass
        try:
            model = XGBClassifier()
            model.fit(X, y)
            importances = model.feature_importances_
            namedFeatures = []
            for i in range(0, len(self.X_train.columns)):
                namedFeatures.append((list(self.X_train.columns)[i], importances[i]))
            sortedNamedFeatures = sorted(namedFeatures, key=itemgetter(1))
            sortedNamedFeatures.reverse()
            nFeatures = sortedNamedFeatures[:n]
            finalColNames = []
            for elem in nFeatures:
                finalColNames.append(elem[0])
        except Exception as e:
            print(e)
            print("Failed to get xgboost feature columns")
            raise e

        return finalColNames

    def getExtraTreesFeatureColumns(
        self, X: pd.DataFrame, y: pd.Series, n: int
    ) -> List[str]:
        """Selects the top n features using ExtraTreesClassifier feature importances.

        Note:
            This method has a side effect of re-assigning `self.X_train` and
            `self.y_train`. This is not ideal and could lead to unexpected
            behavior if the class is used in other contexts.

        Args:
            X: The training features DataFrame.
            y: The training target Series.
            n: The number of top features to select.

        Returns:
            A list of the names of the top n features.
        """
        independentVariables = X

        self.X_train = X
        self.y_train = y
        try:
            X = X.drop("index", axis=1)
        except:
            X = X
        forest = ExtraTreesClassifier(random_state=1)
        forest.fit(self.X_train, self.y_train)
        importances = forest.feature_importances_
        namedFeatures = []
        for i in range(0, len(self.X_train.columns)):
            namedFeatures.append((list(self.X_train.columns)[i], importances[i]))
        sortedNamedFeatures = sorted(namedFeatures, key=itemgetter(1))
        sortedNamedFeatures.reverse()

        #     xFeatureColNames = []
        #     for i in range(0, n):
        #         xFeatureColNames.append(sortedNamedFeatures[i][0])
        nFeatures = sortedNamedFeatures[:n]
        finalColNames = []
        for elem in nFeatures:
            finalColNames.append(elem[0])
        return finalColNames


    # # Base learner and ensemble generation  
            




    

    def get_featured_selected_training_data(self, method='anova'):
        #global self.X_train, X_test, self.y_train, y_test
            # Select n features------------------------------------------------------------------------
        f = self.feature_parameter_vector 
        
        nFeatures = random.choice(f)
        
        if(method=='anova'):
            xFeatureColumnNames = feature_selection_methods_class.getNfeaturesANOVAF(self, 
                nFeatures)
        
        elif(method=='randomforest'):
            xFeatureColumnNames = feature_selection_methods_class.getRandomForestFeatureColumns(self, 
            self.X_train, self.y_train, nFeatures
        )
        elif(method=='extratrees'):
            xFeatureColumnNames = feature_selection_methods_class.getExtraTreesFeatureColumns(self,
            self.X_train, self.y_train, nFeatures
        )
        elif(method=='xgb'):
            xFeatureColumnNames = feature_selection_methods_class.getXGBoostFeatureColumns(self,
            self.X_train, self.y_train, nFeatures
        )
            
            
        X_train_fs = self.X_train[xFeatureColumnNames].copy()
        X_test_fs = self.X_test[xFeatureColumnNames].copy()
        
        return X_train_fs, X_test_fs
