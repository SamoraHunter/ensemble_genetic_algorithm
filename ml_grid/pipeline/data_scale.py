import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler


class data_scale_methods:

    def __init__(self):
        """ "data scaling methods"""

    def standard_scale_method(self, X):
        """
        Standardize data by subtracting the mean and dividing by the standard deviation.
        This is done column-wise.

        Parameters
        ----------
        X : DataFrame
            DataFrame to be standardized

        Returns
        -------
        X_scaled : DataFrame
            Scaled DataFrame
        """

        # can add param dict method for split

        col_names = X.columns
        scaler = ColumnTransformer(
            # name of transformer
            # instance of transformer
            # column names to be transformed
            [
                (
                    "standard scaler",
                    StandardScaler(),
                    list(X.columns),
                )
            ],
            # what to do with the rest of the columns
            remainder="passthrough",
        )
        X = scaler.fit_transform(X)

        X = pd.DataFrame(X, columns=col_names)

        return X
