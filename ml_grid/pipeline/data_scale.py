import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class data_scale_methods:
    """A collection of methods for scaling numerical data."""

    def __init__(self):
        """Initializes the data_scale_methods class."""

    def standard_scale_method(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes features by removing the mean and scaling to unit variance.

        This method applies scikit-learn's `StandardScaler` to all columns
        of the input DataFrame while preserving the original index.

        Args:
            X: The input DataFrame with numerical features to be scaled.

        Returns:
            A new DataFrame with the same columns and index as the input, but
            with scaled values.
        """
        col_names = X.columns
        original_index = X.index

        scaler = ColumnTransformer(
            transformers=[
                (
                    "standard_scaler",
                    StandardScaler(),
                    col_names,
                )
            ],
            remainder="passthrough",
        )
        X_scaled_np = scaler.fit_transform(X)

        X = pd.DataFrame(X_scaled_np, columns=col_names, index=original_index)

        return X
