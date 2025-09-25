import re
import logging

import pandas as pd

from ml_grid.util.global_params import global_parameters
logger = logging.getLogger("ensemble_ga")


class clean_up_class:
    """A collection of methods for cleaning and standardizing pandas DataFrames.

    This class provides utilities for common data cleaning tasks such as
    removing duplicated columns, screening for non-numeric data types, and
    sanitizing column names to be compatible with machine learning libraries
    like XGBoost.

    Attributes:
        verbose (int): The verbosity level, controlled by global parameters.
        rename_cols (bool): Flag to determine if column renaming should occur.
    """

    def __init__(self):
        """Initializes the clean_up_class and loads global parameters."""

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        self.rename_cols = self.global_params.rename_cols

    def handle_duplicated_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicated columns from a DataFrame.

        Args:
            X: DataFrame to process.

        Returns:
            A copy of the DataFrame with duplicated columns removed.
        """

        try:
            if self.verbose > 1:
                logger.debug("dropping duplicated columns")

            assert X is not None, "Null pointer exception: X cannot be None."

            X = X.loc[:, ~X.columns.duplicated()].copy()

            assert X is not None, "Null pointer exception: X became None after dropping duplicated columns."

        except AssertionError as e:
            logger.error(str(e))
            raise

        except Exception as e:
            logger.error("Unhandled exception in handle_duplicated_columns: %s", e)
            raise

        return X

    def screen_non_float_types(self, X: pd.DataFrame) -> None:
        """Prints the names of columns that are not integer or float types.

        This is a debugging utility to quickly identify non-numeric columns
        in a DataFrame.

        Args:
            X: The DataFrame to screen.
        """
        if self.verbose > 1:
            logger.debug("Screening for non float data types:")
            for col in X.columns:
                if X[col].dtype != int and X[col].dtype != float:
                    logger.debug(col)

    def handle_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes DataFrame column names for XGBoost compatibility.

        This function renames columns by replacing characters that are
        incompatible with XGBoost (e.g., '[', ']', '<') with underscores.
        The operation is only performed if `self.rename_cols` is True.

        Args:
            X: The DataFrame whose columns will be sanitized.

        Returns:
            The DataFrame with sanitized column names.
        """

        if self.rename_cols:

            regex = re.compile(r"\[|\]|<", re.IGNORECASE)

            new_col_names = []

            for col in X.columns.values:
                if any(X in str(col) for X in set(("[", "]", "<"))):
                    new_col_names.append(regex.sub("_", col))
                else:
                    new_col_names.append(col)

            X.columns = new_col_names

        return X
