import logging
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("ensemble_ga")


def remove_constant_columns(
    X: pd.DataFrame, drop_list: Optional[List[str]] = None, verbose: int = 1
) -> List[str]:
    """Identifies constant columns and adds them to a drop list.

    This function inspects a DataFrame to find columns where all values are
    the same. The names of these constant columns are then appended to the
    provided `drop_list`.

    Args:
        X: The DataFrame to check for constant columns.
        drop_list: A list of column names already marked for dropping. If
            None, a new list is created. Defaults to None.
        verbose: The verbosity level for logging. Defaults to 1.

    Returns:
        The updated list of columns to drop, now including any constant
        columns found in `X`.
    """
    try:
        if verbose > 1:
            logger.debug("Identifying constant columns")

        assert X is not None, "Null pointer exception: X cannot be None."

        # Initialize drop_list if not provided
        if drop_list is None:
            drop_list = []

        # Identify constant columns
        constant_columns = [col for col in X.columns if X[col].nunique() == 1]

        if constant_columns:
            if verbose > 1:
                logger.debug("Constant columns identified: %s", constant_columns)

            # Add constant columns to drop_list
            drop_list.extend(constant_columns)

    except AssertionError as e:
        logger.error(str(e))
        raise

    except Exception as e:
        logger.error("Unhandled exception: %s", str(e))
        raise

    return drop_list


def remove_constant_columns_with_debug(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_test_orig: pd.DataFrame,
    verbosity: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identifies and removes constant columns across multiple data splits.

    This function finds columns that have zero variance (are constant) in any
    of the provided data splits (train, test, and original test). It then
    removes these columns from all three DataFrames to ensure they maintain
    a consistent feature set.

    Args:
        X_train: The training features DataFrame.
        X_test: The testing features DataFrame.
        X_test_orig: The original (validation) testing features DataFrame.
        verbosity: The verbosity level for logging debug information.

    Returns:
        A tuple containing the three input DataFrames with constant columns removed.
    """
    if verbosity > 0:
        logger.debug("Initial X_train shape: %s", X_train.shape)
        logger.debug("Initial X_test shape: %s", X_test.shape)
        logger.debug("Initial X_test_orig shape: %s", X_test_orig.shape)

    # Calculate variance for each dataset
    train_variances = X_train.var(axis=0)
    test_variances = X_test.var(axis=0)
    test_orig_variances = X_test_orig.var(axis=0)  # ADD THIS

    if verbosity > 1:
        logger.debug("Variance of X_train columns:\n%s", train_variances)
        logger.debug("Variance of X_test columns:\n%s", test_variances)
        logger.debug("Variance of X_test_orig columns:\n%s", test_orig_variances)  # ADD THIS

    # Identify constant columns in each dataset
    constant_columns_train = train_variances[train_variances == 0].index
    constant_columns_test = test_variances[test_variances == 0].index
    constant_columns_test_orig = test_orig_variances[
        test_orig_variances == 0
    ].index  # ADD THIS

    if verbosity > 0:
        logger.debug("Constant columns in X_train: %s", list(constant_columns_train))
        logger.debug("Constant columns in X_test: %s", list(constant_columns_test))
        logger.debug("Constant columns in X_test_orig: %s", list(constant_columns_test_orig))

    # Combine constant columns from ALL THREE datasets
    constant_columns = constant_columns_train.union(constant_columns_test).union(
        constant_columns_test_orig
    )  # MODIFY THIS

    if verbosity > 0:
        logger.debug("Total constant columns to remove: %s", list(constant_columns))

    # Remove the constant columns from all datasets
    X_train = X_train.loc[:, ~X_train.columns.isin(constant_columns)]
    X_test = X_test.loc[:, ~X_test.columns.isin(constant_columns)]
    X_test_orig = X_test_orig.loc[:, ~X_test_orig.columns.isin(constant_columns)]

    if verbosity > 0:
        logger.debug("Final X_train shape: %s", X_train.shape)
        logger.debug("Final X_test shape: %s", X_test.shape)
        logger.debug("Final X_test_orig shape: %s", X_test_orig.shape)

        # Verify all datasets have same columns
        assert (
            set(X_train.columns) == set(X_test.columns) == set(X_test_orig.columns)
        ), "Column mismatch after constant column removal!"

    return X_train, X_test, X_test_orig
