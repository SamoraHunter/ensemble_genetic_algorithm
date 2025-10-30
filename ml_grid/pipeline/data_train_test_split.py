from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import logging
from typing import Dict, Tuple, Union

logger = logging.getLogger("ensemble_ga")


def get_data_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    local_param_dict: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Splits data into training, testing, and validation sets.

    This function splits the input data (`X`, `y`) into three sets:
    1.  `X_train`, `y_train`: For training the model.
    2.  `X_test`, `y_test`: For evaluating the model during the genetic algorithm.
    3.  `X_test_orig`, `y_test_orig`: A hold-out validation set for final evaluation.

    It supports three modes based on `local_param_dict['resample']`:
    - `None`: Performs a standard stratified split.
    - `undersample`: Applies random undersampling to the entire dataset before splitting.
    - `oversample`: Splits the data first, then applies random oversampling only
      to the initial training portion to prevent data leakage.

    The split proportions are fixed: an initial 75/25 split creates the
    validation set (`_orig`), and the 75% portion is then split again 75/25
    to create the final training and testing sets.

    Args:
        X: The input features, either a pandas DataFrame or a NumPy array.
        y: The target variable, either a pandas Series or a NumPy array.
        local_param_dict: A dictionary of parameters, which must contain
            the 'resample' key to determine the sampling strategy.

    Returns:
        A tuple containing six elements in the following order:
        `X_train`, `X_test`, `y_train`, `y_test`, `X_test_orig`, `y_test_orig`.
    """
    # X = X
    # y = y
    # local_param_dict = local_param_dict
    # X_train_orig, X_test_orig, y_train_orig, y_test_orig = None, None, None, None
    
    random.seed(1234)
    np.random.seed(1234)

    # Check if data is valid
    if not is_valid_shape(X):
        local_param_dict["resample"] = None
        logger.warning("overriding resample with None")

    # No resampling
    if local_param_dict.get("resample") is None:
        # Split into training and testing sets
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Split training set into final training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

    # Undersampling
    elif local_param_dict.get("resample") == "undersample":

        # Undersample data
        rus = RandomUnderSampler(random_state=0)
        X, y = rus.fit_resample(X, y)

        # Split into training and testing sets
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Split training set into final training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )
        X = X_train_orig.copy()
        y = y_train_orig.copy()

    # Oversampling
    elif local_param_dict.get("resample") == "oversample":
        # Train test split
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Oversample training set
        sampling_strategy = 1
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=1)
        X_train_orig, y_train_orig = ros.fit_resample(X_train_orig, y_train_orig)
        logger.debug(y_train_orig.value_counts())

        # Split training set into final training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

    return X_train, X_test, y_train, y_test, X_test_orig, y_test_orig



def is_valid_shape(input_data: Union[pd.DataFrame, np.ndarray]) -> bool:
    """Checks if the input data is a 2D array-like structure.

    Args:
        input_data: The data to check, expected to be a pandas DataFrame
            or a NumPy array.

    Returns:
        True if the input data has 2 dimensions, False otherwise.
    """
    if isinstance(input_data, np.ndarray):
        return input_data.ndim == 2

    elif isinstance(input_data, pd.DataFrame):
        input_array = input_data.values
        return input_array.ndim == 2

    else:
        return False


## Reproduce data split:


# from ml_grid.pipeline.data_train_test_split import get_data_split

# local_param_dict  = {'resample': str(df.iloc[0]['resample'])}

# # replace nan value in local_param_dict with None

# local_param_dict = {k: v if v!= 'nan' else None for k, v in local_param_dict.items()}

# X_train, X_test, y_train, y_test, X_train_orig, y_test_orig = get_data_split(X, y, local_param_dict)