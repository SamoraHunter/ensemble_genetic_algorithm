from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def get_data_split(X, y, local_param_dict):
    """
    This function gets data split based on the resampling
    method defined in local_param_dict. The function
    returns training sets and final validation sets.

    Parameters
    ----------
    X: DataFrame
        Feature data

    y: Series
        Target data

    local_param_dict: dict
        Dictionary of parameters defined in the
        local parameter grid

    Returns
    -------
    X_train: DataFrame
        Training feature data

    X_test: DataFrame
        Testing feature data

    y_train: Series
        Training target data

    y_test: Series
        Testing target data

    X_test_orig: DataFrame
        Testing feature data before oversampling or
        undersampling

    y_test_orig: Series
        Testing target data before oversampling or
        undersampling
    """

    if local_param_dict.get("resample") is None:
        # If resampling is set to None, use default train_test_split
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

    elif local_param_dict.get("resample") == "undersample":
        # Undersample the data by randomly selecting samples from the majority class
        # print("undersample..")
        # print(y.shape)
        # print(X.shape)
        rus = RandomUnderSampler(random_state=0)
        X, y = rus.fit_resample(X, y)
        # Create validation set
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Resplit holding back _orig
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )
        X = X_train_orig.copy()
        y = y_train_orig.copy()

    elif local_param_dict.get("resample") == "oversample":
        # Oversample the data by randomly selecting samples from the minority class
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        sampling_strategy = 1
        ros = RandomOverSampler(sampling_strategy=sampling_strategy)
        X_train_orig, y_train_orig = ros.fit_resample(X_train_orig, y_train_orig)
        print(y_train_orig.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

    return X_train, X_test, y_train, y_test, X_test_orig, y_test_orig
