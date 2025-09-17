import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn import metrics
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model
from ml_grid.util.param_space import ParamSpace


def AdaBoostClassifierModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[float, AdaBoostClassifier, List[str], int, float, np.ndarray]:
    """Generates, trains, and evaluates an AdaBoostClassifier model.

    This function performs a single trial of training and evaluating an
    AdaBoostClassifier. It uses a random search approach for hyperparameter
    tuning.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Randomly sampling hyperparameters from a predefined search space.
    3.  Training the AdaBoostClassifier with the selected parameters.
    4.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    5.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object (Any): An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (Dict): A dictionary of local parameters for this
            specific model run, which may include 'param_space_size'.

    Returns:
        A tuple containing the following elements:
            - mccscore (float): The Matthews Correlation Coefficient.
            - model (AdaBoostClassifier): The trained model object.
            - feature_names (List[str]): A list of feature names used for training.
            - model_train_time (int): The model training time in seconds.
            - auc_score (float): The ROC AUC score.
            - y_pred (np.ndarray): The model's predictions on the test set.
    """
    # Retrieve global parameters
    global_parameter_val = global_parameters()
    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners

    # Retrieve data
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    # Start timer
    start = time.time()

    # Perform feature selection
    X_train, X_test = feature_selection_methods_class(
        ml_grid_object
    ).get_featured_selected_training_data(method="anova")

    # Generate parameter space
    param_dict = ParamSpace(size=local_param_dict.get("param_space_size")).param_dict

    log_small = param_dict["log_small"]
    bool_param = param_dict["bool_param"]
    log_large = param_dict["log_large"]
    log_large_long = param_dict["log_large_long"]
    log_med_long = param_dict["log_med_long"]
    log_med = param_dict["log_med"]
    log_zero_one = param_dict["log_zero_one"]
    lin_zero_one = param_dict["lin_zero_one"]

    # Initialise global parameter space----------------------------------------------------------------
    parameter_space = {
        "algorithm": [
            "SAMME",
        ],  # New in version 1.2: base_estimator_ was renamed to estimator_.
        "estimator": [None],
        "learning_rate": log_small,
        "n_estimators": log_large_long,
        "random_state": [None],
    }

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    # fit model with random sample of global parameter space
    model = AdaBoostClassifier(**sample_parameter_space)

    # Train the model--------------------------------------------------------------------
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    mccscore = matthews_corrcoef(y_test, y_pred)

    auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)
    end = time.time()
    model_train_time = int(end - start)

    if verbose >= 2:
        # Print performance of base learner
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)

    if store_base_learners:
        # Store the model, performance, and other relevant information
        store_model(
            ml_grid_object,
            local_param_dict,
            mccscore,
            model,
            list(X_train.columns),
            model_train_time,
            auc_score,
            y_pred,
        )

    # Return performance of base learner
    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
