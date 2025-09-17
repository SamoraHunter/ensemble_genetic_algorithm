import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn import metrics
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model
from ml_grid.util.param_space import ParamSpace


def QuadraticDiscriminantAnalysis_ModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[
    float, QuadraticDiscriminantAnalysis, List[str], int, float, np.ndarray
]:
    """Generates, trains, and evaluates a QuadraticDiscriminantAnalysis model.

    This function performs a single trial of training and evaluating a
    QuadraticDiscriminantAnalysis model. It uses a random search approach
    for hyperparameter tuning.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Randomly sampling hyperparameters from a predefined search space.
    3.  Training the QuadraticDiscriminantAnalysis model.
    4.  Handling potential `LinAlgError` during fitting by falling back to
        random predictions if the model cannot be trained (e.g., due to
        collinear features).
    5.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    6.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object (Any): An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (Dict): A dictionary of local parameters for this
            specific model run, which may include 'param_space_size'.

    Returns:
        A tuple containing the following elements:
            - mccscore (float): The Matthews Correlation Coefficient.
            - model (QuadraticDiscriminantAnalysis): The trained model object.
            - feature_names (List[str]): A list of feature names used for training.
            - model_train_time (int): The model training time in seconds.
            - auc_score (float): The ROC AUC score.
            - y_pred (np.ndarray): The model's predictions on the test set.
    """
    global_parameter_val = global_parameters()

    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners

    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    start = time.time()

    X_train, X_test = feature_selection_methods_class(
        ml_grid_object
    ).get_featured_selected_training_data(method="anova")

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
        "priors": [None],
        "reg_param": log_small,
        "store_covariance": [False],
        "tol": log_small,
    }

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    # fit model with random sample of global parameter space
    model = QuadraticDiscriminantAnalysis(**sample_parameter_space)

    # Train the model--------------------------------------------------------------------
    try:

        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)
    except np.linalg.LinAlgError as e:
        if verbose >= 1:
            print("Encountered LinAlgError:", e)

        X_test_length = len(X_test)

        y_pred = np.random.randint(2, size=X_test_length)

    mccscore = matthews_corrcoef(y_test, y_pred)

    auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)
    end = time.time()
    model_train_time = int(end - start)

    if verbose >= 2:
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)

    if store_base_learners:
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

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
