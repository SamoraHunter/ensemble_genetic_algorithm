import logging
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC

from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.model_methods_ga import store_model
from ml_grid.util.param_space import ParamSpace

logger = logging.getLogger("ensemble_ga")


def SVC_ModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[float, SVC, List[str], int, float, np.ndarray]:
    """Generates, trains, and evaluates a Support Vector Classifier (SVC) model.

    This function performs a single trial of training and evaluating an SVC
    model. It uses a random search approach for hyperparameter tuning.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Randomly sampling hyperparameters from a comprehensive, predefined
        search space covering kernel types, regularization (C), and other
        key parameters.
    3.  Training the SVC model with the selected parameters.
    4.  Handling potential exceptions during model fitting by returning a
        default, low-performing result.
    5.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    6.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object: An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (dict): A dictionary of local parameters for this
            specific model run, which may include 'param_space_size'.

    Returns:
        tuple: A tuple containing mccscore (float), the trained model object,
        a list of feature names, the model training time (int), the
        auc_score (float), and the predictions (np.ndarray). In case of an
        error during training, a default tuple with an MCC of 0 and AUC of
        0.5 is returned.

    """
    from ml_grid.util.global_params import global_parameters

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
        "C": log_small,
        "break_ties": bool_param,
        "cache_size": [200],
        "class_weight": [None, "balanced"]
        + [{0: w} for w in [1, 2, 4, 6, 10]],  # enumerate class weight
        "coef0": log_small,
        "decision_function_shape": ["ovr"],  # , 'ovo'
        "degree": log_med,
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "max_iter": log_large_long,
        "probability": [False],
        "random_state": [None],
        "shrinking": bool_param,
        "tol": log_small,
        "verbose": [False],
    }

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    # fit model with random sample of global parameter space
    model = SVC(**sample_parameter_space)

    try:
        # Train the model--------------------------------------------------------------------
        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)
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

        return (
            mccscore,
            model,
            list(X_train.columns),
            model_train_time,
            auc_score,
            y_pred,
        )

    except Exception as e:
        logger.error(e)
        end = time.time()
        model_train_time = int(end - start)
        return (
            0,
            model,
            list(X_train.columns),
            model_train_time,
            0.5,
            np.random.choice(
                a=[False, True],
                size=(
                    len(
                        y_test,
                    )
                ),
            ).astype(int),
        )
