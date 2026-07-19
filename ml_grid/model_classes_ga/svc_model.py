import logging
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.model_methods_ga import store_model
from ml_grid.util.param_space import ParamSpace

logger = logging.getLogger("ensemble_ga")


"""
Module providing SVC model generation with ANOVA feature selection and random search hyperparameter tuning.

This module implements the SVC_ModelGenerator function which creates, trains, and evaluates Support Vector
Classifier models. It uses ANOVA-based feature selection to preprocess data and performs random search over
a comprehensive parameter space including kernel types, regularization (C), gamma, degree, and other SVM-specific
hyperparameters.

The generated models are evaluated using Matthews Correlation Coefficient (MCC) and ROC AUC scores, with optional
model storage functionality for downstream analysis.
"""


def SVC_ModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[float, Any, List[str], int, float, np.ndarray]:
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
    verbose = ml_grid_object.global_params.verbose

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
    log_large_long = param_dict["log_large_long"]
    log_med = param_dict["log_med"]

    # Initialise global parameter space----------------------------------------------------------------

    parameter_space = {
        "C": log_small,  # Regularization parameter for SVM
        "break_ties": bool_param,  # Whether to break ties in decision function predictions
        "cache_size": [200],  # Size of the kernel cache in KB
        "class_weight": [
            None,
            "balanced",
        ]  # Class weight configuration for imbalanced data
        + [{0: w} for w in [1, 2, 4, 6, 10]],  # enumerate class weight
        "coef0": log_small,  # Kernel coefficient for poly and sigmoid kernels
        "decision_function_shape": [
            "ovr"
        ],  # Decision function shape (hardcoded to 'ovr' for one-vs-rest)
        "degree": log_med,  # Degree of polynomial kernel
        "gamma": [
            "scale",
            "auto",
        ],  # Kernel coefficient for rbf, poly, and sigmoid kernels
        "kernel": ["rbf", "linear", "poly", "sigmoid"],  # SVM kernel type
        "max_iter": log_large_long,  # Maximum number of iterations allowed
        "probability": [False],  # Whether to enable probability estimates
        "random_state": [None],  # Random state for reproducible results
        "shrinking": bool_param,  # Whether to use the shrinking heuristic
        "tol": log_small,  # Tolerance for stopping criterion
        "verbose": [False],  # Enable verbose output
    }

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    # fit model with random sample of global parameter space
    model = make_pipeline(StandardScaler(), SVC(**sample_parameter_space))

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
