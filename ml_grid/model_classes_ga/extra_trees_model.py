import logging
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.model_methods_ga import store_model

logger = logging.getLogger("ensemble_ga")


def extraTreesModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[float, ExtraTreesClassifier, List[str], int, float, np.ndarray]:
    """Generates, trains, and evaluates an ExtraTreesClassifier model.

    This function performs a single trial of training and evaluating an
    ExtraTreesClassifier. It uses a random search approach for hyperparameter
    tuning from a hardcoded parameter space.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Randomly sampling hyperparameters from a predefined search space.
    3.  Training the ExtraTreesClassifier with the selected parameters.
    4.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    5.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object (Any): An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (Dict): A dictionary of local parameters for this
            specific model run (used for model storage).

    Returns:
        A tuple containing the following elements:
            - mccscore (float): The Matthews Correlation Coefficient.
            - model (ExtraTreesClassifier): The trained model object.
            - feature_names (List[str]): A list of feature names used for training.
            - model_train_time (int): The model training time in seconds.
            - auc_score (float): The ROC AUC score.
            - y_pred (np.ndarray): The model's predictions on the test set.
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

    # Initialise parameter space-----------------------------------------------------------------
    max_features_n = random.choice(list(np.arange(0.05, 0.2, 0.001)))
    min_samples_leaf_n = random.choice([2, 5, 8, 10, 15, 20, 25, 40, 50])
    n_estimators_n = random.choice(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            400,
            425,
            450,
            475,
            500,
            550,
            600,
            650,
            700,
        ]
    )
    max_depth_n = random.choice([2, 4, 8, 10, None])
    class_weight_n = random.choice(["balanced"])

    # warm_start = True
    model = ExtraTreesClassifier(
        n_estimators=n_estimators_n,
        max_features=max_features_n,
        min_samples_leaf=min_samples_leaf_n,
        max_depth=max_depth_n,
        class_weight=class_weight_n,
    )

    # Fit model-------------------------------------------------------------------------------------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mccscore = metrics.matthews_corrcoef(y_test, y_pred)

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
