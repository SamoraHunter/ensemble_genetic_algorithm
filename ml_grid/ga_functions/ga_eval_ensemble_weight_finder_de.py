"""
Finds optimal ensemble weights using Differential Evolution for evaluation."""

import logging
import time
from typing import Any, List

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn import metrics

from ml_grid.ga_functions.ga_ann_util import BinaryClassification
from ml_grid.ga_functions.ga_ensemble_weight_finder_de import (
    get_weighted_ensemble_prediction_de,
)
from ml_grid.pipeline.torch_binary_classification_method_ga import TestData

logger = logging.getLogger("ensemble_ga")


def find_ensemble_weights_de_eval(
    best: List, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """Finds optimal ensemble weights using Differential Evolution for evaluation.

    This function fits each model in the provided ensemble on the full training
    data, then generates predictions on a target set (test or validation). It
    then uses the Differential Evolution (DE) optimization algorithm to find
    the optimal set of weights for combining these predictions to maximize the
    ROC AUC score.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            information, the model object, and feature columns.
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test`, `y_test`, `X_test_orig`, `y_test_orig`) and configuration
            like `verbose`.
        valid: If True, optimizes weights based on the validation set
            (`X_test_orig`). If False, uses the standard test set (`X_test`).
            Defaults to False.

    Returns:
        The array of optimal weights for the ensemble models as determined
        by Differential Evolution.

    Raises:
        ValueError: If model fitting or prediction fails.
        Exception: If the differential evolution optimization fails.
    """
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    # Choose prediction target based on valid parameter
    if valid:
        if ml_grid_object.verbose >= 1:
            logger.info("Finding optimal weights using validation set")
        x_test = X_test_orig.copy()
        y_test_target = y_test_orig.copy()
    else:
        if ml_grid_object.verbose >= 1:
            logger.info("Finding optimal weights using test set")
        x_test = X_test.copy()
        y_test_target = y_test.copy()

    if isinstance(y_test_target, pd.Series):
        y_test_target = y_test_target.values

    debug = ml_grid_object.verbose > 11

    if debug:
        logger.debug("find_ensemble_weights_de_eval, best: %s", best)

    model_train_time_warning_threshold = 5

    # Always fit models and generate prediction matrix
    prediction_array = []
    target_ensemble = best[0]

    for i in range(len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        existing_columns = [
            col
            for col in feature_columns
            if col in X_train.columns and col in x_test.columns
        ]

        missing_columns = [
            col for col in existing_columns if col not in feature_columns
        ]

        if ml_grid_object.verbose >= 1 and len(missing_columns) >= 1:
            logger.warning(
                "Warning: The following columns do not exist in feature_columns:"
            )
            logger.warning("\n".join(missing_columns))

        feature_columns = existing_columns.copy()

        if not isinstance(target_ensemble[i][1], BinaryClassification):
            model = target_ensemble[i][1]
            if ml_grid_object.verbose >= 2:
                logger.debug(f"Fitting model {i+1} for weight optimization")

            try:
                model.fit(X_train[feature_columns], y_train)
                y_pred = model.predict(x_test[feature_columns])
            except ValueError as e:
                logger.error(f"ValueError on fit for model {i+1}: {e}")
                logger.error("feature_columns length: %s", len(feature_columns))
                logger.error(
                    "X_train shape: %s, x_test shape: %s", X_train.shape, x_test.shape
                )
                raise e

            prediction_array.append(y_pred)

        else:
            # Handle BinaryClassification (PyTorch) models
            if ml_grid_object.verbose >= 2:
                logger.debug(f"Handling torch model {i+1} for weight optimization")

            test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))

            device = torch.device("cpu")
            model = target_ensemble[i][1]
            model.to(device)

            y_hat = model(test_data.X_data.to(device))
            y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()
            y_hat = y_hat.astype(int).flatten()

            if np.isnan(y_hat).any():
                logger.warning(
                    f"Returning dummy random yhat vector for torch model {i+1}, nan found"
                )
                y_hat = np.random.choice(a=[False, True], size=(len(y_hat),))

            prediction_array.append(y_hat)

    prediction_matrix = np.matrix(prediction_array).astype(float)
    prediction_matrix_raw = prediction_matrix

    # Calculate unweighted ensemble performance
    y_pred_best = [
        round(np.mean(prediction_matrix_raw[:, i]))
        for i in range(len(prediction_array[0]))
    ]
    auc = metrics.roc_auc_score(y_test_target, y_pred_best)
    logger.info("Unweighted ensemble AUC: %s", auc)

    bounds = [(0, 1) for _ in range(len(best[0]))]

    start = time.time()
    try:
        de = scipy.optimize.differential_evolution(
            get_weighted_ensemble_prediction_de,
            bounds,
            args=((prediction_matrix_raw, y_test_target)),
            strategy="best1bin",
            maxiter=20,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=False,
            polish=True,
            init="latinhypercube",
            atol=0,
            updating="immediate",
            workers=4,
            constraints=(),
            x0=None,
        )
    except Exception as e:
        logger.error("Failed on find_ensemble_weights_de_eval: %s", e)
        logger.error("%s, %s", prediction_matrix_raw, y_test_target)
        raise e

    score = 1 - de.fun
    optimal_weights = de.x

    end = time.time()
    model_train_time = int(end - start)
    if debug:
        if model_train_time > model_train_time_warning_threshold:
            logger.warning(
                "Warning long DE weights train time, %s, %s",
                model_train_time,
                model_train_time_warning_threshold,
            )

    logger.info("best weighted score: %s, difference: %s", score, score - auc)
    return optimal_weights
