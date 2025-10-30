"""Finds optimal ensemble weights using Differential Evolution."""

import logging
import time
from typing import Any, List

import numpy as np
import scipy
from sklearn import metrics

from ml_grid.ga_functions.ga_ann_util import normalize

logger = logging.getLogger("ensemble_ga")

round_v = np.vectorize(round)


def get_weighted_ensemble_prediction_de(
    weights: np.ndarray, prediction_matrix_raw: np.ndarray, y_test: np.ndarray
) -> float:
    """Computes weighted ensemble prediction and returns 1 - AUC score.

    This function is used as an objective function for optimization algorithms
    like Differential Evolution. It takes a set of weights, applies them to
    a matrix of model predictions, calculates the resulting ensemble
    prediction, and evaluates it against the ground truth using the ROC AUC
    score. The function returns `1 - AUC`, so minimizing this value maximizes
    the AUC.

    Args:
        weights: An array of weights for each model in the ensemble.
        prediction_matrix_raw: A 2D array where each row contains the
            predictions from a single model for all test samples.
        y_test: The ground truth labels for the test samples.

    Returns:
        The value of `1 - ROC AUC score` for the weighted ensemble prediction.

    Raises:
        Exception: If an error occurs during the ROC AUC score calculation.
    """

    clean_prediction_matrix = prediction_matrix_raw.copy()
    weights = normalize(weights)

    weighted_prediction_matrix_array = (
        np.array(clean_prediction_matrix) * weights[:, None]
    )
    collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
        axis=0
    )

    y_pred_best = round_v(collapsed_weighted_prediction_matrix_array)
    try:
        auc = metrics.roc_auc_score(y_test, y_pred_best)
        score = auc
        return 1 - score
    except Exception as e:
        logger.error(y_test)
        logger.error(y_pred_best)
        logger.error(type(y_test))
        logger.error(type(y_pred_best))
        raise e


def find_ensemble_weights_de(best: List, ml_grid_object: Any) -> np.ndarray:
    """Finds optimal ensemble weights using Differential Evolution.

    This function uses the Differential Evolution (DE) optimization algorithm
    to find the optimal set of weights for combining predictions from an
    ensemble of models. The objective is to maximize the ROC AUC score on the
    test set. It uses pre-computed predictions stored within the `best`
    configuration.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            metadata and pre-computed predictions at index 5.
        ml_grid_object: An object containing data splits (`y_test`) and
            configuration like `verbose`.

    Returns:
        The array of optimal weights for the ensemble models as determined
        by Differential Evolution.

    Raises:
        Exception: If the Differential Evolution optimization fails, the exception is printed and re-raised.
    """
    y_test = ml_grid_object.y_test.copy()  # WRITEABLE error fix?
    if hasattr(y_test, "values"):
        y_test = y_test.values

    debug = ml_grid_object.verbose > 11

    if debug:
        logger.debug("find_ensemble_weights_de, best: %s", best)

    model_train_time_warning_threshold = 5
    # Get prediction matrix:
    prediction_array = []
    target_ensemble = best[0]
    for i in range(len(target_ensemble)):
        y_pred = target_ensemble[i][5]
        prediction_array.append(y_pred)

    prediction_matrix = np.matrix(prediction_array).astype(float)
    prediction_matrix_raw = prediction_matrix
    y_pred_best = [
        round(np.mean(prediction_matrix_raw[:, i]))
        for i in range(len(prediction_array[0]))
    ]
    auc = metrics.roc_auc_score(y_test, y_pred_best)
    logger.info("Unweighted ensemble AUC: %s", auc)

    bounds = [(0, 1) for _ in range(len(best[0]))]

    start = time.time()
    try:
        de = scipy.optimize.differential_evolution(
            get_weighted_ensemble_prediction_de,
            bounds,
            args=((prediction_matrix_raw, y_test)),
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
        logger.error("Failed on find_ensemble_weights_de: %s", e)
        logger.error("%s, %s", prediction_matrix_raw, y_test)
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
