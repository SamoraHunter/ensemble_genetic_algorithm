"""Evaluate an ANN-weighted ensemble."""

import logging
from typing import Any, List

import numpy as np
import torch

from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
)
from ml_grid.ga_functions.ga_ann_weight_methods import train_ann_weight

logger = logging.getLogger("ensemble_ga")


def get_ann_weighted_ensemble_predictions_eval(
    best: List, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """
    Generates and evaluates ANN-weighted ensemble predictions.

    This function fits each model in the provided ensemble on the training data,
    generates predictions for both the training set and a target set (test or
    validation), and then trains an ANN to learn optimal weights for combining
    the model predictions. The function returns the final weighted ensemble
    predictions for the target dataset.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            information, the model object, and feature columns.
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test`, `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, evaluates on the validation set (`X_test_orig`).
            If False, evaluates on the standard test set (`X_test`).
            Defaults to False.

    Returns:
        The predicted labels for the target dataset (test or validation)
            using the ANN-weighted ensemble.

    Notes:
        - Supports both scikit-learn and custom PyTorch `BinaryClassification` models.
        - Computes and prints AUC and MCC scores if verbosity is set.
        - Handles NaN predictions from the ANN by returning a random vector.
    """

    if ml_grid_object.verbose >= 11:
        logger.debug("get_ann_weighted_ensemble_predictions_eval")
        logger.debug("best: %s", best)
        logger.debug("len best: %s", len(best))

    y_test = ml_grid_object.y_test
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    y_train = ml_grid_object.y_train
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test

    target_ensemble = best[0]

    # Choose prediction target based on valid parameter
    if valid:
        if ml_grid_object.verbose >= 1:
            logger.info(f"Evaluating ANN weighted ensemble on validation set: {valid}")
        x_test = X_test_orig.copy()
        y_test_ann = y_test_orig.copy()
    else:
        if ml_grid_object.verbose >= 1:
            logger.info(f"Evaluating ANN weighted ensemble on test set: {valid}")
        x_test = X_test.copy()
        y_test_ann = y_test.copy()

    # Always fit models and generate predictions for training the ANN weighting
    prediction_array = []

    for i in range(len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        if not isinstance(target_ensemble[i][1], BinaryClassification):
            model = target_ensemble[i][1]
            if ml_grid_object.verbose >= 2:
                logger.debug(f"Fitting model {i+1}")
            model.fit(X_train[feature_columns], y_train)

            prediction_array.append(model.predict(X_train[feature_columns]))
        else:
            test_data = TestData(torch.FloatTensor(X_train[feature_columns].values))

            device = torch.device("cpu")
            model = target_ensemble[i][1]
            model.to(device)
            y_hat = model(test_data.X_data.to(device))

            y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

            y_hat = y_hat.astype(int).flatten()
            prediction_array.append(y_hat)

    prediction_matrix_X_train = np.matrix(prediction_array).astype(float)
    prediction_matrix_raw_X_train = prediction_matrix_X_train

    X_prediction_matrix_raw_X_train = prediction_matrix_raw_X_train.T

    # Now generate predictions on the target dataset (test or validation)
    prediction_array = []
    for i in range(len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        existing_columns = [
            col
            for col in feature_columns
            if col in X_train.columns and col in x_test.columns
        ]
        feature_columns = existing_columns.copy()

        if not isinstance(target_ensemble[i][1], BinaryClassification):
            prediction_array.append(
                target_ensemble[i][1].predict(x_test[feature_columns])
            )
        else:
            test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))

            device = torch.device("cpu")
            model = target_ensemble[i][1]
            model.to(device)
            y_hat = model(test_data.X_data.to(device))

            y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

            y_hat = y_hat.astype(int).flatten()
            prediction_array.append(y_hat)

    prediction_matrix_X_test = np.matrix(prediction_array).astype(float)
    prediction_matrix_raw_X_test = prediction_matrix_X_test

    prediction_matrix_raw_X_test = prediction_matrix_raw_X_test.T
    test_data = TestData(torch.FloatTensor(prediction_matrix_raw_X_test))

    train_data = TrainData(
        torch.FloatTensor(X_prediction_matrix_raw_X_train),
        torch.FloatTensor(np.array(y_train)),
    )

    y_pred_ensemble = train_ann_weight(
        X_prediction_matrix_raw_X_train.shape[1],
        int(X_prediction_matrix_raw_X_train.shape[0]),
        train_data,
        test_data,
    )

    if any(np.isnan(y_pred_ensemble)):
        logger.warning("Torch model nan, returning random y pred vector")
        random_y_pred_vector = (
            np.random.choice(
                a=[False, True],
                size=(len(y_test_ann),),
            )
        ).astype(int)
        y_pred_ensemble = random_y_pred_vector

    torch.cuda.empty_cache()

    return y_pred_ensemble
