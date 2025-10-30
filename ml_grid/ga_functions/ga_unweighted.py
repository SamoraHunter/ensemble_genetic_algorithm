"""Get unweighted ensemble predictions."""

import logging
from typing import Any, List

import numpy as np
import torch
from scipy import stats

from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData

logger = logging.getLogger("ensemble_ga")


def get_unweighted_ensemble_predictions(
    best: List, ml_grid_object: Any, valid: bool = False
) -> List:
    """
    Generates an unweighted ensemble prediction by majority vote (mode).

    This function generates predictions from an ensemble of models. If `valid`
    is True, it fits each model on the training data and predicts on the
    validation set. If `valid` is False, it uses pre-computed predictions
    stored within the `best` configuration. The final prediction for each
    sample is the mode of the predictions from all models in the ensemble.

    Args:
        best: A list containing the best ensemble configuration. The first
            element (`best[0]`) is a list of tuples, where each tuple holds
            model information, the model object, feature columns, and, if
            `valid` is False, pre-computed predictions.
        ml_grid_object: An object containing data splits (`X_train`,
            `y_train`, `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, predict on the validation set by refitting models.
            If False, use pre-computed predictions. Defaults to False.

    Returns:
        A list of final ensemble predictions, determined by the mode of
        individual model predictions for each sample.

    Notes:
        - Supports both scikit-learn style models and PyTorch `BinaryClassification` models.
        - For PyTorch models, predictions are generated via the forward pass and
          thresholded with `torch.sigmoid`.
        - In non-validation mode (`valid`=False), predictions are directly taken
          from `target_ensemble[i][5]`.
    """

    if ml_grid_object.verbose >= 2:
        logger.debug("get_unweighted_ensemble_predictions: best: %s", best)
        logger.debug("len(best): %s", len(best))
        logger.debug("len(best[0]): %s", len(best[0]))
        logger.debug("Valid: %s", valid)

    X_test_orig = ml_grid_object.X_test_orig
    X_train = ml_grid_object.X_train
    y_train = ml_grid_object.y_train

    if valid:
        x_test = X_test_orig.copy()

    prediction_array = []
    target_ensemble = best[0]
    if valid:
        if ml_grid_object.verbose >= 1:
            logger.info("Predicting on validation set...")
        for i in range(len(target_ensemble)):

            feature_columns = list(target_ensemble[i][2])

            if not isinstance(target_ensemble[i][1], BinaryClassification):
                model = target_ensemble[i][1]
                if ml_grid_object.verbose >= 2:
                    logger.debug(f"Fitting model {i+1}")
                model.fit(X_train[feature_columns], y_train)

                prediction_array.append(model.predict(x_test[feature_columns]))

            else:
                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))

                device = torch.device("cpu")
                model = target_ensemble[i][1]
                model.to(device)

                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()
                prediction_array.append(y_hat)

    else:
        if ml_grid_object.verbose >= 1:
            logger.info("Predicting on non-validation set...")
        if ml_grid_object.verbose >= 2:
            logger.debug("Evaluating... %s", target_ensemble)
            logger.debug("%s len(target_ensemble)", len(target_ensemble))
            logger.debug("%s", target_ensemble)
        for i in range(len(target_ensemble)):
            y_pred = target_ensemble[i][5]
            prediction_array.append(y_pred)

    y_pred_best = []
    for i in range(len(prediction_array[0])):
        try:
            # Scipy v1.9.0 and later
            y_pred_best.append(
                stats.mode(
                    np.matrix(prediction_array)[:, i].astype(int), keepdims=True
                )[0][0][0]
            )
        except (TypeError, IndexError):  # E722
            # Scipy v1.8.0 and earlier
            y_pred_best.append(
                stats.mode(np.matrix(prediction_array)[:, i].astype(int))[0][0]
            )
    return y_pred_best
