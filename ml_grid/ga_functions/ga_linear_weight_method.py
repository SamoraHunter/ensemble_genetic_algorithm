"""Linear Weight Optimization methods for ensemble predictions."""

import logging
from typing import Any, List

import numpy as np
from sklearn.linear_model import LinearRegression

from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData, normalize

logger = logging.getLogger("ensemble_ga")


def get_linear_weighted_ensemble_predictions(
    best: List, weights: np.ndarray, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """
    Generates linear-weighted ensemble predictions.

    This function fits a LinearRegression model on the predictions from base
    learners to find optimal weights for combining them. The resulting weights
    are then used to create a weighted combination of predictions.

    If `valid` is True, it fits models on training data and predicts on the
    validation set. If `valid` is False, it uses pre-computed predictions
    stored within the `best` configuration.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            information and pre-computed predictions at index 5.
        weights: Initial weights (used as starting point for optimization).
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, predict on the validation set by refitting models.
            If False, use pre-computed predictions. Defaults to False.

    Returns:
        The linear-weighted ensemble predictions as a NumPy array.
    """
    round_v = np.vectorize(round)

    X_test_orig = ml_grid_object.X_test_orig
    X_train = ml_grid_object.X_train
    y_train = ml_grid_object.y_train

    target_ensemble = best[0]

    if valid:
        x_test = X_test_orig.copy()

        prediction_array = []

        for i in range(len(target_ensemble)):
            feature_columns = target_ensemble[i][2]

            model = target_ensemble[i][1]

            if not isinstance(model, BinaryClassification):
                try:
                    model.fit(X_train[feature_columns], y_train)
                except ValueError as e:
                    logger.error(e)
                    logger.error("ValueError on fit")

                prediction_array.append(model.predict(x_test[feature_columns]))
            else:
                import torch

                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))

                device = torch.device("cpu")
                model.to(device)
                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()
                y_hat = y_hat.astype(int).flatten()

                if np.isnan(y_hat).any():
                    logger.warning(
                        "Returning dummy random yhat vector for torch pred, nan found"
                    )
                    y_hat = np.random.choice(a=[False, True], size=(len(y_hat),))

                prediction_array.append(y_hat)
    else:
        prediction_array = []

        for i in range(len(target_ensemble)):
            prediction_array.append(target_ensemble[i][5])

    prediction_matrix = np.array(prediction_array).astype(float)

    # Normalize weights
    weights = normalize(weights)

    # Apply linear weighting: weighted sum of predictions
    weighted_prediction = np.dot(prediction_matrix.T, weights)

    # Round to nearest integer (0 or 1 for classification)
    y_pred_weighted = round_v(weighted_prediction)

    return y_pred_weighted.astype(int)


def find_linear_weights(best: List, ml_grid_object: Any) -> np.ndarray:
    """
    Finds optimal linear weights using Linear Regression.

    This function uses LinearRegression to learn the optimal combination
    of base learner predictions by regressing actual labels on model predictions.
    The learned coefficients represent the weights for each model.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            metadata and pre-computed predictions at index 5.
        ml_grid_object: An object containing data splits (`y_test`, `X_train`,
            etc.) and configuration like `verbose`.

    Returns:
        The array of optimal weights learned by LinearRegression.
    """

    y_test = ml_grid_object.y_test.copy()
    if hasattr(y_test, "values"):
        y_test = y_test.values

    target_ensemble = best[0]

    # Get all predictions into a matrix
    prediction_array = []
    for i in range(len(target_ensemble)):
        pred = target_ensemble[i][5]
        prediction_array.append(pred)

    prediction_matrix = np.array(prediction_array).astype(float).T

    # Fit LinearRegression to find optimal weights
    lr = LinearRegression(fit_intercept=True)
    lr.fit(prediction_matrix, y_test)

    # Get coefficients (weights)
    weights = lr.coef_

    # Ensure non-negative weights and normalize
    weights = np.maximum(weights, 0)  # Ensure non-negative

    if not np.any(weights):
        # If all weights are zero, use equal weights
        weights = np.ones(len(weights))

    # Normalize to sum to 1
    weights = normalize(weights)

    return weights
