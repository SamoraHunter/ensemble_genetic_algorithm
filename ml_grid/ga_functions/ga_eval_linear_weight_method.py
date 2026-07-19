"""Linear Weight Optimization evaluation methods for ensemble predictions."""

import logging
from typing import Any, List

import numpy as np
from sklearn.linear_model import LinearRegression

from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData, normalize

logger = logging.getLogger("ensemble_ga")


def get_linear_weighted_ensemble_predictions_eval(
    best: List, weights: np.ndarray, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """
    Generates linear-weighted ensemble predictions for evaluation.

    This function uses LinearRegression to find optimal weights and then applies
    them to combine predictions from base learners. Unlike the training-time
    version, this function always refits models on the full training data before
    making predictions.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            information and pre-computed predictions at index 5.
        weights: Initial weights (used as starting point for optimization).
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test`, `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, predictions are made on the validation set
            (`X_test_orig`). If False, predictions are made on the standard
            test set (`X_test`). Defaults to False.

    Returns:
        The linear-weighted ensemble predictions as a NumPy array.
    """

    X_test_orig = ml_grid_object.X_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train

    target_ensemble = best[0]

    # Choose prediction target based on valid parameter
    if valid:
        if ml_grid_object.verbose >= 1:
            logger.info("Evaluating linear-weighted ensemble on validation set")
        x_test = X_test_orig.copy()
    else:
        if ml_grid_object.verbose >= 1:
            logger.info("Evaluating linear-weighted ensemble on test set")
        x_test = X_test.copy()

    prediction_array = []

    # Always fit models and make predictions
    for i in range(len(target_ensemble)):
        feature_columns = target_ensemble[i][2]

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

        model = target_ensemble[i][1]

        if not isinstance(model, BinaryClassification):
            if ml_grid_object.verbose >= 2:
                logger.debug(f"Fitting model {i + 1}")
            try:
                model.fit(X_train[feature_columns], y_train)
                prediction_array.append(model.predict(x_test[feature_columns]))
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"Fitting/Prediction failed for model {i + 1}: {e}")
                prediction_array.append(np.zeros(len(x_test)))
        else:
            if ml_grid_object.verbose >= 2:
                logger.debug(f"Handling torch model prediction for model {i + 1}")

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

    # If using pre-computed predictions (valid=False with stored predictions):
    if not valid:
        prediction_array = []
        for i in range(len(target_ensemble)):
            pred = target_ensemble[i][5]
            prediction_array.append(pred)

    prediction_matrix = np.array(prediction_array).astype(float)

    # Normalize weights
    weights = normalize(weights)

    # Apply linear weighting: weighted sum of predictions
    weighted_prediction = np.dot(prediction_matrix.T, weights)

    # Round to nearest integer (0 or 1 for classification)
    y_pred_weighted = np.round(weighted_prediction)

    return y_pred_weighted.astype(int)


def find_linear_weights_eval(best: List, ml_grid_object: Any) -> np.ndarray:
    """
    Finds optimal linear weights using Linear Regression for evaluation.

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
