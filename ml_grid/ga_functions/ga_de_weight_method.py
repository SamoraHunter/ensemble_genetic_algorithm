"""Method for generating weighted ensemble predictions using differential evolution."""

import logging
from typing import Any, List

import numpy as np
import torch

from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData, normalize

logger = logging.getLogger("ensemble_ga")


def get_weighted_ensemble_prediction_de_y_pred_valid(
    best: List, weights: np.ndarray, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """
    Generates weighted ensemble predictions for a given set of models and weights.

    This function computes weighted ensemble predictions. If `valid` is True,
    it fits each model on the training data and predicts on the validation set.
    If `valid` is False, it uses pre-computed predictions stored within the
    `best` configuration. The individual model predictions are then combined
    using the provided `weights`, normalized by L1 norm, and rounded to produce
    the final ensemble prediction.

    Args:
        best: List containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, each representing a model and its
            associated metadata (model object, feature columns, and predictions).
        weights: Array of weights to apply to each model's predictions.
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, predict on the validation set by refitting models.
            If False, use pre-computed predictions. Defaults to False.

    Returns:
        The final weighted ensemble predictions, rounded to the nearest integer
        (typically 0 or 1 for classification tasks).
    """

    round_v = np.vectorize(round)

    X_test_orig = ml_grid_object.X_test_orig
    X_train = ml_grid_object.X_train

    target_ensemble = best[0]

    if valid:
        y_train = ml_grid_object.y_train
        x_test = X_test_orig.copy()
        prediction_array = []

        for i in range(0, len(target_ensemble)):
            feature_columns = target_ensemble[i][2]

            existing_columns = [
                col
                for col in feature_columns
                if col in X_train.columns and (hasattr(x_test, "columns") or True)
            ]

            if hasattr(x_test, "columns"):
                existing_columns = [
                    col
                    for col in feature_columns
                    if col in X_train.columns and col in x_test.columns
                ]
            else:
                existing_columns = feature_columns

            model = target_ensemble[i][1]

            if not isinstance(model, BinaryClassification):
                try:
                    model.fit(X_train[existing_columns], y_train)
                    if hasattr(x_test, "columns"):
                        prediction_array.append(model.predict(x_test[existing_columns]))
                    else:
                        prediction_array.append(
                            model.predict(x_test[:, existing_columns])
                        )
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.warning(f"Fitting/Prediction failed for model {i+1}: {e}")
                    if hasattr(x_test, "columns"):
                        prediction_array.append(np.zeros(len(x_test)))
                    else:
                        prediction_array.append(np.zeros(x_test.shape[0]))
            else:
                if hasattr(x_test, "columns"):
                    test_data = TestData(
                        torch.FloatTensor(x_test[existing_columns].values)
                    )
                else:
                    test_data = TestData(torch.FloatTensor(x_test[:, existing_columns]))

                device = torch.device("cpu")
                model.to(device)
                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()

                if np.isnan(y_hat).any():
                    logger.warning(
                        "Returning dummy random yhat vector for torch pred, nan found"
                    )
                    if hasattr(x_test, "columns"):
                        y_hat = np.random.choice(a=[False, True], size=(len(y_hat),))
                    else:
                        y_hat = np.random.choice(
                            a=[False, True], size=(x_test.shape[0],)
                        )

                prediction_array.append(y_hat)

    else:
        prediction_array = []

        for i in range(0, len(target_ensemble)):
            prediction_array.append(target_ensemble[i][5])

    prediction_matrix = np.matrix(prediction_array).astype(float)

    weights = normalize(weights)

    weighted_prediction_matrix_array = np.array(prediction_matrix) * weights[:, None]
    collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
        axis=0
    )

    y_pred_weighted = round_v(collapsed_weighted_prediction_matrix_array)

    torch.cuda.empty_cache()

    return y_pred_weighted
