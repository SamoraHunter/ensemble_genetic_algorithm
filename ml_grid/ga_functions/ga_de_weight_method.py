"""Method for generating weighted ensemble predictions using differential evolution."""

import numpy as np
import torch
from typing import Any, List
import logging

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
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train

    target_ensemble = best[0]
    if valid:
        if ml_grid_object.verbose >= 1:
            logger.info("Evaluating weighted ensemble on validation set")
        x_test = X_test_orig.copy()

        prediction_array = []

        for i in range(0, len(target_ensemble)):
            feature_columns = target_ensemble[i][2]

            model = target_ensemble[i][1]

            if not isinstance(model, BinaryClassification):
                if ml_grid_object.verbose >= 2:
                    logger.debug(f"Fitting model {i+1}")

                try:
                    model.fit(X_train[feature_columns], y_train)
                except ValueError as e:
                    logger.error(e)
                    logger.error("ValueError on fit")
                    logger.error("feature_columns")
                    logger.error(len(feature_columns))
                    logger.error(
                        "%s, %s, %s, %s, %s",
                        X_train.shape,
                        X_test.shape,
                        type(X_train),
                        type(y_train),
                        type(feature_columns),
                    )

                prediction_array.append(model.predict(x_test[feature_columns]))
            else:
                if ml_grid_object.verbose >= 2:
                    logger.debug(f"Handling torch model prediction for model {i+1}")
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

        for i in range(0, len(target_ensemble)):
            prediction_array.append(target_ensemble[i][5])

    prediction_matrix = np.matrix(prediction_array)
    prediction_matrix = prediction_matrix.astype(float)

    weights = normalize(weights)

    weighted_prediction_matrix_array = (
        np.array(prediction_matrix) * weights[:, None]
    )
    collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
        axis=0
    )

    y_pred_weighted = round_v(collapsed_weighted_prediction_matrix_array)

    torch.cuda.empty_cache()

    return y_pred_weighted
