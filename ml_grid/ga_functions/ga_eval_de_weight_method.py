"Evaluate a DE-weighted ensemble."

import logging
from typing import Any, List

import numpy as np
import torch

from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData, normalize

logger = logging.getLogger("ensemble_ga")


def get_de_weighted_ensemble_predictions_eval(
    best: List, weights: np.ndarray, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """
    Generates weighted ensemble predictions for evaluation.

    This function fits each model in the provided ensemble on the full training
    data, then generates predictions on either the test set or a separate
    validation set. The individual model predictions are then combined
    using the provided `weights`, normalized by L1 norm, and rounded to produce
    the final ensemble prediction.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            information, the model object, and feature columns.
        weights: Array of weights to apply to each model's predictions.
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test`, `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, predictions are made on the validation set
            (`X_test_orig`). If False, predictions are made on the standard
            test set (`X_test`). Defaults to False.

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

    # Choose prediction target based on valid parameter
    if valid:
        if ml_grid_object.verbose >= 1:
            logger.info("Evaluating weighted ensemble on validation set")
        x_test = X_test_orig.copy()
    else:
        if ml_grid_object.verbose >= 1:
            logger.info("Evaluating weighted ensemble on test set")
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
                    x_test.shape,
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

    prediction_matrix = np.matrix(prediction_array).astype(float)
    prediction_matrix_raw = prediction_matrix

    clean_prediction_matrix = prediction_matrix_raw.copy()
    weights = normalize(weights)

    weighted_prediction_matrix_array = (
        np.array(clean_prediction_matrix) * weights[:, None]
    )
    collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
        axis=0
    )

    y_pred_weighted = round_v(collapsed_weighted_prediction_matrix_array)

    torch.cuda.empty_cache()

    return y_pred_weighted
