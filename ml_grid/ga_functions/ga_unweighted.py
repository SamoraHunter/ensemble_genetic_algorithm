from typing import Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData


def get_best_y_pred_unweighted(best: List, ml_grid_object: Any, valid: bool = False) -> List:
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
        - In non-validation mode (`valid=False`), predictions are directly taken
          from `target_ensemble[i][5]`.
    """

    if ml_grid_object.verbose >= 1:
        print("get_best_y_pred_unweighted: best:")
        print(best)
        print("len(best)")
        print(len(best))
        print("len(best[0])")
        print(len(best[0]))
        print("Valid", valid)

    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train

    if valid:
        x_test = X_test_orig.copy()
        y_test = y_test_orig.copy()

    prediction_array = []
    target_ensemble = best[0]
    if valid:
        if ml_grid_object.verbose >= 1:
            print("Predicting on validation set...")
        for i in range(0, len(target_ensemble)):

            feature_columns = list(target_ensemble[i][2])

            if type(target_ensemble[i][1]) is not BinaryClassification:
                model = target_ensemble[i][1]
                if ml_grid_object.verbose >= 2:
                    print(f"Fitting model {i+1}")
                model.fit(X_train[feature_columns], y_train)

                prediction_array.append(model.predict(x_test[feature_columns]))

            else:
                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                device = torch.device("cpu")
                model = target_ensemble[i][1]
                model.to(device)

                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()
                prediction_array.append(y_hat)

    else:
        if ml_grid_object.verbose >= 1:
            print("Predicting on non-validation set...")
            print("Evaluating...", target_ensemble)
            print(len(target_ensemble), "len(target_ensemble)")
            print(target_ensemble)
        for i in range(0, len(target_ensemble)):
            y_pred = target_ensemble[i][5]
            prediction_array.append(y_pred)

    prediction_matrix = np.matrix(prediction_array)

    y_pred_best = []
    for i in range(0, len(prediction_array[0])):
        try:
            y_pred_best.append(
                stats.mode(
                    np.matrix(prediction_array)[:, i].astype(int), keepdims=True
                )[0][0][0]
            )
        except:
            y_pred_best.append(
                stats.mode(np.matrix(prediction_array)[:, i].astype(int))[0][0][0]
            )
    return y_pred_best
