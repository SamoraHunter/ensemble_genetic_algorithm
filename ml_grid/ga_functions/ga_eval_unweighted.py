from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, List
import numpy as np
from scipy import stats


def get_best_y_pred_unweighted_eval(best: List, ml_grid_object: Any, valid: bool = False) -> List:
    """
    Generates unweighted ensemble predictions by majority vote for evaluation.

    This function fits each model in the provided ensemble on the full training
    data, then generates predictions on either the test set or a separate
    validation set. The final prediction for each sample is determined by the
    mode (majority vote) of the predictions from all models in the ensemble.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, where each tuple holds model
            information, the model object, and feature columns.
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test`, `X_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, predictions are made on the validation set
            (`X_test_orig`). If False, predictions are made on the standard
            test set (`X_test`). Defaults to False.

    Returns:
        A list of the final ensemble predictions for the selected dataset,
        determined by majority vote.
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

    # Choose prediction target based on valid parameter
    if valid:
        x_test = X_test_orig.copy()
        y_test = y_test_orig.copy()
        if ml_grid_object.verbose >= 1:
            print("Predicting on validation set...")
    else:
        x_test = X_test.copy()
        if ml_grid_object.verbose >= 1:
            print("Predicting on test set...")

    prediction_array = []
    target_ensemble = best[0]

    # Always fit models and make predictions
    for i in range(0, len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        if type(target_ensemble[i][1]) is not BinaryClassification:
            model = target_ensemble[i][1]
            if ml_grid_object.verbose >= 2:
                print(f"Fitting model {i+1}")
            model.fit(X_train[feature_columns], y_train)

            prediction_array.append(model.predict(x_test[feature_columns]))

        else:
            # Handle BinaryClassification (PyTorch) models
            test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
            test_loader = DataLoader(dataset=test_data, batch_size=1)

            device = torch.device("cpu")
            model = target_ensemble[i][1]
            model.to(device)

            y_hat = model(test_data.X_data.to(device))

            y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

            y_hat = y_hat.astype(int).flatten()
            prediction_array.append(y_hat)

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
