import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData


def get_best_y_pred_unweighted(best, ml_grid_object, valid=False):
    """
    Generates the ensemble prediction by taking the mode (majority vote) of predictions from multiple models.

    Args:
        best (list): A list containing the best ensemble configuration. The first element (best[0]) is expected to be a list of tuples,
            where each tuple contains model information and associated feature columns.
        ml_grid_object (object): An object containing training and test data, as well as configuration parameters such as verbosity.
            Expected attributes include:
                - X_train, X_test: Feature data for training and testing.
                - y_train: Target data for training.
                - X_test_orig, y_test_orig: Original test feature and target data.
                - verbose: Verbosity level for logging.
        valid (bool, optional): If True, predictions are made on the validation set (original test data).
            If False, predictions are taken from precomputed values in the ensemble configuration. Defaults to False.

    Returns:
        list: The final ensemble predictions, where each prediction is determined by the mode of the individual model predictions for that sample.

    Notes:
        - For non-binary classification models, the model is fitted on the training data and predictions are made on the test/validation set.
        - For binary classification models (assumed to be PyTorch models), predictions are made using the model's forward pass and thresholded.
        - In non-validation mode, predictions are directly taken from the ensemble configuration without refitting.
        - The function assumes that each model in the ensemble is associated with a set of feature columns and either a fitted model or precomputed predictions.
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
