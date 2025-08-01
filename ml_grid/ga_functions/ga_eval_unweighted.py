from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats


def get_best_y_pred_unweighted_eval(best, ml_grid_object, valid=False):
    """
    Generates ensemble predictions using the unweighted majority vote from a set of models.

    This function fits each model in the provided ensemble on the training data, makes predictions
    on either the test or validation set (depending on the `valid` flag), and aggregates the predictions
    using the statistical mode (majority vote) for each sample.

    Args:
        best (list): A list containing the ensemble configuration. The first element should be a list of tuples,
            where each tuple contains information about a model, its instance, and the feature columns to use.
        ml_grid_object (object): An object containing the dataset splits and configuration. Must have attributes:
            - X_train, X_test, y_train, X_test_orig, y_test_orig, verbose.
        valid (bool, optional): If True, predictions are made on the original validation set (X_test_orig, y_test_orig).
            If False (default), predictions are made on the test set (X_test).

    Returns:
        list: The ensemble's predicted labels for each sample in the selected test/validation set, determined by majority vote.

    Notes:
        - Supports both scikit-learn style models and custom BinaryClassification (PyTorch) models.
        - Assumes all models are classifiers and output integer class labels.
        - Prints verbose output if `ml_grid_object.verbose` is set.
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
