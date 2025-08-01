from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData, TrainData
from ml_grid.ga_functions.ga_ann_weight_methods import train_ann_weight
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import time


def get_y_pred_ann_torch_weighting_eval(best, ml_grid_object, valid=False):
    """
    Generates ensemble predictions using an ANN-based weighting method for a set of models,
    and evaluates the ensemble on either the test or validation set.

    This function fits each model in the provided ensemble on the training data,
    generates predictions for both training and target (test/validation) datasets,
    and then trains an ANN to learn optimal weights for combining the model predictions.
    The function returns the weighted ensemble predictions for the target dataset.

    Args:
        best (list): List of tuples representing the ensemble. Each tuple contains
            (model_name, model_object, feature_columns).
        ml_grid_object (object): An object containing all necessary data and configuration,
            including training and test/validation sets, labels, and verbosity level.
        valid (bool, optional): If True, evaluates on the validation set; otherwise,
            evaluates on the test set. Default is False.

    Returns:
        np.ndarray: The predicted labels for the target dataset (test or validation)
            using the ANN-weighted ensemble.

    Notes:
        - The function supports both scikit-learn models and custom PyTorch models
          (e.g., BinaryClassification).
        - It computes and prints AUC and MCC scores for both unweighted and weighted ensembles
          if verbosity is set appropriately.
        - Handles NaN predictions from the ANN by returning a random prediction vector.
        - Uses torch for ANN training and prediction, and clears CUDA cache at the end.
    """

    if ml_grid_object.verbose >= 11:
        print("get_y_pred_ann_torch_weighting")
        print(best)
        print("len best", len(best))

    y_test = ml_grid_object.y_test
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    y_train = ml_grid_object.y_train
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test

    model_train_time_warning_threshold = 15
    start = time.time()

    target_ensemble = best[0]

    # Choose prediction target based on valid parameter
    if valid:
        if ml_grid_object.verbose >= 1:
            print(f"Evaluating ANN weighted ensemble on validation set: {valid}")
        x_test = X_test_orig.copy()
        y_test_ann = y_test_orig.copy()
    else:
        if ml_grid_object.verbose >= 1:
            print(f"Evaluating ANN weighted ensemble on test set: {valid}")
        x_test = X_test.copy()
        y_test_ann = y_test.copy()

    # Always fit models and generate predictions for training the ANN weighting
    prediction_array = []

    for i in range(0, len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        if type(target_ensemble[i][1]) is not BinaryClassification:
            model = target_ensemble[i][1]
            if ml_grid_object.verbose >= 2:
                print(f"Fitting model {i+1}")
            model.fit(X_train[feature_columns], y_train)

            prediction_array.append(model.predict(X_train[feature_columns]))
        else:
            test_data = TestData(torch.FloatTensor(X_train[feature_columns].values))
            test_loader = DataLoader(dataset=test_data, batch_size=1)

            device = torch.device("cpu")
            model = target_ensemble[i][1]
            model.to(device)
            y_hat = model(test_data.X_data.to(device))

            y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

            y_hat = y_hat.astype(int).flatten()
            prediction_array.append(y_hat)

    prediction_matrix_X_train = np.matrix(prediction_array)
    prediction_matrix_X_train = prediction_matrix_X_train.astype(float)
    prediction_matrix_raw_X_train = prediction_matrix_X_train

    X_prediction_matrix_raw_X_train = prediction_matrix_raw_X_train.T

    # Now generate predictions on the target dataset (test or validation)
    prediction_array = []
    for i in range(0, len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        existing_columns = [
            col
            for col in feature_columns
            if col in X_train.columns and col in x_test.columns
        ]
        feature_columns = existing_columns.copy()

        if type(target_ensemble[i][1]) is not BinaryClassification:
            prediction_array.append(
                target_ensemble[i][1].predict(x_test[feature_columns])
            )
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

    prediction_matrix_X_test = np.matrix(prediction_array)
    prediction_matrix_X_test = prediction_matrix_X_test.astype(float)
    prediction_matrix_raw_X_test = prediction_matrix_X_test

    prediction_matrix_raw_X_test = prediction_matrix_raw_X_test.T
    test_data = TestData(torch.FloatTensor(prediction_matrix_raw_X_test))

    train_data = TrainData(
        torch.FloatTensor(X_prediction_matrix_raw_X_train),
        torch.FloatTensor(np.array(y_train)),
    )

    y_pred_unweighted = []
    for i in range(0, len(prediction_array[0])):
        y_pred_unweighted.append(round(np.mean(prediction_matrix_raw_X_test.T[:, i])))

    auc = metrics.roc_auc_score(y_test_ann, y_pred_unweighted)

    mccscore_unweighted = matthews_corrcoef(y_test_ann, y_pred_unweighted)

    y_pred_ensemble = train_ann_weight(
        X_prediction_matrix_raw_X_train.shape[1],
        int(X_prediction_matrix_raw_X_train.shape[0]),
        train_data,
        test_data,
    )

    if any(np.isnan(y_pred_ensemble)):
        print("Torch model nan, returning random y pred vector")
        random_y_pred_vector = (
            np.random.choice(
                a=[False, True],
                size=(
                    len(
                        y_test_ann,
                    )
                ),
            )
        ).astype(int)
        y_pred = random_y_pred_vector
        y_pred_ensemble = random_y_pred_vector

    auc_score_weighted = metrics.roc_auc_score(y_test_ann, y_pred_ensemble)
    mccscore_weighted = matthews_corrcoef(y_test_ann, y_pred_ensemble)

    auc_score_weighted = round(metrics.roc_auc_score(y_test_ann, y_pred_ensemble), 4)
    if ml_grid_object.verbose >= 5:
        print("ANN unweighted ensemble AUC: ", auc)
        print("ANN weighted   ensemble AUC: ", auc_score_weighted)
        print("ANN weighted   ensemble AUC difference: ", auc_score_weighted - auc)
        print("ANN unweighted ensemble MCC: ", mccscore_unweighted)
        print("ANN weighted   ensemble MCC: ", mccscore_weighted)

    end = time.time()
    model_train_time = int(end - start)
    torch.cuda.empty_cache()

    return y_pred_ensemble
