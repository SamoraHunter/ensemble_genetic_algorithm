"""Methods for training an Artificial Neural Network (ANN) to determine ensemble weights."""

import itertools
import random
import time
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
    binary_acc,
    get_free_gpu,
)


def get_y_pred_ann_torch_weighting(
    best: List, ml_grid_object: Any, valid: bool = False
) -> np.ndarray:
    """Generates ANN-weighted ensemble predictions.

    This function takes an ensemble of models and uses an Artificial Neural
    Network (ANN) to learn optimal weights for combining their predictions.
    If `valid` is True, it fits models on training data, generates predictions
    on both training and validation sets, then trains the weighting ANN.
    If `valid` is False, it uses pre-computed predictions from the `best`
    configuration to train the ANN.

    Args:
        best: A list containing the ensemble configuration. The first element
            (`best[0]`) is a list of tuples, each representing a model and its
            metadata (model object, feature columns, and optional pre-computed
            predictions).
        ml_grid_object: An object containing data splits (`X_train`, `y_train`,
            `X_test_orig`, `y_test_orig`, etc.) and configuration like `verbose`.
        valid: If True, refits models and predicts on the validation set.
            If False, uses pre-computed predictions. Defaults to False.

    Returns:
        The final ensemble predictions as a 1D NumPy array, weighted by the
        trained ANN.

    Notes:
        - Handles both scikit-learn and custom PyTorch `BinaryClassification` models.
        - If the ANN produces NaN predictions, a random prediction vector is
          returned as a fallback.
        - Computes and prints AUC and MCC scores for both unweighted and
          ANN-weighted ensembles if verbosity is high.
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

    y_test_ann = y_test.copy()

    start = time.time()

    target_ensemble = best[0]
    if valid:
        if ml_grid_object.verbose >= 1:
            print(f"Evaluating ANN weighted ensemble on validation set: {valid}")
        x_test = X_test_orig.copy()
        y_test_ann = y_test_orig.copy()

        prediction_array = []

        for i in range(0, len(target_ensemble)):
            feature_columns = list(target_ensemble[i][2])

            if not isinstance(target_ensemble[i][1], BinaryClassification):
                model = target_ensemble[i][1]
                if ml_grid_object.verbose >= 2:
                    print(f"Fitting model {i+1}")
                model.fit(X_train[feature_columns], y_train)

                prediction_array.append(model.predict(X_train[feature_columns]))
            else:
                test_data = TestData(torch.FloatTensor(X_train[feature_columns].values))
                
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

        prediction_array = []
        for i in range(0, len(target_ensemble)):
            feature_columns = list(target_ensemble[i][2])
            if not isinstance(target_ensemble[i][1], BinaryClassification):
                prediction_array.append(
                    target_ensemble[i][1].predict(x_test[feature_columns])
                )
            else:
                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))

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

    elif not valid:
        if ml_grid_object.verbose >= 11:
            print("Training ANN weighted ensemble on full data")
            print(target_ensemble)
        prediction_array = []
        for i in tqdm(range(0, len(target_ensemble))):
            prediction_array.append(target_ensemble[i][5])

        prediction_matrix_X_train = np.matrix(prediction_array)
        prediction_matrix_X_train = prediction_matrix_X_train.astype(float)
        prediction_matrix_raw_X_train = prediction_matrix_X_train

        X_prediction_matrix_raw_X_train = prediction_matrix_raw_X_train.T
        test_data = TestData(torch.FloatTensor(X_prediction_matrix_raw_X_train))

        prediction_matrix_raw_X_test = X_prediction_matrix_raw_X_train

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
    if ml_grid_object.verbose >= 5:
        print("Model training time: ", model_train_time)
    torch.cuda.empty_cache()

    return y_pred_ensemble


def train_ann_weight(
    input_shape: int, hidden_layer_size: int, train_data: TrainData, test_data: TestData
) -> np.ndarray:
    """Trains an ANN to find ensemble weights and returns predictions.

    This function initializes a `BinaryClassification` neural network, trains
    it on the provided training data (which consists of predictions from base
    models), and then uses the trained network to generate final predictions
    on the test data. Hyperparameters for the network are chosen randomly
    from a predefined search space.

    Args:
        input_shape: The number of input features for the ANN (i.e., the
            number of models in the ensemble).
        hidden_layer_size: The batch size for training the ANN.
        train_data: A `TrainData` object containing the training features
            (base model predictions) and true labels.
        test_data: A `TestData` object containing the test features (base
            model predictions) for which to generate final predictions.

    Returns:
        A NumPy array of the final predictions from the trained ANN model.
    """
    try:
        free_gpu = str(get_free_gpu())
    except Exception:
        free_gpu = "-1"

    # Initialise global parameter space----------------------------------------------------------------

    parameter_space = {
        "column_length": [input_shape],
        "hidden_layer_size": [hidden_layer_size],
        "deep_layers_1": [2],
        "dropout_val": [0.001],
    }

    additional_grid = {
        "epochs": [20],
        "learning_rate": [0.0001],
    }
    size_test = []
    # Loop over al grid search combinations
    for values in itertools.product(*additional_grid.values()):
        point = dict(zip(additional_grid.keys(), values))
        # merge the general settings
        settings = {**point}
        size_test.append(settings)

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    additional_param_sample = random.choice(size_test)

    additional_param_sample = {}
    for key in additional_grid.keys():
        additional_param_sample[key] = random.choice(additional_grid.get(key))

    device = torch.device(f"cuda:{free_gpu}" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=sample_parameter_space["hidden_layer_size"],
        shuffle=True,
    )

    # fit model with random sample of global parameter space
    model = BinaryClassification(**sample_parameter_space)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=additional_param_sample["learning_rate"]
    )
    model.train()
    for e in range(1, additional_param_sample["epochs"] + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(
            f"Epoch {e:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}"
        )

    y_pred_ensemble = model(test_data.X_data.to(device))

    y_pred_ensemble = (
        torch.round(torch.sigmoid(y_pred_ensemble)).cpu().detach().numpy().flatten()
    )

    return y_pred_ensemble