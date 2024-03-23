from io import StringIO
import itertools
import random
import subprocess
import time
import pandas as pd
import torchmetrics

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

from sklearn import datasets, feature_selection, linear_model, metrics, svm, tree

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

import torch.optim as optim

from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
    binary_acc,
    get_free_gpu,
)


# should call from import:


# def get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=False):
#     y_test = ml_grid_object.y_test
#     X_test_orig = ml_grid_object.X_test_orig
#     y_test_orig = ml_grid_object.y_test_orig
#     y_train = ml_grid_object.y_train
#     X_train = ml_grid_object.X_train

#     y_test_ann = y_test.copy()

#     model_train_time_warning_threshold = 15
#     start = time.time()

#     target_ensemble = best[0]
#     # Get prediction matrix
#     # print('Get prediction matrix')
#     if valid:
#         print(f"get_y_pred_ann_torch_weighting {valid}")
#         x_test = X_test_orig.copy()
#         y_test_ann = y_test_orig.copy()

#         prediction_array = []

#         for i in tqdm(range(0, len(target_ensemble))):
#             # For model i, predict it's x_test
#             feature_columns = list(
#                 target_ensemble[i][2]
#             )  # get features model was trained on

#             if type(target_ensemble[i][1]) is not BinaryClassification:
#                 model = target_ensemble[i][1]

#                 model.fit(X_train[feature_columns], y_train)

#                 prediction_array.append(
#                     model.predict(X_train[feature_columns])
#                 )  # Use model to predict x train

#             else:
#                 test_data = TestData(torch.FloatTensor(X_train[feature_columns].values))
#                 test_loader = DataLoader(dataset=test_data, batch_size=1)

#                 device = torch.device("cpu")
#                 model = target_ensemble[i][1]
#                 model.to(device)  # Has this model been fitted??
#                 y_hat = model(test_data.X_data.to(device))

#                 y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

#                 y_hat = y_hat.astype(int).flatten()
#                 prediction_array.append(y_hat)

#         prediction_matrix_X_train = np.matrix(prediction_array)
#         prediction_matrix_X_train = prediction_matrix_X_train.astype(float)
#         prediction_matrix_raw_X_train = (
#             prediction_matrix_X_train  # Store predictions from x_train into matrix
#         )

#         X_prediction_matrix_raw_X_train = prediction_matrix_raw_X_train.T

#         # Produce test results for valid
#         prediction_array = []
#         for i in tqdm(range(0, len(target_ensemble))):
#             feature_columns = list(target_ensemble[i][2])
#             if type(target_ensemble[i][1]) is not BinaryClassification:
#                 prediction_array.append(
#                     target_ensemble[i][1].predict(x_test[feature_columns])
#                 )  # Generate predictions from stored models on validset

#             else:
#                 test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
#                 test_loader = DataLoader(dataset=test_data, batch_size=1)

#                 device = torch.device("cpu")
#                 model = target_ensemble[i][1]
#                 model.to(device)
#                 y_hat = model(test_data.X_data.to(device))

#                 y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

#                 y_hat = y_hat.astype(int).flatten()
#                 prediction_array.append(y_hat)

#         prediction_matrix_X_test = np.matrix(prediction_array)
#         prediction_matrix_X_test = prediction_matrix_X_test.astype(float)
#         prediction_matrix_raw_X_test = prediction_matrix_X_test

#         prediction_matrix_raw_X_test = (
#             prediction_matrix_raw_X_test.T
#         )  # Transpose predictions into columns for each model. X >>y
#         test_data = TestData(torch.FloatTensor(prediction_matrix_raw_X_test))

#     elif valid == False:
#         # Make predictions on xtrain and y train, feed results into nn to learn weights to map ensemble to true. Apply nn to test ensemble preds
#         prediction_array = []
#         for i in tqdm(range(0, len(target_ensemble))):
#             prediction_array.append(
#                 target_ensemble[i][5]
#             )  # Get stored y_pred from x_test (non validation set)

#         prediction_matrix_X_train = np.matrix(prediction_array)
#         prediction_matrix_X_train = prediction_matrix_X_train.astype(float)
#         prediction_matrix_raw_X_train = prediction_matrix_X_train

#         X_prediction_matrix_raw_X_train = (
#             prediction_matrix_raw_X_train.T
#         )  # transpose to matrix, columns are each model yhat vector

#         test_data = TestData(
#             torch.FloatTensor(X_prediction_matrix_raw_X_train)
#         )  # set test data to train set, only learn weights from training
#         # y_test = y_train.copy()
#         prediction_matrix_raw_X_test = X_prediction_matrix_raw_X_train

#     train_data = TrainData(
#         torch.FloatTensor(X_prediction_matrix_raw_X_train),
#         torch.FloatTensor(np.array(y_train)),
#     )  # data set to learn weights for x_train model preds to y_train labels

#     # print(len(prediction_array[0]))
#     # print(prediction_matrix_raw_X_test.shape)

#     y_pred_unweighted = []
#     for i in range(0, len(prediction_array[0])):
#         y_pred_unweighted.append(round(np.mean(prediction_matrix_raw_X_test.T[:, i])))

#     auc = metrics.roc_auc_score(y_test_ann, y_pred_unweighted)

#     mccscore_unweighted = matthews_corrcoef(y_test_ann, y_pred_unweighted)

#     y_pred_ensemble = train_ann_weight(
#         X_prediction_matrix_raw_X_train.shape[1],
#         int(X_prediction_matrix_raw_X_train.shape[0]),
#         train_data,
#         test_data,
#     )

#     # print("Ensemble ANN weighting training AUC: ", auc_score_weighted)

#     if any(np.isnan(y_pred_ensemble)):
#         print("Torch model nan, returning random y pred vector")
#         # zero_vector = [x for x in range(0, len(y_pred))]
#         # y_pred = zero_vector
#         random_y_pred_vector = (
#             np.random.choice(
#                 a=[False, True],
#                 size=(
#                     len(
#                         y_test_ann,
#                     )
#                 ),
#             )
#         ).astype(int)
#         y_pred = random_y_pred_vector
#         y_pred_ensemble = random_y_pred_vector
#     else:
#         # plot_auc(y_hat, f"Deep ANN Torch {para_str}")
#         pass

#     auc_score_weighted = metrics.roc_auc_score(y_test_ann, y_pred_ensemble)

#     mccscore_weighted = matthews_corrcoef(y_test_ann, y_pred_ensemble)

#     auc_score_weighted = round(metrics.roc_auc_score(y_test_ann, y_pred_ensemble), 4)
#     print("ANN unweighted ensemble AUC: ", auc)
#     print("ANN weighted   ensemble AUC: ", auc_score_weighted)
#     print("ANN weighted   ensemble AUC difference: ", auc_score_weighted - auc)
#     print("ANN unweighted ensemble MCC: ", mccscore_unweighted)
#     print("ANN weighted   ensemble MCC: ", mccscore_weighted)

#     # score = (1-de.fun)
#     # optimal_weights = de.x

#     end = time.time()
#     model_train_time = int(end - start)
#     #
#     # print(len(y_pred_ensemble))
#     torch.cuda.empty_cache()  # exp

#     return y_pred_ensemble


def get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=False):

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

    model_train_time_warning_threshold = 15
    start = time.time()

    target_ensemble = best[0]
    if valid:
        if ml_grid_object.verbose >= 1:
            print(f"Evaluating ANN weighted ensemble on validation set: {valid}")
        x_test = X_test_orig.copy()
        y_test_ann = y_test_orig.copy()

        prediction_array = []

        for i in tqdm(range(0, len(target_ensemble))):
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

        prediction_array = []
        for i in tqdm(range(0, len(target_ensemble))):
            feature_columns = list(target_ensemble[i][2])
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

        prediction_matrix_raw_X_test = X_prediction_matrix_raw_X_train  # missing line?

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
    print("ANN unweighted ensemble AUC: ", auc)
    print("ANN weighted   ensemble AUC: ", auc_score_weighted)
    print("ANN weighted   ensemble AUC difference: ", auc_score_weighted - auc)
    print("ANN unweighted ensemble MCC: ", mccscore_unweighted)
    print("ANN weighted   ensemble MCC: ", mccscore_weighted)

    end = time.time()
    model_train_time = int(end - start)
    torch.cuda.empty_cache()

    return y_pred_ensemble


def train_ann_weight(input_shape, batch_size, train_data, test_data):

    try:
        free_gpu = str(get_free_gpu())
    except:
        free_gpu = "-1"

    # Initialise global parameter space----------------------------------------------------------------

    parameter_space = {
        "column_length": [input_shape],
        #'epochs': [50, 200],
        "batch_size": [
            batch_size
        ],  # ,int(X_train.shape[0]/100), int(X_train.shape[0]/200)],
        #'learning_rate': lr_space,
        #'learning_rate': [0.1, 0.001, 0.0005, 0.0001],
        "deep_layers_1": [2],
        "dropout_val": [0.001],
    }

    additional_grid = {
        "epochs": [20],
        #'epochs':[100],
        "learning_rate": [0.0001],
    }
    size_test = []
    # Loop over al grid search combinations
    for values in itertools.product(*additional_grid.values()):
        point = dict(zip(additional_grid.keys(), values))
        # merge the general settings
        settings = {**point}
        # print(settings)
        size_test.append(settings)

    # print(len(size_test))

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    additional_param_sample = random.choice(size_test)

    additional_param_sample = {}
    for key in additional_grid.keys():
        additional_param_sample[key] = random.choice(additional_grid.get(key))

    # print(sample_parameter_space)

    # print(additional_param_sample)

    # os.environ["CUDA_VISIBLE_DEVICES"]=free_gpu

    device = torch.device(f"cuda:{free_gpu}" if torch.cuda.is_available() else "cpu")

    # print(device)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=sample_parameter_space["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    # fit model with random sample of global parameter space
    model = BinaryClassification(**sample_parameter_space)

    model.to(device)
    # print(model)

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
        # print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | AUC: {torchmetrics.functional.auc(y_batch, y_pred, reorder=True)}')

    print(
        f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | AUC: {torchmetrics.functional.auc(y_batch, y_pred, reorder=True)}"
    )

    para_str = (
        str(settings)
        .replace("'", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace("{", "_")
        .replace("}", "_")
        .replace(" ", "_")
    ).replace("__", "_")

    y_pred_ensemble = model(test_data.X_data.to(device))

    y_pred_ensemble = (
        torch.round(torch.sigmoid(y_pred_ensemble)).cpu().detach().numpy().flatten()
    )

    return y_pred_ensemble
