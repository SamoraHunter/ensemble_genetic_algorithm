import numpy as np
import torch
import numpy
from torch.utils.data import Dataset, DataLoader

from numpy.linalg import norm

import itertools
import random
import time

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from scipy import stats

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


def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# should call from import:
class BinaryClassification(nn.Module):
    def __init__(self, column_length, deep_layers_1, batch_size, dropout_val):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(column_length, batch_size)
        self.layer_2 = nn.Linear(batch_size, batch_size)
        layers = []
        for i in range(0, deep_layers_1):
            layers.append(nn.Linear(batch_size, batch_size))

        self.deep_layers = nn.Sequential(*layers)

        self.layer_out = nn.Linear(batch_size, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_val)
        self.batchnorm1 = nn.BatchNorm1d(batch_size)
        self.batchnorm2 = nn.BatchNorm1d(batch_size)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.relu(self.deep_layers(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


## train data
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


# def get_best_y_pred_unweighted(best, ml_grid_object, valid=False):
#     X_test_orig = ml_grid_object.X_test_orig
#     y_test_orig = ml_grid_object.y_test_orig
#     X_train = ml_grid_object.X_train
#     X_test = ml_grid_object.X_test
#     y_train = ml_grid_object.y_train

#     if valid:
#         x_test = X_test_orig.copy()
#         y_test = y_test_orig.copy()

#     prediction_array = []
#     target_ensemble = best[0]
#     if valid:
#         for i in range(0, len(target_ensemble)):
#             feature_columns = list(
#                 target_ensemble[i][2]
#             )  # list(target_ensemble[i][3].columns)

#             if type(target_ensemble[i][1]) is not BinaryClassification:
#                 model = target_ensemble[i][1]

#                 model.fit(X_train[feature_columns], y_train)

#                 prediction_array.append(model.predict(x_test[feature_columns]))

#             else:
#                 test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
#                 test_loader = DataLoader(dataset=test_data, batch_size=1)

#                 device = torch.device("cpu")
#                 model = target_ensemble[i][1]  # Has this model been fitted already?
#                 model.to(device)

#                 # model.fit(X_train, y_train)

#                 y_hat = model(test_data.X_data.to(device))

#                 y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

#                 y_hat = y_hat.astype(int).flatten()
#                 prediction_array.append(y_hat)

#     else:
#         for i in range(0, len(target_ensemble)):
#             y_pred = target_ensemble[i][5]
#             prediction_array.append(y_pred)

#     prediction_matrix = np.matrix(prediction_array)

#     # collapse the mean of each models prediction for each case into a binary label returning a final y_pred composite score from each model
#     y_pred_best = []
#     for i in range(0, len(prediction_array[0])):
#         # y_pred_best.append(round(np.mean(np.matrix(prediction_array)[:,i])))
#         try:
#             y_pred_best.append(
#                 stats.mode(np.matrix(prediction_array)[:, i], keepdims=True)[0][0][0]
#             )
#         except:
#             y_pred_best.append(stats.mode(np.matrix(prediction_array)[:, i])[0][0][0])
#     return y_pred_best


def get_best_y_pred_unweighted(best, ml_grid_object, valid=False):

    if ml_grid_object.verbose >= 1:
        print("get_best_y_pred_unweighted: best:")
        print(best)

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
        for i in range(0, len(target_ensemble)):
            y_pred = target_ensemble[i][5]
            prediction_array.append(y_pred)

    prediction_matrix = np.matrix(prediction_array)

    y_pred_best = []
    for i in range(0, len(prediction_array[0])):
        try:
            y_pred_best.append(
                stats.mode(np.matrix(prediction_array)[:, i], keepdims=True)[0][0][0]
            )
        except:
            y_pred_best.append(stats.mode(np.matrix(prediction_array)[:, i])[0][0][0])
    return y_pred_best
