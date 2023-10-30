import numpy as np
import torch
import numpy
from torch.utils.data import Dataset, DataLoader

from numpy.linalg import norm

import torch.nn as nn


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


def get_weighted_ensemble_prediction_de_y_pred_valid(
    best, weights, ml_grid_object, valid=False
):
    """Return the weighted prediction vector"""

    # global get_weighted_ensemble_prediction

    # Need to calculate weights from x_train (weights are passed in already)
    # then call each model in the ensemble and have them predict on the x_test_orig validation
    # Then apply the weights learned from training set

    # Get prediction matrix:

    round_v = np.vectorize(round)

    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train

    target_ensemble = best[0]
    if valid:
        print("Evaluating weighted ensemble on validation set")
        x_test = X_test_orig.copy()
        y_test = y_test_orig.copy()

        prediction_array = []

        for i in range(0, len(target_ensemble)):
            # For model i, predict it's x_test
            feature_columns = target_ensemble[i][2]

            model = target_ensemble[i][1]

            #         is evaluate calling model text also fitting?
            if type(target_ensemble[i][1]) is not BinaryClassification:
                model.fit(X_train[feature_columns], y_train)
                prediction_array.append(
                    model.predict(x_test[feature_columns])
                )  # Target for resolving GPU/torch model types

            else:
                print(
                    "get_weighted_ensemble_prediction_de_y_pred_valid, handling torch model prediction"
                )

                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                device = torch.device("cpu")
                #             model = target_ensemble[i][1]                                     #Has this model already been trained?
                model.to(device)
                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()

                print("numpy.isnan(y_hat).any()", numpy.isnan(y_hat).any())
                if numpy.isnan(y_hat).any():
                    print(
                        "Returning dummy random yhat vector for torch pred, nan found"
                    )
                    y_hat = np.random.choice(
                        a=[False, True],
                        size=(
                            len(
                                y_hat,
                            )
                        ),
                    )

                prediction_array.append(y_hat)
    else:
        prediction_array = []

        for i in range(0, len(target_ensemble)):
            # print(target_ensemble[i][5].shape)
            prediction_array.append(target_ensemble[i][5])

    prediction_matrix = np.matrix(prediction_array)
    prediction_matrix = prediction_matrix.astype(float)
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

    torch.cuda.empty_cache()  # exp

    return y_pred_weighted
