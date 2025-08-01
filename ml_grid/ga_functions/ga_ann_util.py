from io import StringIO
import subprocess
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import norm


class BinaryClassification(nn.Module):
    """
    A PyTorch neural network module for binary classification tasks with customizable depth and regularization.

    Args:
        column_length (int): Number of input features.
        deep_layers_1 (int): Number of additional fully connected hidden layers after the initial two layers.
        batch_size (int): Number of neurons in each hidden layer.
        dropout_val (float): Dropout probability for regularization.

    Attributes:
        layer_1 (nn.Linear): First fully connected layer.
        layer_2 (nn.Linear): Second fully connected layer.
        deep_layers (nn.Sequential): Sequence of additional fully connected layers.
        layer_out (nn.Linear): Output layer producing a single value.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer for regularization.
        batchnorm1 (nn.BatchNorm1d): Batch normalization after the first layer.
        batchnorm2 (nn.BatchNorm1d): Batch normalization after the deep layers.

    Forward Pass:
        Applies a sequence of linear transformations, activations, batch normalizations, dropout, and outputs a single value for binary classification.
        Input shape: (batch_size, column_length)
        Output shape: (batch_size, 1)
    """

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
        self.batchnorm1 = torch.nn.BatchNorm1d(batch_size)
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


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    torch.cuda.empty_cache()

    return acc


class TrainData(Dataset):
    """
    A custom PyTorch Dataset for handling training data.

    Args:
        X_data (array-like or torch.Tensor): Input features for the dataset.
        y_data (array-like or torch.Tensor): Target labels corresponding to the input features.

    Methods:
        __getitem__(index): Returns a tuple (X_data[index], y_data[index]) for the given index.
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class TestData(Dataset):
    """
    A custom Dataset class for handling test data in PyTorch.

    Args:
        X_data (array-like or torch.Tensor): The input data to be wrapped by the dataset.

    Methods:
        __getitem__(index): Returns the data sample at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def get_free_gpu(ml_grid_object=None):
    """
    Returns the index of the GPU with the most free memory.

    This function queries the available GPUs using `nvidia-smi` and returns the index of the GPU
    with the highest amount of free memory. If an `ml_grid_object` is provided and has a `verbose`
    attribute, the function will print additional information if verbosity is set to 6 or higher.

    Args:
        ml_grid_object (optional): An object with a `verbose` attribute to control verbosity. Default is None.

    Returns:
        int: The index of the GPU with the most free memory, or -1 if an error occurs or no GPU is available.
    """
    verbosity = 0
    if ml_grid_object is not None:
        verbosity = ml_grid_object.verbose

    try:
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
        )
        gpu_df = pd.read_csv(
            StringIO(gpu_stats.decode("utf-8")),
            names=["memory.used", "memory.free"],
            skiprows=1,
        )
        # print('GPU usage:\n{}'.format(gpu_df))
        gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: x.rstrip(" [MiB]"))
        idx = gpu_df["memory.free"].astype(int).idxmax()

        if verbosity >= 6:
            print(
                "Returning GPU{} with {} free MiB".format(
                    idx, gpu_df.iloc[idx]["memory.free"]
                )
            )
        return int(idx)
    except Exception as e:
        # print("Error:", e)
        return -1


def normalize(weights):
    """
    Normalizes a vector of weights using the L1 norm (sum of absolute values).

    If the input vector consists entirely of zeros, it is returned unchanged.
    Otherwise, each element is divided by the L1 norm so that the sum of the absolute values equals 1.

    Args:
        weights (numpy.ndarray): The vector of weights to normalize.

    Returns:
        numpy.ndarray: The normalized weight vector with L1 norm equal to 1, or the original vector if all elements are zero.
    """
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result
