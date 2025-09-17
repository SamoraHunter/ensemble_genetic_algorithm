from io import StringIO
import subprocess
from typing import Any, Optional, Tuple
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import norm
import numpy as np


class BinaryClassification(nn.Module):
    """
    A PyTorch neural network module for binary classification tasks with customizable depth and regularization.

    Args:
        column_length: Number of input features.
        deep_layers_1: Number of additional fully connected hidden layers
            after the initial two layers.
        hidden_layer_size: Number of neurons in each hidden layer.
        dropout_val: Dropout probability for regularization.

    Attributes:
        layer_1: First fully connected layer.
        layer_2: Second fully connected layer.
        deep_layers: Sequence of additional fully connected layers.
        layer_out: Output layer producing a single value.
        relu: ReLU activation function.
        dropout: Dropout layer for regularization.
        batchnorm1: Batch normalization after the first layer.
        batchnorm2: Batch normalization after the deep layers.
    """

    def __init__(self, column_length: int, deep_layers_1: int, hidden_layer_size: int, dropout_val: float):
        super(BinaryClassification, self).__init__()
        self.layer_1: nn.Linear = nn.Linear(column_length, hidden_layer_size)
        self.layer_2: nn.Linear = nn.Linear(hidden_layer_size, hidden_layer_size)
        layers = []
        for i in range(0, deep_layers_1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))

        self.deep_layers: nn.Sequential = nn.Sequential(*layers)

        self.layer_out: nn.Linear = nn.Linear(hidden_layer_size, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_val)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_layer_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            inputs: Input tensor of shape (batch_size, column_length).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.relu(self.deep_layers(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def binary_acc(y_pred: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
    """Calculates binary classification accuracy.

    Args:
        y_pred: The predicted values from the model.
        y_test: The ground truth labels.

    Returns:
        The accuracy as a percentage.
    """
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
        X_data: Input features for the dataset.
        y_data: Target labels corresponding to the input features.
    
    Attributes:
        X_data (torch.Tensor): Input features for the dataset.
        y_data (torch.Tensor): Target labels corresponding to the input features.
    """

    def __init__(self, X_data: torch.Tensor, y_data: torch.Tensor):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_data[index], self.y_data[index]

    def __len__(self) -> int:
        return len(self.X_data)


## test data
class TestData(Dataset):
    """
    A custom Dataset class for handling test data in PyTorch.

    Args:
        X_data: The input data to be wrapped by the dataset.
    
    Attributes:
        X_data (torch.Tensor): The input data.
    """

    def __init__(self, X_data: torch.Tensor):
        self.X_data = X_data

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.X_data[index]

    def __len__(self) -> int:
        return len(self.X_data)


def get_free_gpu(ml_grid_object: Optional[Any] = None) -> int:
    """
    Returns the index of the GPU with the most free memory.

    This function queries the available GPUs using `nvidia-smi` and returns
    the index of the GPU with the highest amount of free memory. If an
    `ml_grid_object` is provided and has a `verbose` attribute, the function
    will print additional information if verbosity is set to 6 or higher.

    Args:
        ml_grid_object: An object with a `verbose` attribute to control
            verbosity. Defaults to None.

    Returns:
        The index of the GPU with the most free memory, or -1 if an error
        occurs or no GPU is available.
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


def normalize(weights: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector of weights using the L1 norm (sum of absolute values).

    If the input vector consists entirely of zeros, it is returned unchanged.
    Otherwise, each element is divided by the L1 norm so that the sum of the
    absolute values equals 1.

    Args:
        weights: The vector of weights to normalize.

    Returns:
        The normalized weight vector with L1 norm equal to 1, or the
        original vector if all elements are zero.
    """
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result
