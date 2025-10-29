"""Utility functions for PyTorch-based neural networks in the genetic algorithm."""

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
    """A PyTorch neural network for binary classification.

    Attributes:
        layer_1: First fully connected layer.
        layer_2: Second fully connected layer.
        deep_layers: Additional hidden layers.
        layer_out: Output layer.
        relu: ReLU activation function.
        dropout: Dropout layer.
        batchnorm1: Batch normalization for the first layer.
        batchnorm2: Batch normalization for the second set of layers.
    """

    def __init__(self, column_length: int, deep_layers_1: int, hidden_layer_size: int, dropout_val: float):
        """Initializes the BinaryClassification model.

        Args:
            column_length: The number of input features.
            deep_layers_1: The number of additional deep layers.
            hidden_layer_size: The size of the hidden layers.
            dropout_val: The dropout probability.
        """
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
            inputs: The input tensor.

        Returns:
            The output tensor.
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
    """Calculates the accuracy for binary classification.

    Args:
        y_pred: The predicted values.
        y_test: The true values.

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
    """Custom PyTorch Dataset for training data."""

    def __init__(self, X_data: torch.Tensor, y_data: torch.Tensor):
        """Initializes the training dataset.

        Args:
            X_data: The input features.
            y_data: The target labels.
        """
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a single sample of the data."""
        return self.X_data[index], self.y_data[index]

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.X_data)


class TestData(Dataset):
    """Custom PyTorch Dataset for test data."""

    def __init__(self, X_data: torch.Tensor):
        """Initializes the test dataset.

        Args:
            X_data: The input features.
        """
        self.X_data = X_data

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns a single sample of the data."""
        return self.X_data[index]

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.X_data)


def get_free_gpu(ml_grid_object: Optional[Any] = None) -> int:
    """Gets the index of the GPU with the most free memory.

    Args:
        ml_grid_object: Optional ml_grid object for verbose output.

    Returns:
        The index of the GPU with the most free memory, or -1 if an error occurs.
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
        gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: x.rstrip(" [MiB]"))
        idx = gpu_df["memory.free"].astype(int).idxmax()

        if verbosity >= 6:
            print(
                f"Returning GPU{idx} with {gpu_df.iloc[idx]['memory.free']} free MiB"
            )
        return int(idx)
    except Exception:
        return -1


def normalize(weights: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array.

    Args:
        weights: The array to normalize.

    Returns:
        The normalized array.
    """
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result