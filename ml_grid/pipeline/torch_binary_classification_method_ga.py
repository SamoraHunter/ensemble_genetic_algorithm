import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple


## train data
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


class BinaryClassification(nn.Module):
    """
    A PyTorch neural network module for binary classification tasks.

    Note:
        The `batch_size` argument is used to define the hidden layer size,
        which might be a misnomer. It does not control the batch size of the
        DataLoader.

    Args:
        column_length: Number of input features.
        deep_layers_1: Number of additional fully connected hidden layers.
        batch_size: Number of neurons in each hidden layer.
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

    def __init__(
        self, column_length: int, deep_layers_1: int, batch_size: int, dropout_val: float
    ):
        super(BinaryClassification, self).__init__()
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