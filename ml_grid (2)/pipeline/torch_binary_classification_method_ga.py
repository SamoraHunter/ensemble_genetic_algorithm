
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


## train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)



## test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    


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
    
    