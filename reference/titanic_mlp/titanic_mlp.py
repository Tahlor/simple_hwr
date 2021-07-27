import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import sys

""" Test MLP with Titanic Data
"""



class SingleHidden(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SingleHidden, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.out = nn.Linear(H, D_out)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.linear1(x)
        hidden_relu = nn.functional.relu(x)
        y_pred = self.out(x)
        return y_pred

class ReadData(Dataset):
    def __init__(self, file_name):
        self.dataframe = pd.read_csv(file_name)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        val = self.dataframe.loc[index]
        val = val.values.reshape(val.shape[0], 1)
        label = np.asarray(val[0])  # , dtype=np.float32)
        features = np.asarray(val[1:])  # , dtype=np.float32)
        data_obj = {
            'label': label,
            'features': features
        }
        return data_obj


N, D_in, H, D_out = 4, 784, 100, 10

dataset = ReadData("./data/mnist/train.csv")
dataloader = DataLoader(dataset=dataset, batch_size=4)

_brain = SingleHidden(D_in=D_in, H=H, D_out=D_out)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
optimizer = optim.SGD(_brain.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    for obj in dataloader:
        features = obj['features']
        labels = obj['label'].view(-1)
        optimizer.zero_grad()
        features = features.float()
        output = _brain(features)
        loss = criterion(output, labels)

        # Compute gradients for all backprop elements.
        loss.backward()

        optimizer.step()

print('Finished')

