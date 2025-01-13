import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.datasets as dsets

import torch.nn.utils.prune as prune
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from torch.utils.data import random_split

class MLPr2(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다. 
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLPr2, self).__init__()
        self.fc1_2 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2_3_i = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2_3_ii = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3_4_i = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3_4_ii = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4_5 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5_6 = nn.Linear(hidden_sizes[3], output_size)

    def forward(self,x):
        x = self.fc1_2(x)
        x = F.relu(x)
        x = self.fc2_3_i(x)+self.fc2_3_ii(x)
        x = F.relu(x)

        x = self.fc3_4_i(x)+self.fc3_4_ii(x)
        x = F.relu(x)

        x = self.fc4_5(x)
        x = F.relu(x)

        out = self.fc5_6(x)

        out = F.log_softmax(out, dim=1)
        return out