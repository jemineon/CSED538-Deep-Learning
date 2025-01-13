import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다. 
    def __init__(self, input_size, output_size, hidden_size):
        super(DeepMLP, self).__init__()
        self.layers = dict()
        self.layers["fc1_2"] = nn.Linear(input_size, hidden_size)
        # self.layers = []
        for i in range(2, 14):
            self.layers[f"fc{i}_{i+1}"] = nn.Linear(hidden_size, hidden_size)

        self.layers["fc14_15"] = nn.Linear(hidden_size, output_size)

        self.layers = nn.ModuleDict(self.layers)

    def forward(self,x):
        for i in range(1, 15):
            layer = self.layers[f"fc{i}_{i+1}"]
            x = layer(x)
            x = F.relu(x)

        x = F.log_softmax(x, dim=1)
        return x