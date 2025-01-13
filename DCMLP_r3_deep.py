
import torch.nn as nn
import torch.nn.functional as F

class DeepDCMLPr3(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다. 
    def __init__(self, input_size, output_size, hidden_size):
        super(DeepDCMLPr3, self).__init__()
        self.layers = dict()
        self.layers["fc1_2"] = nn.Linear(input_size, hidden_size)
        for i in range(2, 14):
            self.layers[f"fc{i}_{i+1}"] = nn.Linear(hidden_size, hidden_size)
            if i < 13:
                self.layers[f"fc{i}_14"] = nn.Linear(hidden_size, hidden_size)

        self.layers["fc14_15"] = nn.Linear(hidden_size, output_size)
        self.layers = nn.ModuleDict(self.layers)

    def forward(self,x):
        x_14 = 0
        for i in range(1, 14):
            from_ = i
            to = i + 1
            layer = self.layers[f"fc{i}_{i+1}"]
            
            if from_ > 1 and from_ < 13:
                x_14 = x_14 + self.layers[f"fc{from_}_14"](x)

            x = layer(x)

            if to == 14:
                x = x + x_14

            x = F.relu(x)

        out = self.layers["fc14_15"](x)
        out = F.log_softmax(out, dim=1)
        return out
