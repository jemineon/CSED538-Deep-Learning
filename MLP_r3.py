import torch.nn as nn
import torch.nn.functional as F

class MLPr3(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다. 
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLPr3, self).__init__()
        self.fc1_2 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2_3_i = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2_3_ii = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2_3_iii = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3_4_i = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3_4_ii = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4_5 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5_6 = nn.Linear(hidden_sizes[3], output_size)

    def forward(self,x):
        x = self.fc1_2(x)
        x = F.relu(x)
        x = self.fc2_3_i(x) + self.fc2_3_ii(x) + self.fc2_3_iii(x)
        x = F.relu(x)

        x = self.fc3_4_i(x)+self.fc3_4_ii(x)
        x = F.relu(x)

        x = self.fc4_5(x)
        x = F.relu(x)

        out = self.fc5_6(x)

        out = F.log_softmax(out, dim=1)
        return out


class DeepMLPr3(nn.Module):  # nn.Module은 모든 neural network의 base class라고 한다.
    def __init__(self, input_size, output_size, hidden_size):
        super(DeepMLPr3, self).__init__()
        self.layers = dict()
        self.layers["fc1_2"] = nn.Linear(input_size, hidden_size)
        for i in range(2, 14):
            self.layers[f"fc{i}_{i+1}"] = nn.Linear(hidden_size, hidden_size)

        self.layers["fc14_15"] = nn.Linear(hidden_size, output_size)

        self.layers["fc2_3_ii"] = nn.Linear(hidden_size, hidden_size)
        self.layers["fc2_3_iii"] = nn.Linear(hidden_size, hidden_size)
        self.layers["fc6_7_ii"] = nn.Linear(hidden_size, hidden_size)

        self.layers = nn.ModuleDict(self.layers)

    def forward(self, x):
        x2_3_ii = x2_3_iii = x6_7_ii = 0
        for i in range(1, 14):
            from_ = i
            to = i + 1
            layer = self.layers[f"fc{i}_{i+1}"]

            if from_ == 2:
                x2_3_ii = self.layers["fc2_3_ii"](x)
                x2_3_iii = self.layers["fc2_3_iii"](x)
            elif from_ == 6:
                x6_7_ii = self.layers["fc6_7_ii"](x)

            x = layer(x)

            if to == 3:
                x = x + x2_3_ii + x2_3_iii
            elif to == 7:
                x = x + x6_7_ii

            x = F.relu(x)

        out = self.layers["fc14_15"](x)
        out = F.log_softmax(out, dim=1)
        return out
