import torch.nn as nn
import torch.nn.functional as F

class DCMLPr3(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다. 
    def __init__(self, input_size, output_size, hidden_sizes):
        super(DCMLPr3, self).__init__()
        self.fc1_2 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2_3 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2_4 = nn.Linear(hidden_sizes[0], hidden_sizes[2])
        self.fc2_5 = nn.Linear(hidden_sizes[0], hidden_sizes[3])
        self.fc3_4 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3_5 = nn.Linear(hidden_sizes[1], hidden_sizes[3])
        self.fc4_5 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5_6 = nn.Linear(hidden_sizes[3], output_size)

    def forward(self,x):
        x = self.fc1_2(x)
        x = F.relu(x)

        x2_4 = self.fc2_4(x)
        x2_5 = self.fc2_5(x)

        x = self.fc2_3(x)
        x = F.relu(x)

        x3_5 = self.fc3_5(x)

        x = self.fc3_4(x)+x2_4
        x = F.relu(x)

        x = self.fc4_5(x)+x2_5+x3_5
        x = F.relu(x)

        out = self.fc5_6(x)

        out = F.log_softmax(out, dim=1)
        return out
