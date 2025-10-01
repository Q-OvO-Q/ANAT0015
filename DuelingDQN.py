import torch
from torch import nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.Linear(512, 512)
        self.state_values = nn.Linear(512, 1)
        self.advantages = nn.Linear(512, output_dim)


    def forward(self, x):
        x = F.relu6(self.layer1(x))
        x = F.relu6(self.layer2(x))
        x = F.relu6(self.layer3(x))
        x = F.relu6(self.layer4(x))
        x = F.relu6(self.layer5(x))
        state_values = self.state_values(x)
        advantages = self.advantages(x)
        output = state_values + (advantages - torch.max((advantages), dim=1, keepdim=True)[0])
        return output
