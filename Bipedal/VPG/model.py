import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn

class Critic(nn.Module):
    def __init__(self,dim_state_in, dim_action_in):
        super(Critic, self).__init__()

        self.bn1 = nn.BatchNorm1d(dim_state_in)
        self.fc1 = nn.Linear(dim_state_in,240)
        self.dp1 = nn.Dropout()
        self.fc2 = nn.Linear(240,120)
        self.dp2 = nn.Dropout()
        self.fc3 = nn.Linear(120,1)

        self.fc_action = nn.Linear(dim_action_in,120)

    def forward(self,x,x_action):
        x = F.relu(self.fc1(self.bn1(x)))
        x = self.dp1(x)
        x = F.relu(self.fc2(x)+self.fc_action(x_action))
        x = self.dp2(x)
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(Actor, self).__init__()
        self.dim_in = dim_in

        self.bn1 = nn.BatchNorm1d(dim_in)
        self.fc1 = nn.Linear(dim_in,240)
        self.dp1 = nn.Dropout()
        self.fc2 = nn.Linear(240,120)
        self.dp2 = nn.Dropout()
        self.fc3 = nn.Linear(120,dim_out)

    def forward(self,x):
        x = F.relu(self.fc1(self.bn1(x)))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.dp2(x)
        x = F.tanh(self.fc3(x))
        return x
