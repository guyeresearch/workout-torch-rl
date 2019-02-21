import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn

# simple three layers
class Policy(nn.Module):
    def __init__(self,dim_in, dim_out, hidden=100):
        super().__init__()
        self.fc1 = nn.Linear(dim_in,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,dim_out)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
#
#
class Q(nn.Module):
    def __init__(self,dim_in, action_dim, hidden=100):
        super().__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(dim_in,hidden)
        self.fc2 = nn.Linear(hidden+action_dim,hidden)
        self.fc3 = nn.Linear(hidden,action_dim)


    def forward(self,x,a):
        x = torch.relu(self.fc1(x))
        x = torch.cat((x,a))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#