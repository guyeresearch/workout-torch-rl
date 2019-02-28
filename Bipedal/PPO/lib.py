import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn
import numpy as np

# simple three layers
class Policy(nn.Module):
    def __init__(self,dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,dim_out)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#
#
class Val(nn.Module):
    def __init__(self,dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(dim_in,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,1)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
