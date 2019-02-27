import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn

# simple three layers
class Policy(nn.Module):
    def __init__(self,dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,dim_out)

    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Val(nn.Module):
    def __init__(self,dim_in):
        super().__init__()

        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)


    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

## more complex four layers
#class Policy(nn.Module):
#    def __init__(self,dim_in, dim_out):
#        super().__init__()
#        self.fc1 = nn.Linear(dim_in,100)
#        self.fc2 = nn.Linear(100,200)
#        self.fc3 = nn.Linear(200,dim_out)
#
#    def forward(self,x):
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = torch.sigmoid(self.fc3(x))
#        return x
#
#
#class Val(nn.Module):
#    def __init__(self,dim_in):
#        super().__init__()
#
#        self.dim_in = dim_in
#
#        self.fc1 = nn.Linear(dim_in,100)
#        self.fc2 = nn.Linear(100,200)
#        self.fc3 = nn.Linear(200,1)
#
#    def forward(self,x):
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
