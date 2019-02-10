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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=0)
        return x
#
#
class Val(nn.Module):
    def __init__(self,dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(dim_in,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#



#class Policy(nn.Module):
#    def __init__(self,dim_in, dim_out):
#        super().__init__()
#        self.fc1 = nn.Linear(dim_in,128)
#        self.fc2 = nn.Linear(128,dim_out)
#
#
#    def forward(self,x):
#        x = torch.tanh(self.fc1(x))
#        x = F.softmax(self.fc2(x),dim=0)
#        return x


#class Val(nn.Module):
#    def __init__(self,dim_in):
#        super().__init__()
#
#        self.dim_in = dim_in
#
#        self.fc1 = nn.Linear(dim_in,128)
#        self.fc2 = nn.Linear(128,1)
#
#
#    def forward(self,x):
#        x = torch.tanh(self.fc1(x))
#        x = self.fc2(x)
#        return x
