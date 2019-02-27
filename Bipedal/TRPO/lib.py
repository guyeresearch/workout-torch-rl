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
        self.fc1 = nn.Linear(dim_in,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,dim_out)


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
        self.fc1 = nn.Linear(dim_in,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,1)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#

class ParamReshape():
    def __init__(self,model):
        self.shapes = [x.shape for x in model.parameters()]
    
    def param2vec(self,params):
        return torch.cat([x.view(-1) for x in params])

    def vec2param(self,vec):
        pointer = 0
        # params = []
        for shape in self.shapes:
            flat_len = np.prod(shape)
            sub = vec[pointer:pointer+flat_len]
            yield sub.view(shape)
            # params.append(sub.view(shape))
            pointer += flat_len
        # return params