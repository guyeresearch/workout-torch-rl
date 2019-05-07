import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self,dim_in=1, dim_out=1, hidden=40):
        super().__init__()
        self.fc1 = nn.Linear(dim_in,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,dim_out)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class ParamReshape():
    def __init__(self,model):
        self.shapes = [x.shape for x in model.parameters()]
    
    def param2vec(self,params):
        return torch.cat([x.contiguous().view(-1) 
            for x in params])

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