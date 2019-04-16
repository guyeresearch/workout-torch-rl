import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn
# import random
import numpy as np
import functools

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


class MultiRand():
    def __init__(self,seeds,noise_bank,size):
        self.noise_bank = noise_bank
        self.states = []
        for seed in seeds:
            torch.manual_seed(seed)
            state = torch.get_rng_state()
            self.states.append(state)
    
    def get_noises(self):
        old_states = self.states
        self.states = []
        noises = []
        for state in old_states:
            torch.set_rng_state(state)
            index = torch.randint(self.low,self.high,
                (self.size,))
            noise = self.noise_bank[index]
            noises.append(noise)
            self.states.append(
                torch.get_rng_state()
            )
        return noises

    
class ParamReshape():
    def __init__(self,model):
        self.shapes = [x.shape for x in model.parameters()]
        self.vec_dim = functools.reduce(lambda a,b: np.prod(a)+np.prod(b),
             self.shapes)
        
    
    def param2vec(self,params):
        return torch.cat([x.contiguous().view(-1) 
            for x in params])

    def vec2param(self,vec):
        pointer = 0
        params = []
        for shape in self.shapes:
            flat_len = np.prod(shape)
            sub = vec[pointer:pointer+flat_len]
            # yield sub.view(shape)
            params.append(sub.view(shape))
            pointer += flat_len
        return params