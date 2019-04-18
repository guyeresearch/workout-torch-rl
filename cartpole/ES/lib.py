import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn
# import random
import numpy as np
import functools

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



class MultiRand():
    def __init__(self,seeds,noise_bank,vec_dim):
        self.noise_bank = noise_bank
        self.states = []
        self.vec_dim = vec_dim
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
            index = torch.randint(0,self.noise_bank.shape[0],
                (self.vec_dim,))
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

def get_utils(lam):
    l = [x+1 for x in range(lam)]
    f = lambda i: np.max([0, np.log(lam/2+1)-np.log(i)])
    t = np.array([f(x) for x in l])
    divisor = np.sum(t)
    final = t/divisor - 1/lam
    return final