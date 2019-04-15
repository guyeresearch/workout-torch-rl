import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn
import random

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
    def __init__(self,seeds,low,high,size):
        self.low = low
        self.high = high
        self.size = size
        self.states = []
        for seed in seeds:
            torch.manual_seed(seed)
            state = torch.get_rng_state()
            self.states.append(state)
    
    def get_rand_indices(self):
        old_states = self.states
        self.states = []
        indices = []
        for state in old_states:
            torch.set_rng_state(state)
            index = torch.randint(self.low,self.high,
                (self.size,))
            indices.append(index)
            self.states.append(
                torch.get_rng_state()
            )
        return indices

    
    # def step(self):
