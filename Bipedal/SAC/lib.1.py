import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn
import random

# OpenAI param but they did not use clip to force
# logstd being within the range
MIN_LOG_STD = -20
MAX_LOG_STD = 2

# simple three layers
# paper implementation
class Policy(nn.Module):
    def __init__(self,dim_in, dim_out, hidden=100):
        super().__init__()
        self.fc1 = nn.Linear(dim_in,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,dim_out)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # clamp log std
        x_mean = x[..., :int(dim_out/2)]
        x_log_std = x[..., int(dim_out/2):]
        x_log_std = torch.clamp(x_log_std,MIN_LOG_STD,MAX_LOG_STD)
        x_std = torch.exp(x_log_std)
        return x_mean, x_std

class Q(nn.Module):
    def __init__(self,dim_in, action_dim, hidden=100):
        super().__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(dim_in+action_dim,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,1)


    def forward(self,x,a):
        x = torch.cat((x,a),1)
        x = torch.relu(self.fc1(x))
        # is this the right implementation?
        # concatenate along dimension 1
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class V(nn.Module):
    def __init__(self,dim_in, hidden=100):
        super().__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(dim_in,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,1)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Buffer():
    def __init__(self,batch_size=64,max_size=1e6):
        self.pointer = 0
        self.arr = []
        self.max = max_size
        self.batch_size = batch_size
    
    def add(self,trans):
        if self.pointer < self.max:
            self.arr.append(trans)
            self.pointer += 1
        else:
            idx = int(self.pointer % self.max)
            self.arr[idx] = trans
            self.pointer += 1
    
    def add_many(self,trans_set):
        for item in trans_set:
            self.add(item)
    
    def sample(self):
        return random.sample(self.arr,self.batch_size)

    
