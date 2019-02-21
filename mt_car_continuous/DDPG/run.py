import gym
import numpy as np
import torch
from model_vpg import Policy, Q
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal
import copy
import random

obs_dim = 2
action_dim = 1

epochs = 200
q_lr = 3e-4
policy_lr = 3e-4

policy = Policy(obs_dim,action_dim)
q = Q(obs_dim,action_dim)

# change to parameter copy ???
policy_target = copy.deepcopy(policy)
q_target = copy.deepcopy(q)

q_optim = optim.Adam(q.parameters(), lr=q_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

class Buffer():
    def __init__(self):
        self.pointer = 0
        self.arr = []
        self.max = 1e6
        self.batch_size = 64
    
    def add(self,trans):
        if self.pointer < self.max:
            self.arr.append(trans)
            self.pointer += 1
        else:
            idx = self.pointer % self.max
            self.arr[idx] = trans
            self.pointer += 1
    
    def sample(self):
        return random.sample(self.arr,self.batch_size)



env = gym.make('MountainCarContinuous-v0')
for k in range(epochs):
    if k < 20:
        pass
    else:
        pass
    
    