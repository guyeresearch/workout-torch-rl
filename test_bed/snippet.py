#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 03:29:12 2019

@author: qiaonan
"""

import gym
import numpy as np
import torch
#from model import Policy, Val
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad


# test set grad.
x = torch.tensor(5.,requires_grad=True)
y = x*x
g = grad(y,x)[0]
# because gradient descent of optim, set g negative
x.grad = -g
x_optim = optim.Adam([x],lr=0.1)
x_optim.step()

# test network
class Val(nn.Module):
    def __init__(self,dim_in):
        super().__init__()
        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,60)
        self.fc2 = nn.Linear(60,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

val = Val(4)
t_in = torch.tensor([1.,2.,3.,4.],dtype=torch.float32)
out = val(t_in)
g = grad(out,val.parameters())
for item in zip(list(val.parameters()),g):
    # set negative for optim
    item[0].grad = -item[1]

val_optim = optim.Adam(val.parameters(),lr=0.1)
val_optim.step()
