#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 08:49:47 2019

@author: qiaonan
"""

import gym
import numpy as np
import torch
from model import Policy, Val
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.binomial import Binomial

obs_dim = 4
action_dim = 1

# parameters
epochs = 1
D_size = 20
gamma = 0.97
lda = 0.92 # for generalized advantage esitmate

val_epochs = 20
val_batch = 32
val_lr = 5e-4

policy_lr = 1e-5 / D_size
lower_bound = 1e-8

torch.manual_seed(123)

policy = Policy(obs_dim,action_dim)
val = Val(obs_dim)

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

#binomial = Binomial(1,torch.tensor(0.5))

env = gym.make('CartPole-v1')
env.seed(123)

#%%
for k in range(epochs):
    print('epoch: {}'.format(k))
    # collect D
    D = []
    for _ in range(D_size):
        eps = []
        obs = env.reset()
#        print(obs)
        done = False
        i = 0
        while not done:
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            p = policy(obs_tensor)[0]
            binomial = Binomial(1,p)
            a = binomial.sample()
#            print(p_value)
            loss = -torch.log(a*p + (1-a)*(1-p))
            #g = grad(obj,policy.parameters())
            obs_new, r, done, info = env.step(int(a.data))
            eps.append([obs,a,r,p,loss,obs_new])
            obs = obs_new
            i += 1
        print('end in {} steps'.format(i+1))
        D.append(eps)
    
    logj = 0
    for eps in D:
        R = sum(x[2] for x in eps)
        logj += torch.log(torch.tensor(R,dtype=torch.float32))
        for item in eps:
            obs,a,r,p,loss,obs_new =  item
            p_a = a*p + (1-a)*(1-p)
            logj += torch.log(p_a)
