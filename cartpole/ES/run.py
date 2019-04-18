import gym
import numpy as np
import torch
from lib import *

import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from torch.distributions.normal import Normal
import copy
import sys
import pdb


obs_dim = 4
action_dim = 1

## params that works
# noise_bank_size = int(1e5)
# lr = 1e-3
# std = 0.3
# weight_decay = 0.0005
# # weight_decay = 0
# size = 50
# eps_total = size*100

# torch.manual_seed(0)

noise_bank_size = int(1e4)
lr = .1
std = 0.2
# weight_decay = 0.0005
weight_decay = 0
size = 50
eps_total = size*30

# torch.manual_seed(1)


policy = Policy(obs_dim, action_dim)
paramReshape = ParamReshape(policy)
param_vec = paramReshape.param2vec(policy.parameters())
param_vec_dim = param_vec.shape[0]

utils = torch.tensor(get_utils(size),dtype=torch.float32)
dbu = Normal(torch.tensor(0.),torch.tensor(1.))
noise_bank = dbu.sample((noise_bank_size,))

# pdb.set_trace()

env = gym.make('CartPole-v1')
noises = []
rets = []
for i in range(eps_total):
    idx = torch.randint(0,noise_bank.shape[0],
        (param_vec_dim,))
    # pdb.set_trace()
    noise = noise_bank[idx]
    noises.append(noise)
    param_vec_noise = param_vec + noise*std
    params = paramReshape.vec2param(param_vec_noise)
    for w_act, w in zip(policy.parameters(),params):
        w_act.data = w.data
    
    obs = env.reset()
    done = False
    ret = 0
    j = 0
    while not done:
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        a = policy(obs_tensor).data.tolist()[0]
        a = 0 if a < 0.5 else 1
        obs_new, r, done, info = env.step(a)
        ret += r
        j += 1
        obs = obs_new
    if (i+1) % 10 == 0:
        print('Finishes {}th episode in {} steps with a return of {:.4f}.'.
        format(i, j, ret))
        #print('param_vec {}'.format(param_vec[:10]))
    rets.append(ret)

    if (i+1) % size == 0:
        idx = np.argsort(-np.array(rets))
        current_utils = np.zeros(size)
        current_utils[idx] = utils
        grads = 0
        # pdb.set_trace()
        for u, noise in zip(current_utils,noises):
            # print(u,noise)
            grads += u*noise
        param_vec +=  lr/std*grads - param_vec*weight_decay
        
        noises = []
        rets = []
    
    if (i+1) % size*10 == 0:
        params = paramReshape.vec2param(param_vec)
        for w_act, w in zip(policy.parameters(),params):
            w_act.data = w.data
        torch.save(policy.state_dict(), 
        'models/policy_{}.pkl'.format(int((i+1)/size/10)))

    
