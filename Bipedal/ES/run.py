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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.manual_seed(0)

obs_dim = 14
action_dim = 4
hidden = 200

noise_bank_size = int(1e5)
eps_total = 3000
lr = 0.01
std = 0.5
# weight_decay = 0.0005
weight_decay = 0


r_min = -100

policy = Policy(obs_dim, action_dim, hidden)
paramReshape = ParamReshape(policy)
param_vec = paramReshape.param2vec(policy.parameters())

utils = torch.tensor(get_utils(size),dtype=torch.float32)
print(utils)

dbu = Normal(torch.tensor(0.),torch.tensor(1.))
noise_bank = dbu.sample((noise_bank_size,))

# random seeds for each worker
seeds = torch.randint(0,100*size,(size,))
multiRand = MultiRand(seeds, noise_bank, paramReshape.vec_dim)

if rank == 0:
    print(param_vec[:10])

env = gym.make('BipedalWalker-v2')
for i in range(eps_total):
    # param_vec, act_policy, policy
    noises = multiRand.get_noises()
    noise = noises[rank]
    param_vec_noise = param_vec + noise*std
    params = paramReshape.vec2param(param_vec_noise)
    for w_act, w in zip(policy.parameters(),params):
        w_act.data = w.data
    
    obs = env.reset()
    obs = obs[:14]
    done = False
    ret = 0
    j = 0
    while not done:
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        a = policy(obs_tensor)
        obs_new, r, done, info = env.step(a.data.tolist())
        obs_new = obs_new[:14]
        if r < -90:
            r = r_min
        ret += r
        j += 1
        obs = obs_new
    if (i+1) % 10 == 0:
        # print('Worker {} finishes {}th episode in {} steps with a return of {:.4f}.'.
        # format(rank, i, j, ret))
        print('Worker {} param_vec {}'.format(rank, param_vec[:10]))

    rets = np.zeros(size,dtype='d')
    comm.Allgather(ret,rets)

    # argsort sorts from low to high
    idx = np.argsort(-rets)
    current_utils = utils[idx]

    grads = 0
    for u, noise in zip(current_utils,noises):
        grads += u*noise
    param_vec += - param_vec*weight_decay + \
         lr/size/std*grads

    if rank ==0 and (i+1) % 50 == 0:
        print(param_vec[:10])

    if rank==0 and (i+1) % 500 == 0:
        params = paramReshape.vec2param(param_vec)
        for w_act, w in zip(policy.parameters(),params):
            w_act.data = w.data
        torch.save(policy.state_dict(), 'models/policy_{}.pkl'.format(i+1))



    

    

