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
import pdb

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.manual_seed(0)

obs_dim = 14
action_dim = 4
hidden = 200
noise_bank_size = int(1e5)
eps_total = 5000
lr = 1e-3
std = 0.5
weight_decay = 0.0005
# weight_decay = 0
cycle_multiplier = 20

r_min = -100

cycle_len = size*cycle_multiplier

policy = Policy(obs_dim, action_dim, hidden)
paramReshape = ParamReshape(policy)
param_vec = paramReshape.param2vec(policy.parameters())

utils = torch.tensor(get_utils(cycle_len),dtype=torch.float32)
if rank == 0:
    print(utils)

dbu = Normal(torch.tensor(0.),torch.tensor(1.))
noise_bank = dbu.sample((noise_bank_size,))

# random seeds for each worker
seeds = torch.randint(0,100*size,(size,))
multiRand = MultiRand(seeds, noise_bank, paramReshape.vec_dim)

env = gym.make('BipedalWalker-v2')
cycle_rets = []
cycle_noises = []
for i in range(eps_total):
    # param_vec, act_policy, policy
    noises = multiRand.get_noises()
    cycle_noises += noises
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
        print('Worker {} finishes {}th episode in {} steps with a return of {:.4f}.'.
        format(rank, i, j, ret))

    rets = np.zeros(size,dtype='d')
    comm.Allgather(ret,rets)
    cycle_rets = np.concatenate((cycle_rets,rets))

    if (i+1) % cycle_multiplier == 0:
        if rank == 0:
            print('Update param_vec length {}'.format(cycle_rets.shape[0]))
        # argsort sorts from low to high
        idx = np.argsort(-cycle_rets)
        current_utils = np.zeros(cycle_len)
        current_utils[idx] = utils
        # # this is wrong!!!!
        # current_utils = utils[idx]
        grads = 0
        # pdb.set_trace()
        for u, noise in zip(current_utils,cycle_noises):
            # print(u,noise)
            grads += u*noise
        # print(grads)
        # no dividing by n when using utility
        param_vec +=  lr/std*grads - param_vec*weight_decay
        cycle_noises = []
        cycle_rets = []
        # break


    if rank==0 and (i+1) % 100 == 0:
        print(param_vec[:20])
        params = paramReshape.vec2param(param_vec)
        for w_act, w in zip(policy.parameters(),params):
            w_act.data = w.data
        torch.save(policy.state_dict(), 'models/policy_{}.pkl'.format(i+1))



    

    

