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
import os

os.system('rm models/*')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.manual_seed(0)

obs_dim = 4
action_dim = 1
hidden = 64
noise_bank_size = int(1e5)
eps_total = 5000
lr = 1e-1
std = 0.3
weight_decay = 0.0005
# weight_decay = 0
cycle_multiplier = 10
cycle_len = size*cycle_multiplier
save_size = cycle_len*5


r_min = -100


policy = Policy(obs_dim, action_dim, hidden)
paramReshape = ParamReshape(policy)
param_vec = paramReshape.param2vec(policy.parameters())
param_vec_dim = param_vec.shape[0]

utils = torch.tensor(get_utils(cycle_len),dtype=torch.float32)
if rank == 0:
    print(utils)
dbu = Normal(torch.tensor(0.),torch.tensor(1.))
noise_bank = dbu.sample((noise_bank_size,))

# random seeds for each worker
seeds = torch.randint(0,100*size,(size,))
multiRand = MultiRand(seeds, noise_bank, paramReshape.vec_dim)

env = gym.make('CartPole-v1')
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
    done = False
    ret = 0
    j = 0
    while not done:
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        # pdb.set_trace()
        a = policy(obs_tensor).data.tolist()[0]
        a = 0 if a < 0.5 else 1
        obs_new, r, done, info = env.step(a)
        ret += r
        j += 1
        obs = obs_new
    if (i+1) % 10 == 0:
        print('Worker {} finishes {}th episode in {} steps with a return of {:.4f}.'.
        format(rank, i, j, ret))

    rets = np.zeros(size,dtype='d')
    # print(type(ret),type(rets))
    # print(rets)
    comm.Allgather(np.float64(ret),rets)
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


    if rank==0 and (i+1) % save_size == 0:
        # print(param_vec[:20])
        params = paramReshape.vec2param(param_vec)
        for w_act, w in zip(policy.parameters(),params):
            w_act.data = w.data
        torch.save(policy.state_dict(), 
        'models/policy_{}.pkl'.format(int((i+1)/save_size)))



    

    

