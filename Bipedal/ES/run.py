import gym
import numpy as np
import torch
from lib import Policy, MultiRand

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

policy = Policy(obs_dim, action_dim, hidden)
dbu = Normal(torch.tensor(0.),torch.tensor(1.))
noise_bank = dbu.sample((noise_bank_size,))

# random seeds for each worker
seeds = torch.randint(0,100*size,(size,))
    




