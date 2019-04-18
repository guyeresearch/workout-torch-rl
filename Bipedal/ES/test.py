from mpi4py import MPI
import numpy as np
import gym
from lib import *
from torch.distributions.normal import Normal
import pdb


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# rets = np.zeros(size,dtype='d')
# comm.Allgather(1.,rets)


# test Allgather
env = gym.make('BipedalWalker-v2')
obs = env.reset()
_, r, _, _ = env.step(env.action_space.sample())

# print('rank {} reward {}'.format(rank,r))

# pdb.set_trace()
rets = np.zeros(size,dtype='d')
print(rank,type(r),type(rets),r,rets)
comm.Allgather(np.float64(1.),rets)
print(rets)

# pdb.set_trace()

# print('rank {} rets {}'.format(rank,rets))

# # test noises
# torch.manual_seed(0)
# noise_bank_size = 100

# dbu = Normal(torch.tensor(0.),torch.tensor(1.))
# noise_bank = dbu.sample((noise_bank_size,))

# seeds = torch.randint(0,100*size,(size,))
# multiRand = MultiRand(seeds, noise_bank, 2)

# for i in range(10):
#     noises = multiRand.get_noises()
#     print('{}th run rank {} noises {}'.format(i,rank, noises))