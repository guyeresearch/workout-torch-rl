from mpi4py import MPI
import numpy as np
import gym

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


env = gym.make('BipedalWalker-v2')
obs = env.reset()
_, r, _, _ = env.step(env.action_space.sample())

print('rank {} reward {}'.format(rank,r))

rets = np.zeros(size,dtype='d')
comm.Allgather(r,rets)

print('rank {} rets {}'.format(rank,rets))