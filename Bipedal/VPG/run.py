import gym
import numpy as np
from model import Critic, Actor
import random
import qn

import torch
import torch.nn.functional as F
from torch.autograd import Variable,grad
import torch.nn as nn

def noise_factory(mu=0,theta=0.2,sigma=0.5):
    def noise(x):
        return theta * (mu - x) + sigma * np.random.randn(len(x))
    return noise

def getMaxReturn(episode):
    cumsum = np.cumsum([x[2] for x in episode])
    idx = np.argmax(cumsum)
    return idx, cumsum[idx], len(episode)

noise = noise_factory()
buffer = []
counts = []

#%%

env = gym.make('BipedalWalker-v2')
for i_episode in range(3000):
   obs = env.reset()
   action = env.action_space.sample()
   eps_buffer = []
   for t in range(2000):
       if (t+1)%100 == 0:
           print(t+1)
#       env.render()
#       action = [1,0,-1,-1]
#       action = noise(action)
       action = env.action_space.sample()
       obs_new, reward, done, info = env.step(action)
       eps_buffer.append([obs,action,reward,obs_new])
       obs = obs_new

       if done:
           print('done')
           break
   print("Episode {} finished after {} timesteps. buffer size {}"
       .format(i_episode,t+1,len(buffer)))
   if getMaxReturn(eps_buffer)[1] > 3:
       buffer += eps_buffer
#   buffer.append(eps_buffer)
#   break
#


#cc = sorted([getMaxReturn(x) for x in buffer], key = lambda x:-x[1])

#def get_r_count(eps):
#    rs = [x[2] for x in eps]
#    return sum(np.array(rs)>0), len(eps)
#pos_counts = [get_r_count(x) for x in buffer]


qn.dump(buffer,'buffer.pkl')