import gym
import numpy as np
from model import Critic, Actor
from copy import deepcopy
import random
import qn
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn.functional as F
from torch import nn, optim


gamma = 0.99
batchsize = 64
actor_lr = 1e-4

buffer = qn.load('buffer.pkl')

#s, a, r, s_next = list(zip(*buffer))
#
#smat = np.array(s)
#amat = np.ravel(np.array(a))

#rf = RandomForestRegressor(50)
#rf.fit(smat,amat)

actor = Actor(2,1)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)

#%%
actor.train()
for i in range(int(1e4)):
    if (i+1)%100 == 0:
        print(i+1)
    batch = random.sample(buffer,batchsize)
    s, a, r, s_next = list(zip(*batch))
    s = torch.from_numpy(np.array(s,dtype='float32'))
    a = torch.from_numpy(np.array(a,dtype='float32'))
    r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
    s_next = torch.from_numpy(np.array(s_next,dtype='float32'))
    
    a_predict = actor(s)
    actor_loss = F.mse_loss(a_predict,a)
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()



#%%
actor.eval()
history = []
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
#        action = rf.predict([obs])
        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
        action = actor(obs_tensor).data.numpy()[0]
#        action_explore = np.clip(action + noise(action),-1,1)
        obs_new, reward, done, info = env.step(action)
#        print(done)
        history.append([obs.tolist(),action[0],reward,obs_new])
        obs = obs_new
        
        if reward > 10:
            break