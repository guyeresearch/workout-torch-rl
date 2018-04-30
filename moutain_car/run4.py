import gym
import numpy as np
from model import Critic, Actor
from copy import deepcopy
import random
import qn
from sklearn.ensemble import RandomForestRegressor
from  scipy import signal

import torch
import torch.nn.functional as F
from torch import nn, optim

gamma = 0.95
batchsize = 64
actor_lr = 1e-4
critic_lr = 1e-3

buffer = qn.load('buffer.pkl')


def alt(rewards, discount):
    """
    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=r)
    return y[::-1]


s, a, r, s_next = list(zip(*buffer))

smat = np.array(s)
amat = np.array(a)
x = np.hstack([smat,amat])

rts = []
rs = []
for item in buffer:
    rs.append(item[2])
    if item[2] > 10:
        rts += list(alt(rs,gamma))
        rs = []

rf = RandomForestRegressor(50)
rf.fit(x,rts)        
        
    

history = []
action_map = [-1,1]
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        actions = rf.predict([obs.tolist()+[-1],obs.tolist()+[1]])
        idx = np.argmax(actions)
        action = [float(action_map[idx])]
#        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
#        action = actor(obs_tensor).data.numpy()[0]
#        action_explore = np.clip(action + noise(action),-1,1)
        obs_new, reward, done, info = env.step(action)
#        print(done)
        history.append([obs.tolist(),action[0],reward,obs_new])
        obs = obs_new
        
        if reward > 10:
            break