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
critic_lr = 1e-3
actor_lr = 1e-4

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

a_next = list(a[1:]) + [a[0]]

buffer2 = list(zip(s,a,r,s_next,a_next))


critic = Critic(2,1)
target_critic = Critic(2,1)
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

actor = Actor(2,1)
target_actor = Actor(2,1)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()

critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
#%% 

critic.train()
actor.train()
for i in range(int(1e4)):
    if (i+1)%100 == 0:
        print(i+1)
    batch = random.sample(buffer2,batchsize)
    s, a, r, s_next, a_next = list(zip(*batch))
    s = torch.from_numpy(np.array(s,dtype='float32'))
    a = torch.from_numpy(np.array(a,dtype='float32'))
    r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
    s_next = torch.from_numpy(np.array(s_next,dtype='float32'))
    
#    non_final_mask = (r < 10).squeeze()
#    s_next_non_final = s_next[non_final_mask,:]
#    a_next_non_final = a_next[non_final_mask,:]
#    
#    q_next = torch.zeros((batchsize,1))
#    q_next[non_final_mask] = target(s_next_non_final,a_next_non_final).detach()
    critic.train()
    a_next = target_actor(s_next).detach()
    q_next = target_critic(s_next,a_next).detach()
    y = r + gamma*q_next
    q = critic(s,a)
    critic_loss = F.mse_loss(q,y)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
        
    critic.eval()
    a_predict = actor(s)
    q = critic(s,a_predict)
    actor_loss = -q.mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()
    
    
    if (i+1)%50 == 0:
        target_critic.load_state_dict(critic.state_dict())
        target_actor.load_state_dict(actor.state_dict())



#%%
#critic.eval()
#s, a, r, s_next, a_next = list(zip(*buffer2[:200]))
#s = torch.from_numpy(np.array(s,dtype='float32'))
#a = torch.from_numpy(np.array(a,dtype='float32'))
#r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
#s_next = torch.from_numpy(np.array(s_next,dtype='float32'))
#a_next = torch.from_numpy(np.array(a_next,dtype='float32'))
#
#q = critic(s,a).detach().numpy()
#cc = np.hstack([np.array(s),np.array(a), q])

         
#%%
actor.eval()
history = []
action_map = [-1,1]
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
        action = actor(obs_tensor).data.numpy()[0]
        obs_new, reward, done, info = env.step(action)
        history.append([obs.tolist(),action[0],reward,obs_new])
        obs = obs_new
        
        if reward > 10:
            break