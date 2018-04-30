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
target = Critic(2,1)
target.load_state_dict(critic.state_dict())
target.eval()

critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
#%% 

critic.train()
for i in range(int(1e4)):
    if (i+1)%100 == 0:
        print(i+1)
    batch = random.sample(buffer2,batchsize)
    s, a, r, s_next, a_next = list(zip(*batch))
    s = torch.from_numpy(np.array(s,dtype='float32'))
    a = torch.from_numpy(np.array(a,dtype='float32'))
    r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
    s_next = torch.from_numpy(np.array(s_next,dtype='float32'))
    a_next = torch.from_numpy(np.array(a_next,dtype='float32'))
    
    non_final_mask = (r < 10).squeeze()
    s_next_non_final = s_next[non_final_mask,:]
    a_next_non_final = a_next[non_final_mask,:]
    
    q_next = torch.zeros((batchsize,1))
    q_next[non_final_mask] = target(s_next_non_final,a_next_non_final).detach()
    y = r + gamma*q_next
    
    q = critic(s,a)

    critic_loss = F.smooth_l1_loss(q,y)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    
    if (i+1)%300 == 0:
        target.load_state_dict(critic.state_dict())



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
critic.eval()
history = []
action_map = [-1,1]
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_mat = np.array([obs,obs],dtype='float32')
        action_mat = np.array([[-1],[1]],dtype='float32')
        actions = critic(torch.from_numpy(obs_mat),torch.from_numpy(action_mat))
        idx = np.argmax(actions.data.numpy()[:,0])
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