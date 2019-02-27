import gym
import numpy as np
from model import Critic, Actor
import random
import qn

import torch
import torch.nn.functional as F
from torch.autograd import Variable,grad
import torch.nn as nn
from torch import nn, optim

gamma = 0.95
batchsize = 64
critic_lr = 1e-3
actor_lr = 1e-4

def noise_factory(mu=0,theta=0.15,sigma=0.2):
    def noise(x):
        return theta * (mu - x) + sigma * np.random.randn(len(x))
    return noise

def getMaxReturn(episode):
    cumsum = np.cumsum([x[2] for x in episode])
    idx = np.argmax(cumsum)
    return idx, cumsum[idx], len(episode)

critic = Critic(24,4)
target_critic = Critic(24,4)
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

actor = Actor(24,4)
target_actor = Actor(24,4)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()

critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
noise = noise_factory()

# epsilon = 1
#%%
env = gym.make('BipedalWalker-v2')
for i_epoch in range(9,100):
    epsilon = 30/(30+i_epoch)
    buffer = []
    for i_episode in range(200):
        obs = env.reset()
        eps_buffer = []
        for t in range(2000):
            if np.random.rand() > epsilon:
                actor.eval()
                obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
                action = actor(obs_tensor).data.numpy()[0]
                action = np.clip(action,-1,1)
            else:
#                print('random')
                action = env.action_space.sample()
            obs_new, reward, done, info = env.step(action)
            eps_buffer.append([obs,action,reward,obs_new])
            obs = obs_new
            if done:
                print("Episode {}/{} finished after {} timesteps, max rewards {:.2f}."
                .format(i_epoch,i_episode,t+1,getMaxReturn(eps_buffer)[1]))
                break
        buffer.append(eps_buffer)
    buffer = sorted(buffer,key=lambda x:-getMaxReturn(x)[1])[:20]
    buffer2 = [item for sub in buffer for item in sub]
    
    s, a, r, s_next = list(zip(*buffer2))
    r = np.array(r)
    r[r<-50] = -2
    r[r>0] = r[r>0]*5
    a_next = list(a[1:]) + [a[0]]
    buffer2 = list(zip(s,a,r,s_next,a_next))

    for i in range(int(1e3)):
        if (i+1)%100 == 0:
            print('train {}/{}'.format(i_epoch,i+1))
        batch = random.sample(buffer2,batchsize)
        s, a, r, s_next, a_next = list(zip(*batch))
        s = torch.from_numpy(np.array(s,dtype='float32'))
        a = torch.from_numpy(np.array(a,dtype='float32'))
        r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
        s_next = torch.from_numpy(np.array(s_next,dtype='float32'))

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
            
actor.eval()
history = []
env = gym.make('BipedalWalker-v2')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
        action = actor(obs_tensor).data.numpy()[0]
        action = np.clip(action,-1,1)
        action = action + noise(action)
        obs_new, reward, done, info = env.step(action)
        history.append([obs.tolist(),action,reward,obs_new])
        obs = obs_new

        if done:
            break

s, a, r, s_next = list(zip(*history))