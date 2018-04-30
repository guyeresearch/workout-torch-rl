import gym
import numpy as np
from model import Critic, Actor
from copy import deepcopy
import random

import torch
import torch.nn.functional as F
from torch import nn, optim

batchsize = 64
gamma = 0.95
tau = 0.001
critic_lr = 1e-3
actor_lr = 1e-4
buffer_size = int(1e6)
train_start_size = int(1e4)

def noise_factory(mu=0,theta=0.15,sigma=0.3):
    def noise(x):
        return theta * (mu - x) + sigma * np.random.randn(len(x))
    return noise

critic = Critic(2,1)
actor = Actor(2,1)
critic_ = deepcopy(critic)
actor_ = deepcopy(actor)
critic_.eval()
actor_.eval()
critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)

noise = noise_factory()
buffer = []

env = gym.make('MountainCarContinuous-v0')
for i_episode in range(1000):
    obs = env.reset()
    action = env.action_space.sample()
    for t in range(10000):
        if (t+1)%100 == 0:
            print(t+1)
        # env.render()
        actor.eval()
#        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
#        action = actor(obs_tensor).data.numpy()[0]
#        action_explore = np.clip(action + noise(action),-1,1)
        if i_episode <=20:
            action = action + noise(action)
        else:
            obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
            action = actor(obs_tensor).data.numpy()[0]
            action = np.clip(action + noise(action),-1,1)
        obs_new, reward, done, info = env.step(action)
        buffer.append([obs,action,reward,obs_new])
        obs = obs_new

        if len(buffer)>buffer_size:
            buffer = buffer[-buffer_size:]

        if len(buffer)>=train_start_size:
            batch = random.sample(buffer,batchsize)
            s, a, r, s_next = list(zip(*batch))
            s = torch.from_numpy(np.array(s,dtype='float32'))
            a = torch.from_numpy(np.array(a,dtype='float32'))
            r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
            s_next = torch.from_numpy(np.array(s_next,dtype='float32'))

            action_next = actor_(s_next)
            q_next = critic_(s_next,action_next)
            y = r + gamma*q_next.detach()
            critic.train()
            q = critic(s,a)
            critic_loss = F.mse_loss(q,y)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            critic.eval()
            actor.train()
            a_predict = actor(s)
            q = critic(s,a_predict)
            actor_loss = -q.mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            actor.eval()
            for item in zip(critic_.parameters(),critic.parameters()):
                item[0].data = tau*item[1].data + (1-tau)*item[0].data
            for item in zip(actor_.parameters(),actor.parameters()):
                item[0].data = tau*item[1].data + (1-tau)*item[0].data

        if reward > 10:
            print('done')
            break
    print("Episode {} finished after {} timesteps. buffer size {}"
        .format(i_episode,t+1,len(buffer)))


history = []
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        actor.eval()
        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
        action = actor(obs_tensor).data.numpy()[0]
#        action_explore = np.clip(action + noise(action),-1,1)
        obs_new, reward, done, info = env.step(action)
#        print(done)
        history.append([obs,action[0],reward,obs_new])
        obs = obs_new