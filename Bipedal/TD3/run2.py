import gym
import numpy as np
import torch
from lib import Policy, Q, Buffer

import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal
import copy
import sys


obs_dim = 14
action_dim = 4

init_eps = 30
epochs = 500

batch_size = 100
buffer_max = 1e6

# check paper. Action smooth parameter
c = 0.5
noise_std = 0.2

gamma = 0.99
rho = 0.995

q_lr = 1e-3
policy_lr = 1e-3
policy_delay = 2

policy = Policy(obs_dim,action_dim,200)
q = Q(obs_dim,action_dim,200)
q2 = Q(obs_dim,action_dim,200)

policy_target = copy.deepcopy(policy)
q_target = copy.deepcopy(q)
q2_target = copy.deepcopy(q2)

policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)
q_optim = optim.Adam(q.parameters(), lr=q_lr)
q2_optim = optim.Adam(q2.parameters(), lr=q_lr)

buffer = Buffer(batch_size,buffer_max)

min_r = -15
# initialization
env = gym.make('BipedalWalker-v2')
for i in range(init_eps):
    obs = env.reset()
    obs = obs[:14]
    done = False
    while not done:
        a = env.action_space.sample()
        obs_new, r, done, info = env.step(a)
        if r < -90:
            r = min_r
        buffer.add([obs,a,r,obs_new[:14], done])
        obs = obs_new[:14]

std = 1
for k in range(epochs):
    print('epoch {}'.format(k))
    t = 1
    obs = env.reset()
    obs = obs[:14]
    done = False
    while not done:
        # running
        obs_tensor = torch.tensor(obs,dtype=torch.float)
        mean = policy(obs_tensor)
        dbu = Normal(mean,std)
        a = dbu.sample().data.tolist()
        # check a shape
        a = np.clip(a, env.action_space.low, env.action_space.high)
        obs_new, r, done, info = env.step(a)
        if r < -90:
            r = min_r
        buffer.add([obs,a,r,obs_new[:14], done])
        obs = obs_new[:14]
        
        if t > 300:
            t += 1
            continue
        # training
        # q update
        batch = buffer.sample()
        obs_train, a, r, obs_new, done_train = [torch.tensor(x,dtype=torch.float) 
            for x in zip(*batch)]
        mean = policy_target(obs_new).data.tolist()
        # check std parameter, which value to take???
        noise = np.clip(np.random.normal(scale=noise_std,size=action_dim), -c, c)
        a_prime = np.clip(mean+noise, env.action_space.low, env.action_space.high)
        a_prime_tensor = torch.tensor(a_prime,dtype=torch.float)
        q_val_target = q_target(obs_new,a_prime_tensor)
        q2_val_target = q2_target(obs_new,a_prime_tensor)
        q_both = torch.cat((q_val_target,q2_val_target),dim=1)
        q_min, _ = torch.min(q_both,dim=1)
        y = (r + gamma*(1-done_train)*q_min).detach()

        q_val = q(obs_train,a)[:,0]
        loss = F.mse_loss(q_val,y)
        q_optim.zero_grad()
        loss.backward()
        q_optim.step()

        q2_val = q2(obs_train,a)[:,0]
        loss = F.mse_loss(q2_val,y)
        q2_optim.zero_grad()
        loss.backward()
        q2_optim.step()

        # policy update
        if t % 2 == 0:
            a_max = policy(obs_train)
            q_val = q(obs_train,a_max)
            q_optim.zero_grad()
            policy_optim.zero_grad()
            # gradient ascent
            torch.mean(-q_val).backward()
            policy_optim.step()
        
            for w, w_target in zip(q.parameters(),q_target.parameters()):
                w_target.data = rho*w_target.data + (1-rho)*w.data
            for w, w_target in zip(q2.parameters(),q2_target.parameters()):
                w_target.data = rho*w_target.data + (1-rho)*w.data
            for w, w_target in zip(policy.parameters(),policy_target.parameters()):
                w_target.data = rho*w_target.data + (1-rho)*w.data

        t += 1

    print('end in {} steps'.format(t))
    if (k+1) % 50 == 0:
        std = std if std <= 0.3 else std*0.8
    
#%% test
env = gym.make('BipedalWalker-v2')
for i_episode in range(5):
    obs = env.reset()
    
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs[:14].astype('float32'))
        a = policy(obs_tensor)
        a = a.data.tolist()
#        a = a.data.tolist()
        obs_new, r, done, info = env.step(a)
        obs = obs_new

