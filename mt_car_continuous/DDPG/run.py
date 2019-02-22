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

obs_dim = 2
action_dim = 1

epochs = 200
epoch_eps_size = 10
update_num = 10
rand_explore_epochs = 20

batch_size = 64
buffer_max = 1e6

gamma = 0.98
rho = 0.99

q_lr = 3e-4
policy_lr = 3e-4

policy = Policy(obs_dim,action_dim)
q = Q(obs_dim,action_dim)

# change to parameter copy ???
policy_target = copy.deepcopy(policy)
q_target = copy.deepcopy(q)

q_optim = optim.Adam(q.parameters(), lr=q_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

buffer = Buffer(batch_size,buffer_max)

env = gym.make('MountainCarContinuous-v0')
for k in range(epochs):
    added = 0
    while added < epoch_eps_size:
        eps = []
        obs = env.reset()
        done = False
        while not done:
            if k < rand_explore_epochs:
                a = env.action_space.sample()
            else:
                a = [policy(obs).data.tolist()[0] + 
                    np.random.normal(scale=0.5)]
            obs_new, r, done, info = env.step(a)
            eps.append([obs,a,r,obs_new, done])
            obs = obs_new
        if len(eps) < 999:
            added += 1
            buffer.add_many(eps)
    

    for _ in range(update_num):
        batch = buffer.sample()
        obs, a, r, obs_new, done = [torch.tensor(x,dtype=torch.float) 
            for x in zip(*batch)]
        a_max = policy_target(obs_new)
        q_val_target = q_target(obs_new,a_max)[:,0]
        y = (r + gamma*(1-done)*q_val_target).detach()
        q_val = q(obs,a)[:,0]
        loss = F.mse_loss(q_val,y)
        q_optim.zero_grad()
        loss.backward()
        q_optim.step()

        a_max = policy(obs)
        q_val = q(obs,a_max)
        q_optim.zero_grad()
        policy_optim.zero_grad()
        # gradient ascent
        torch.mean(-q_val).backward()
        policy_optim.step()

        for w, w_target in zip(q.parameters(),q_target.parameters()):
            w_target = rho*w_target + (1-rho)*w
        for w, w_target in zip(policy.parameters(),policy_target.parameters()):
            w_target = rho*w_target + (1-rho)*w
        
        
