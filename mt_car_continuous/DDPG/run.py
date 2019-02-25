import gym
import numpy as np
import torch
from lib import Policy, Q, Buffer, Q2

import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal
import copy
import sys

_, tag = sys.argv


obs_dim = 2
action_dim = 1

epochs = 40
epoch_eps_size = 3
rand_explore_epochs = 10

batch_size = 64
buffer_max = 1e6

gamma = 0.95
rho = 0.995

q_lr = 1e-3
policy_lr = 3e-4

policy = Policy(obs_dim,action_dim,200)

# change to 300 for fair comparison with Q2 structure
# q = Q(obs_dim,action_dim,300)
q = Q2(obs_dim,action_dim,200)

#policy_target = Policy(obs_dim,action_dim,200)
#q_target = Q(obs_dim,action_dim,200)
#
#for w, w_target in zip(q.parameters(),q_target.parameters()):
#    w_target.data.copy_(w.data)
#for w, w_target in zip(policy.parameters(),policy_target.parameters()):
#    w_target.data.copy_(w.data)

# change to parameter copy ???
policy_target = copy.deepcopy(policy)
q_target = copy.deepcopy(q)

q_optim = optim.Adam(q.parameters(), lr=q_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

# buffer = Buffer(batch_size,buffer_max)
buffer = qn.load('buffer_rand.pkl')
#%%
log = []
env = gym.make('MountainCarContinuous-v0')
for k in range(epochs):
    print('epoch {}'.format(k))
#    if k == rand_explore_epochs:
#        qn.dump(buffer,'buffer_rand.pkl')
    update_num = 800
    if k >= rand_explore_epochs:
        added = 0
        update_num = 0
        while added < epoch_eps_size:
            eps = []
            obs = env.reset()
            done = False
            while not done:
                if k < rand_explore_epochs:
                    a = env.action_space.sample()
                else:
                    obs_tensor = torch.tensor(obs,dtype=torch.float)
                    a = [policy(obs_tensor).data.tolist()[0] + 
                        np.random.normal(scale=0.5)]
                obs_new, r, done, info = env.step(a)
                eps.append([obs,a,r,obs_new, done])
                obs = obs_new
            if len(eps) < 999:
                print('added {} eps, current size {}'.format(added, len(eps)))
                added += 1
                update_num += len(eps)/2
                buffer.add_many(eps)
                log.append((k,len(eps)))
    

    for j in range(int(update_num)):
        if (j+1) % 100 == 0:
            print('perform {} update'.format(j))
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
        
#        # very very wrong!!!
#        for w, w_target in zip(q.parameters(),q_target.parameters()):
#            w_target = rho*w_target + (1-rho)*w
#        for w, w_target in zip(policy.parameters(),policy_target.parameters()):
#            w_target = rho*w_target + (1-rho)*w
        
        for w, w_target in zip(q.parameters(),q_target.parameters()):
            w_target.data = rho*w_target.data + (1-rho)*w.data
        for w, w_target in zip(policy.parameters(),policy_target.parameters()):
            w_target.data = rho*w_target.data + (1-rho)*w.data
        

qn.dump(log,'log2/a{}_q2.pkl'.format(tag))
#%% test
#env = gym.make('MountainCarContinuous-v0')
#for i_episode in range(5):
#    obs = env.reset()
#    
#    for t in range(2000):
#        if (t+1)%100 == 0:
#            print(t+1)
#        env.render()
#        obs_tensor = torch.from_numpy(obs.astype('float32'))
#        a = policy(obs_tensor)
#        a = [a.data.tolist()[0] + 
#                    np.random.normal(scale=1)]
##        a = a.data.tolist()
#        obs_new, r, done, info = env.step(a)
#        obs = obs_new
#
