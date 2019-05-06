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



obs_dim = 14
action_dim = 4
skip_dim = 1

epochs = 600
epoch_eps_size = 3
rand_explore_epochs = 0

batch_size = 64
buffer_max = 1e6

gamma = 0.99
rho = 0.995

q_lr = 1e-3
policy_lr = 3e-4

policy = Policy(obs_dim,action_dim+skip_dim,150)

# change to 300 for fair comparison with Q2 structure
# q = Q(obs_dim,action_dim,300)
q = Q2(obs_dim,action_dim+skip_dim,150)

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

buffer = Buffer(batch_size,buffer_max)
# buffer = qn.load('buffer_rand.pkl')
#%%
log = []
class Env():
    def __init__(self):
        self.env = gym.make('BipedalWalker-v2')
    def reset(self):
        obs = self.env.reset()
        return obs[:14]
    def step(self,a):
        obs_new, r, done, info = self.env.step(a)
        return obs_new[:14], r, done, info
    def render(self):
        self.env.render()
env = Env()

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
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            mean = policy(obs_tensor)
            a_pre = mean[:action_dim].data.tolist() \
                + np.random.normal(np.zeros(action_dim),scale=0.3)
            done = False
            while not done:
                obs_tensor = torch.tensor(obs,dtype=torch.float)
                mean = policy(obs_tensor)
                a_mean = mean[:action_dim]
                skip = (mean[-1].data.tolist()+1)/2
                a_now = a_mean.data.tolist() + \
                    np.random.normal(np.zeros(action_dim),scale=0.3)
                a = a_pre*skip + a_now*(1-skip)
                obs_new, r, done, info = env.step(a)
                if r < -90:
                    r = -30
                eps.append([obs,np.concatenate((a,[skip])),r,obs_new, done])
                obs = obs_new
            
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
        

# qn.dump(log,'log2/a{}_q2.pkl'.format(tag))
#%% test
#env = gym.make('BipedalWalker-v2')
for i_episode in range(5):
    obs = env.reset()
    t = 0
    r_total = 0
    done = False
    while not done and t<2000:
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs.astype('float32'))
#        obs_tensor[-10:] = 0
        mean = policy(obs_tensor)
        print(mean[-1])
#            print(p)
#        dbu = Normal(mean,std)
#        a = dbu.sample()
            #g = grad(obj,policy.parameters())
        obs_new, r, done, info = env.step(mean[:action_dim].data.tolist())
#        obs_new, r, done, info = env.step(mean.data.tolist())
#        action_explore = np.clip(action + noise(action),-1,1)
#        print(done)
        #history.append([obs,action[0],reward,obs_new])
        obs = obs_new
        t += 1
        r_total += r
    print('points: {}'.format(r_total))
#
