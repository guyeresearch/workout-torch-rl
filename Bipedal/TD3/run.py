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

init_eps = 20
epochs = 1000

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

min_r = -100
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

#%%
#std = 1
std = 0.1
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
        
#        if done:
#            for i in range(100):
#                buffer.add([obs,a,r,obs_new[:14], done])
#        
#        break
        
        if t > 500:
            # train at most 300 steps for each episode
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
        
        break
    print('end in {} steps'.format(t))
    if (k+1) % 100 == 0:
        std = std if std <= 0.3 else std*0.9
    break
#%% test
env = gym.make('BipedalWalker-v2')
for i_episode in range(5):
    obs = env.reset()
    obs = obs[:14]
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
#            print(p)
        dbu = Normal(mean,std)
        a = dbu.sample()
            #g = grad(obj,policy.parameters())
#        obs_new, r, done, info = env.step(a.data.tolist())
        obs_new, r, done, info = env.step(mean.data.tolist())
#        action_explore = np.clip(action + noise(action),-1,1)
#        print(done)
        #history.append([obs,action[0],reward,obs_new])
        obs_new = obs_new[:14]
        obs = obs_new
        t += 1
        r_total += r
    print('points: {}'.format(r_total))


#torch.save(policy.state_dict(), 'policy_td3_normal_walking_mid_trainx.pkl')
#torch.save(q.state_dict(), 'q_td3_normal_walking_mid_trainx.pkl')
#torch.save(q2.state_dict(), 'q2_td3_normal_walking_mid_trainx.pkl')
