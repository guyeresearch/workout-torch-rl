import gym
import numpy as np
import torch
from lib_1 import Policy, Q, Buffer, V

import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal
import copy
import sys

EPS = 1e-8

obs_dim = 14
action_dim = 4

init_steps = int(1e4)
epoch_steps = 3000
epochs = 100
train_steps = 3000


batch_size = 100
buffer_max = 1e6

# check paper. Action smooth parameter
# c = 0.5
# check paper for entropy param
# paper uses reward scaling of 5 for simple environments
# which is equivalent to an alpha of 0.2
alpha = 0.012

gamma = 0.99
rho = 0.995

q_lr = 1e-3
policy_lr = 1e-3
v_lr = 1e-3

# check OpenAI implementation of policy network
policy = Policy(obs_dim,action_dim*2,200)
q = Q(obs_dim,action_dim,200)
q2 = Q(obs_dim,action_dim,200)
v = V(obs_dim,200)

v_target = copy.deepcopy(v)


policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)
q_optim = optim.Adam(q.parameters(), lr=q_lr)
q2_optim = optim.Adam(q2.parameters(), lr=q_lr)
v_optim = optim.Adam(v.parameters(),lr=v_lr)

buffer = Buffer(batch_size,buffer_max)

min_r = -100
# initialization
env = gym.make('BipedalWalker-v2')
obs = env.reset()
obs = obs[:14]
eps_len = 0
for i in range(init_steps):
    a = env.action_space.sample()
    obs_new, r, done, info = env.step(a)
    if r < -90:
        r = min_r
    buffer.add([obs,a,r,obs_new[:14], done])
    obs = obs_new[:14]
    eps_len += 1
    if done:
        print('episode end in {} steps.'.format(eps_len))
        eps_len = 0
        obs = env.reset()
        obs = obs[:14]

#%%
obs = env.reset()
obs = obs[:14]
eps_len = 0
for i in range(int(epoch_steps*epochs)):
    # training 
    # first train on random policy buffer
    if i==0 or (i+1) % epoch_steps == 0:
        if i>0:
            print('epoch {} done.'.format(int((i+1)/epoch_steps)))
        for j in range(train_steps):
            if (j+1) % 100 == 0:
                print('trainning {} steps. avg mean {:.4f}, avg std {:.4f}'.format(j+1,
                      torch.mean(torch.abs(mean_train)),
                      torch.mean(std_train)))
            batch = buffer.sample()
            obs_train, a_train, r_train, obs_new_train, done_train = \
                [torch.tensor(x,dtype=torch.float) for x in zip(*batch)]
            v_new = v_target(obs_new_train)
            yq = (r_train + gamma*(1-done_train)*v_new[:,0]).detach()

            mean_train, std_train = policy(obs_train)
            dbu = Normal(mean_train,std_train)
            a_sample = dbu.rsample() # reparametrization

            # need to sum the logp of univariate gaussians
            logp = torch.sum(dbu.log_prob(a_sample),dim=1)
            tanh2 = torch.pow(torch.tanh(a_sample),2)
            logp_tanh = torch.sum(torch.log(1-tanh2+EPS),dim=1)
            logp -= logp_tanh
            a_sample = torch.tanh(a_sample)
            
            q_val = q(obs_train,a_sample)
            q2_val = q2(obs_train,a_sample.detach())
            q_both = torch.cat((q_val,q2_val),dim=1)
            q_min, _ = torch.min(q_both,dim=1)
            yv = (q_min - alpha*logp).detach()
            
#            break
            # don't use q_val here!
            q_val_train = q(obs_train,a_train)
            loss_q = F.mse_loss(q_val_train[:,0],yq)
            q_optim.zero_grad()
            loss_q.backward()
            q_optim.step()

            # don't use q2_val here!
            q2_val_train = q2(obs_train,a_train)
            loss_q2 = F.mse_loss(q2_val_train[:,0],yq)
            q2_optim.zero_grad()
            loss_q2.backward()
            q2_optim.step()

            v_val = v(obs_train)[:,0]
            loss_v = F.mse_loss(v_val,yv)
            v_optim.zero_grad()
            loss_v.backward()
            v_optim.step()

            # should I use the same a_sample for policy training???
            # OpenAI uses the same a_sample for policy training
            # negative for maxmization!
            # notice q_val is wrong! Use q_val[:,0]
            loss = - torch.mean(q_val[:,0] - alpha*logp) 
            policy_optim.zero_grad()
            loss.backward()
            policy_optim.step()
            # remeber to open it up again for q

            for w, w_target in zip(v.parameters(),v_target.parameters()):
                w_target.data = rho*w_target.data + (1-rho)*w.data

    
#    break
    obs_tensor = torch.tensor(obs,dtype=torch.float)
    mean,std = policy(obs_tensor)
    dbu = Normal(mean,std)
    # no reparemterization necessary when running
    a = torch.tanh(dbu.sample()).data.tolist()
    obs_new, r, done, info = env.step(a)
    if r < -90:
        r = min_r
    buffer.add([obs,a,r,obs_new[:14], done])
    obs = obs_new[:14]
    eps_len += 1
    if done:
        print('episode end in {} steps.'.format(eps_len))
        eps_len = 0
        obs = env.reset()
        obs = obs[:14]

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
        mean, std = policy(obs_tensor)
#            print(p)
#        dbu = Normal(mean,std)
#        a = dbu.sample()
#        a = torch.tanh(dbu.sample()).data.tolist()
        a = torch.tanh(mean).data.tolist()
            #g = grad(obj,policy.parameters())
        obs_new, r, done, info = env.step(a)
#        action_explore = np.clip(action + noise(action),-1,1)
#        print(done)
        #history.append([obs,action[0],reward,obs_new])
        obs_new = obs_new[:14]
        obs = obs_new
        t += 1
        r_total += r
    print('points: {}'.format(r_total))


#torch.save(policy.state_dict(), 'policy_sac_310.pkl')
#torch.save(q.state_dict(), 'q_sac_310.pkl')

#torch.save(q2.state_dict(), 'q2_td3_normal_walking_mid_trainx.pkl')
