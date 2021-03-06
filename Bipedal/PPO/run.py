import gym
import numpy as np
import torch
from lib import *
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal

#env knowledge
obs_dim = 14
action_dim = 4

# parameters
epochs = 500
D_size = 10
gamma = 0.99
lda = 0.95 # according to paper

constrain = 0.2 # according to paper
update_steps = 80
target_kl = 0.01

EPS = 1e-8

val_epochs = 20
val_lr = 1e-3

policy_lr = 3e-4


policy = Policy(obs_dim,action_dim)
val = Val(obs_dim)

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

#%%
std = 1.
eps_lens = []
env = gym.make('BipedalWalker-v2')
#%%
for k in range(epochs):
    print('epoch: {}, std: {}'.format(k,std))
    # collect D
    D = []
    for _ in range(D_size):
        eps = []
        obs = env.reset()
        obs = obs[:14]
        done = False
        i = 0
        while not done:
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            mean = policy(obs_tensor)
#            print(p)
            dbu = Normal(mean,std)
            a = dbu.sample()
            logp = torch.sum(dbu.log_prob(a))
            #g = grad(obj,policy.parameters())
            obs_new, r, done, info = env.step(a.data.tolist())
            obs_new = obs_new[:14]
            if r < -90:
                r = -15
            eps.append([obs,a,r,logp,obs_new])
            obs = obs_new
            i += 1
        print('end in {} steps'.format(i+1))
        D.append(eps)
    eps_lens.append(np.mean([len(x) for x in D]))
    if (k+1) % 100 == 0:
        std = std if std <= 0.3 else std*0.7



    #fit val
    mat = []
    y = []
    for eps in D:
        v = 0
        for item in eps[::-1]:
            mat.append(item[0])
            v = v*gamma + item[2]
            y.append(v)
    mat = torch.from_numpy(np.array(mat,dtype='float32'))
    y = torch.from_numpy(np.array(y,dtype='float32')[:,None])
    for _ in range(val_epochs):
        y_pred = val(mat)
        v_loss = F.mse_loss(y_pred,y)
#        print(v_loss)
        val_optim.zero_grad()
        v_loss.backward()
        val_optim.step()

    
    # rewrite policy step for easy logp esitmate in line search
    # check this new implementation to see if it works in vpg. DONE
    #fit policy simple
    scalar = D_size
    obsx = []
    ax = []
    logpx = []
    obs_newx = []
    delta_cumx = []
    for eps in D:
        obs, a, r, logp, obs_new = [x for x in zip(*eps)]
        obs = torch.tensor(obs,dtype=torch.float)
        a = torch.stack(a)
        r = torch.tensor(r, dtype=torch.float)
        # don't use torch.tensor(logp) which will remove the graph
        logp = torch.stack(logp)
        obs_new = torch.tensor(obs_new,dtype=torch.float)
        
#        # this is wrong!! understand the broadcast rule: 3x1 + 3 matrix
#        # yields 3x3 matrix
#        delta = (gamma*val(obs_new) + r - val(obs))[0].detach()
        delta = (gamma*val(obs_new)[:,0] + r - val(obs)[:,0]).detach()
        delta_cum = []
        running = 0
        # torch does not support negative idx yet
        for item in delta.data.tolist()[::-1]:
            running = item + gamma*lda*running
            delta_cum.append(running)
        delta_cum = torch.tensor(delta_cum[::-1])
        obsx.append(obs)
        ax.append(a)
        logpx.append(logp)
        obs_newx.append(obs_new)
        delta_cumx.append(delta_cum)
    
    obs = torch.cat(obsx)
    a = torch.cat(ax)
    logp = torch.cat(logpx)
    obs_new = torch.cat(obs_newx)
    delta_cum = torch.cat(delta_cumx)
    
#    break
    
    logp_k = logp.detach()
    mean_k = policy(obs).detach()
    var = std*std
    for i in range(update_steps):
        if i>0:
            mean = policy(obs)
            dbu = Normal(mean,std)
            logp = torch.sum(dbu.log_prob(a),dim=1)
        p_ratio = torch.exp(logp-logp_k)
        left = p_ratio*delta_cum
    
        indicator = (delta_cum>0).float()*(1+constrain) + \
            (delta_cum<0).float()*(1-constrain)
        right = indicator*delta_cum
    
        L, _ = torch.min(torch.stack((left,right)),dim=0)
        policy_optim.zero_grad()
        (-L.mean()).backward()
        policy_optim.step()

        if i > 0:
            kl = torch.sum(torch.pow((mean-mean_k),2)/2/var,dim=1).mean()
            if kl > 1.5*target_kl:
                print('early stopping at step {} with kl {}'.format(i,kl.data.tolist()))
                break
        
        
    
    

#%%
import matplotlib.pyplot as plt
#
#plt.plot(eps_lens1)
eps_lensd = qn.load('eps_lens_std.pkl')
plt.plot(eps_lensd)
plt.plot(eps_lens[:200])




#%%
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
#        dbu = Normal(mean,std)
#        a = dbu.sample()
            #g = grad(obj,policy.parameters())
#        obs_new, r, done, info = env.step(a.data.tolist())
        obs_new, r, done, info = env.step(mean.data.tolist())
#        a = env.action_space.sample()
#        obs_new, r, done, info = env.step(a)
#        action_explore = np.clip(action + noise(action),-1,1)
#        print(done)
        #history.append([obs,action[0],reward,obs_new])
        obs_new = obs_new[:14]
        obs = obs_new
        t += 1
        r_total += r
    print('points: {}'.format(r_total))
#%%
torch.save(policy.state_dict(), 'policy_PPO_constrain_0.1_300.pkl')
torch.save(val.state_dict(), 'val_PPO_constrain_0.1_300.pkl')