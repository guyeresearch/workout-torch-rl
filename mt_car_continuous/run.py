import gym
import numpy as np
import torch
from model import Policy, Val
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal

#env knowledge
obs_dim = 2
action_dim = 1

# parameters
epochs = 200
D_size = 5
gamma = 0.98
lda = 0.97 # for generalized advantage esitmate

val_epochs = 5
val_lr = 1e-3

policy_lr = 3e-4


policy = Policy(obs_dim,action_dim)
val = Val(obs_dim)

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

env = gym.make('MountainCarContinuous-v0')
#%%

primer_k = 5

eps_lens = []
for k in range(epochs):
    print('epoch: {}'.format(k))
    # collect D
    D = []
    while len(D) < D_size:
        eps = []
        obs = env.reset()
        done = False
        i = 0
        while not done:
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            mean = policy(obs_tensor)
#            print(p)
            dbu = Normal(mean[0],0.5)
            if k < primer_k:
                a = float(env.action_space.sample()[0])
            else:
                a = dbu.sample().data.tolist()
            logp = dbu.log_prob(a)
            #g = grad(obj,policy.parameters())
            obs_new, r, done, info = env.step([a])
            eps.append([obs,a,r,logp,obs_new])
            obs = obs_new
            i += 1
        if len(eps) < 999:
            print('end in {} steps. total D:{}'.format(i+1,len(D)))
            D.append(eps[-300:])
    eps_lens.append(np.mean([len(x) for x in D]))



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
        print(v_loss)
        val_optim.zero_grad()
        v_loss.backward()
        val_optim.step()


 #fit policy simple
#    scalar = 1
    scalar = D_size
#    scalar = D_size*sum([len(x) for x in D])
    policy_optim.zero_grad()
    for eps in D:
        delta_cum = 0
        for item in eps[::-1]:
            obs,a,r,logp, obs_new = item
            # delta for GAE
            obs = torch.from_numpy(obs.astype('float32'))
            obs_new = torch.from_numpy(obs_new.astype('float32'))
            delta = (gamma*val(obs_new)+r - val(obs))[0].detach()
            delta_cum = delta + gamma*lda*delta_cum
#            print(delta,delta_cum)
            # accumulate grads
            (-logp*delta_cum/scalar).backward()
#    for p in policy.parameters():
#        p.grad /= D_size
    policy_optim.step()

#%%
#import matplotlib.pyplot as plt
##
##plt.plot(eps_lens1)
#eps_lensd = qn.load('eps_lens_std.pkl')
#plt.plot(eps_lensd)
#plt.plot(eps_lens[:200])




#%%
#env = gym.make('CartPole-v1')
#for i_episode in range(2):
#    obs = env.reset()
#    for t in range(200):
#        if (t+1)%100 == 0:
#            print(t+1)
#        env.render()
#        obs_tensor = torch.from_numpy(obs.astype('float32'))
#        p = policy(obs_tensor)
##            print(p)
#        categorical = Categorical(p)
#        a = categorical.sample()
#        logp = torch.log(p[a])
#            #g = grad(obj,policy.parameters())
#        obs_new, r, done, info = env.step(a.data.tolist())
##        action_explore = np.clip(action + noise(action),-1,1)
##        print(done)
#        #history.append([obs,action[0],reward,obs_new])
#        obs = obs_new
