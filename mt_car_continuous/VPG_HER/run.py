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
import copy

#env knowledge
obs_dim = 2
action_dim = 1
state_dim = obs_dim + 1
goal_pos = np.float64(0.45) # in source code of gym env

# parameters
epochs = 500
D_size = 1
gamma = 0.97
lda = 0.97 # for generalized advantage esitmate

val_epochs = 5
val_lr = 1e-3

policy_lr = 3e-4

policy = Policy(state_dim,action_dim,100)
val = Val(state_dim,100)

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

env = gym.make('MountainCarContinuous-v0')
#%%

primer_k = 5
std = 0.5
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
        obs = np.concatenate((obs,[goal_pos]))
        while not done:
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            mean = policy(obs_tensor)
            dbu = Normal(mean[0],std)
            if k < primer_k:
                a = float(env.action_space.sample()[0])
                logp = dbu.log_prob(a)
                # importantce weight. Divided by 0.5 because uniform distribution is on [-1,1]
                iw = torch.exp(logp).detach()/0.5
            else:
                a = dbu.sample().data.tolist()
                logp = dbu.log_prob(a)
                iw = torch.tensor(1.)

            #g = grad(obj,policy.parameters())
            obs_new, r, done, info = env.step([a])
            obs_new = np.concatenate((obs_new,[goal_pos]))
            eps.append([obs,a,r,iw,logp,obs_new])
            obs = obs_new
            i += 1
        
        D.append(eps)
        # set reward to 99 for hindersight
        if eps[-1][0][0] < goal_pos:
            eps2 = []
            for item in eps:
                obs,a,r,iw,logp,obs_new = item
                eps2.append([obs,a,r,iw,logp.detach(),obs_new])
            eps2 = copy.deepcopy(eps2)
            eps2[-1][2] = 99
            final_state = eps2[-1][0][0]
            eps_goal = []
            for item in eps2:
                obs,a,r,iw,logp,obs_new = item
                obs[-1] = final_state
                obs_tensor = torch.from_numpy(obs.astype('float32'))
                mean = policy(obs_tensor)
                dbu = Normal(mean[0],std)
                logp2 = dbu.log_prob(a)
                if k < primer_k:
                    iw = torch.exp(logp2).detach()/0.5
                else:
                    iw = torch.exp(logp2-logp).detach()
                obs_new[-1] = final_state
                eps_goal.append([obs,a,r,iw,logp2,obs_new])
            D.append(eps_goal)
        print('end in {} steps. total D:{} last pos {}'.format(i+1,len(D),eps[-1][0][0]))

    # eps_lens.append(np.mean([len(x) for x in D]))

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
        # print(v_loss)
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
            obs,a,r,iw,logp, obs_new = item
            # delta for GAE
            obs = torch.from_numpy(obs.astype('float32'))
            obs_new = torch.from_numpy(obs_new.astype('float32'))
            delta = (gamma*val(obs_new)+r - val(obs))[0].detach()
            delta_cum = delta + gamma*lda*delta_cum
#            print(delta,delta_cum)
            # accumulate grads
            (-iw*logp*delta_cum/scalar).backward()
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
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    obs = np.concatenate((obs,[goal_pos]))
    done = False
    while not done:
        env.render()
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        mean = policy(obs_tensor)
        a = mean.data.tolist()
        obs_new, r, done, info = env.step(a)
        obs_new = np.concatenate((obs_new,[goal_pos]))
        obs = obs_new
