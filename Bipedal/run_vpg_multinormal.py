import gym
import numpy as np
import torch
from model_vpg import Policy, Val
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.multivariate_normal import MultivariateNormal

#env knowledge
obs_dim = 24
action_dim = 4

# parameters
epochs = 500
D_size = 10
gamma = 0.98
lda = 0.97 # for generalized advantage esitmate

val_epochs = 5
val_lr = 1e-3

policy_lr = 3e-4


policy = Policy(obs_dim,action_dim)
val = Val(obs_dim)
policy.load_state_dict(torch.load('policy_vpg_mode2_300.pkl'))
val.load_state_dict(torch.load('val_vpg_mode2_300.pkl'))

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

def get_cov_mat(diag,factor02,factor01=0, factor23=0):
    cov_mat = torch.tensor([
            [diag,    factor01, factor02, 0],
            [factor01, diag,    0,        0],
            [factor02, 0,       diag,     factor23],
            [0,        0,       factor23, diag]
        ],dtype=torch.float32)
    return cov_mat


#%%

eps_lens = []
env = gym.make('BipedalWalker-v2')
for k in range(epochs):
    print('epoch: {}'.format(k))
    # collect D
    D = []
    for _ in range(D_size):
        eps = []
        obs = env.reset()
        done = False
        i = 0
        while not done:
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            mean = policy(obs_tensor)
#            print(p)
            dbu = MultivariateNormal(mean,get_cov_mat(0.3,0))
            a = dbu.sample()
            logp = dbu.log_prob(a)
            #g = grad(obj,policy.parameters())
            obs_new, r, done, info = env.step(a.data.tolist())
            if r < -90:
                r = -15
            eps.append([obs,a,r,logp,obs_new])
            obs = obs_new
            i += 1
        print('end in {} steps'.format(i+1))
#        if (i+1) == 1601:
#            eps[-1][1] = -50
        D.append(eps)
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
#        print(v_loss)
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
import matplotlib.pyplot as plt
#
#plt.plot(eps_lens1)
eps_lensd = qn.load('eps_lens_std.pkl')
plt.plot(eps_lensd)
plt.plot(eps_lens[:200])




#%%
env = gym.make('BipedalWalker-v2')
for i_episode in range(10):
    obs = env.reset()
    t = 0
    r_total = 0
    done = False
    while not done and t < 2000:
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        mean = policy(obs_tensor)
#            print(p)
        dbu = MultivariateNormal(mean,get_cov_mat(0.3,0))
        a = dbu.sample()
#        obs_new, r, done, info = env.step(a.data.tolist())
        obs_new, r, done, info = env.step(mean.data.tolist())


        obs = obs_new
        t += 1
        r_total += r
    print('points: {}'.format(r_total))
#%%
#torch.save(policy.state_dict(), 'policy_vpg_mode2_300.pkl')
#torch.save(val.state_dict(), 'val_vpg_mode2_300.pkl')