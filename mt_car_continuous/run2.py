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
gamma = 0.99
lda = 0.96 # for generalized advantage esitmate

val_epochs = 20
val_lr = 1e-3

policy_lr = 1e-3
std = 0.3


policy = Policy(obs_dim,action_dim)
val = Val(obs_dim)

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

env = gym.make('MountainCarContinuous-v0')
#%%
import matplotlib.pyplot as plt


vx = np.linspace(env.observation_space.low[0],env.observation_space.high[0])
vy = np.linspace(env.observation_space.low[1],env.observation_space.high[1])
vxm,vym = np.meshgrid(vx,vy)
vxf = np.ravel(vxm)
vyf = np.ravel(vym)
vzf = np.array([val(torch.tensor(item,dtype=torch.float32))[0].data.tolist() 
    for item in zip(vxf,vyf)])
vzm = vzf.reshape(vxm.shape)
azf = np.array([policy(torch.tensor(item,dtype=torch.float32))[0].data.tolist() 
    for item in zip(vxf,vyf)])
azm = azf.reshape(vxm.shape)

    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a basic wireframe.
ax.plot_wireframe(vxm, vym, vzm, rstride=2, cstride=2)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a basic wireframe.
ax.plot_wireframe(vxm, vym, azm, rstride=2, cstride=2)

plt.show()


#%%
##D_total = []
#D_total_size = 50
#while len(D_total) < D_total_size:
#    eps = []
#    obs = env.reset()
#    done = False
#    i = 0
#    while not done:    
#        #g = grad(obj,policy.parameters())
#        a = env.action_space.sample()
#        obs_new, r, done, info = env.step(a)
#        eps.append([obs,a,r,obs_new])
#        obs = obs_new
#        i += 1
#    if len(eps) < 999:
#        print('end in {} steps. total D:{}'.format(i+1,len(D_total)))
#        D_total.append(eps[-300:])

D_total = qn.load('primer_D.pkl')


#%%
eps_lens = []
for D in qn.chunks(D_total,5):

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
    mat[:,1] = mat[:,1]*10
    y = torch.from_numpy(np.array(y,dtype='float32')[:,None])
    for _ in range(val_epochs):
        y_pred = val(mat)
        v_loss = F.mse_loss(y_pred,y)
        print(v_loss)
        val_optim.zero_grad()
        v_loss.backward()
        val_optim.step()


    scalar = D_size
    policy_optim.zero_grad()
    for eps in D:
        delta_cum = 0
        for item in eps[::-1]:
            obs,a,r = item[:3]
            obs = np.array([obs[0],obs[1]*10])
            obs_new = item[-1]
            obs_new = np.array([obs_new[0],obs_new[1]*10])
            obs_tensor = torch.from_numpy(obs.astype('float32'))
            mean = policy(obs_tensor)
            dbu = Normal(mean,std)
            logp = dbu.log_prob(a.tolist()[0])[0]
            # delta for GAE
            obs = torch.from_numpy(obs.astype('float32'))
            obs_new = torch.from_numpy(obs_new.astype('float32'))
            delta = (gamma*val(obs_new)+r - val(obs))[0].detach()
            delta_cum = delta + gamma*lda*delta_cum
#            print(delta_cum)
            p = torch.exp(logp)
            # off-policy vpg learning
            (-p*logp*delta_cum/scalar).backward()
#        break
    policy_optim.step()

#%%
D2 = []
for k in range(1000):
    eps = []
    obs = env.reset()
    done = False
    i = 0
    while not done:    
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        mean = policy(obs_tensor)
        dbu = Normal(mean,std)
        a = dbu.sample().data.tolist()
        obs_new, r, done, info = env.step(a)
        eps.append([obs,a,r,obs_new])
        obs = obs_new
        i += 1
    if len(eps) < 999:
        print('end in {} steps. total D:{}. k: {}'.format(i+1,len(D2),k))
        D2.append(eps[-300:])
    



#%%
env = gym.make('MountainCarContinuous-v0')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        mean = policy(obs_tensor)
#     
            #g = grad(obj,policy.parameters())
        obs_new, r, done, info = env.step(mean.data.tolist())
#        action_explore = np.clip(action + noise(action),-1,1)
#        print(done)
        #history.append([obs,action[0],reward,obs_new])
        obs = obs_new
