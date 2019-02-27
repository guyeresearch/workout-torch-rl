import gym
import numpy as np
import torch
from model import Policy, Val
import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle


#env knowledge
obs_dim = 4
action_dim = 1

# parameters
epochs = 200
D_size = 10
gamma = 0.99
lda = 0.97 # for generalized advantage esitmate

val_epochs = 20
val_batch = 32
val_lr = 1e-3

policy_lr = 3e-4


policy = Policy(obs_dim,action_dim)
val = Val(obs_dim)

val_optim = optim.Adam(val.parameters(), lr=val_lr)
policy_optim = optim.Adam(policy.parameters(),lr=policy_lr)

env = gym.make('CartPole-v1')
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
            obs_tensor = torch.from_numpy(
                obs[None,:].astype('float32'))
            p = policy(obs_tensor)
            p_value = p.data.numpy()[0]
#            print(p_value)
            a = np.random.binomial(1,p_value)[0]
            a_tensor = torch.tensor(a,dtype=torch.float32)
            loss = -torch.log(a_tensor*p + (1-a_tensor)*(1-p))
            #g = grad(obj,policy.parameters())
            obs_new, r, done, info = env.step(a)
            eps.append([obs,loss,r,obs_new])
            obs = obs_new
            i += 1
        print('end in {} steps'.format(i+1))
        D.append(eps)


    #fit val
    mat = []
    y = []
    for eps in D:
        v = 0
        for item in eps[::-1]:
            mat.append(item[0])
            v = v*gamma + item[2]
            y.append(v)
    mat = np.array(mat).astype('float32')
    y = np.array(y).astype('float32')
    for _ in range(val_epochs):
        #print('=======')
        rand_idx = list(range(len(mat)))
        shuffle(rand_idx)
        for l in qn.chunks(rand_idx,val_batch):
            chunk_mat = torch.from_numpy(mat[l,:])
            chunk_y = torch.from_numpy(y[l,None])
            v_pred = val(chunk_mat)
            v_loss = F.mse_loss(v_pred,chunk_y)
            #print(v_loss)
            val_optim.zero_grad()
            v_loss.backward()
            val_optim.step()

    #fit policy
    policy_optim.zero_grad()
    for eps in D:
        delta_cum = 0
        for item in eps[::-1]:
            # delta for GAE
            v_obs_new = val(torch.tensor(item[-1],dtype=torch.float32))
            v_obs = val(torch.tensor(item[0],dtype=torch.float32))
            delta = gamma*v_obs_new+r - v_obs
            delta_cum = delta[0].detach() + gamma*lda*delta_cum
            # accumulate grads
            (torch.mean(item[1])*delta_cum).backward()
#    for p in policy.parameters():
#        p.grad /= D_size
    policy_optim.step()


