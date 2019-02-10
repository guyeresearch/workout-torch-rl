import gym
import numpy as np
from model import Critic, Actor
import random
import qn

import torch
import torch.nn.functional as F
from torch import nn, optim

gamma = 0.9
batchsize = 64
critic_lr = 1e-3
actor_lr = 1e-4

buffer = qn.load('buffer.pkl')


s, a, r, s_next = list(zip(*buffer))
r = np.array(r)
r[r<-50] = -5
r[r>0] = r[r>0]*3

a_next = list(a[1:]) + [a[0]]

buffer2 = list(zip(s,a,r,s_next,a_next))


critic = Critic(24,4)
target_critic = Critic(24,4)
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

actor = Actor(24,4)
target_actor = Actor(24,4)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()

critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
#%%

critic.train()
actor.train()
for i in range(int(1e4)):
    if (i+1)%100 == 0:
        print(i+1)
    batch = random.sample(buffer2,batchsize)
    s, a, r, s_next, a_next = list(zip(*batch))
    s = torch.from_numpy(np.array(s,dtype='float32'))
    a = torch.from_numpy(np.array(a,dtype='float32'))
    r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
    s_next = torch.from_numpy(np.array(s_next,dtype='float32'))

#    non_final_mask = (r < 10).squeeze()
#    s_next_non_final = s_next[non_final_mask,:]
#    a_next_non_final = a_next[non_final_mask,:]
#
#    q_next = torch.zeros((batchsize,1))
#    q_next[non_final_mask] = target(s_next_non_final,a_next_non_final).detach()
    critic.train()
    a_next = target_actor(s_next).detach()
    q_next = target_critic(s_next,a_next).detach()
    y = r + gamma*q_next
    q = critic(s,a)
    critic_loss = F.mse_loss(q,y)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    critic.eval()
    a_predict = actor(s)
    q = critic(s,a_predict)
    actor_loss = -q.mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


    if (i+1)%50 == 0:
        target_critic.load_state_dict(critic.state_dict())
        target_actor.load_state_dict(actor.state_dict())



#%%
#critic.eval()
#s, a, r, s_next, a_next = list(zip(*buffer2[:200]))
#s = torch.from_numpy(np.array(s,dtype='float32'))
#a = torch.from_numpy(np.array(a,dtype='float32'))
#r = torch.from_numpy(np.array(r,dtype='float32')[:,None])
#s_next = torch.from_numpy(np.array(s_next,dtype='float32'))
#a_next = torch.from_numpy(np.array(a_next,dtype='float32'))
#
#q = critic(s,a).detach().numpy()
#cc = np.hstack([np.array(s),np.array(a), q])


#%%
def noise_factory(mu=0,theta=1,sigma=0.5):
    def noise(x):
        return theta * (mu - x) + sigma * np.random.randn(len(x))
    return noise
noise = noise_factory()
        
actor.eval()
history = []
env = gym.make('BipedalWalker-v2')
for i_episode in range(2):
    obs = env.reset()
    for t in range(2000):
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs[None,:].astype('float32'))
        action = actor(obs_tensor).data.numpy()[0]
        action = np.clip(action,-1,1)
#        action = action + noise(action)
        obs_new, reward, done, info = env.step(action)
        history.append([obs.tolist(),action,reward,obs_new])
        obs = obs_new

        if done:
            break

s, a, r, s_next = list(zip(*history))

