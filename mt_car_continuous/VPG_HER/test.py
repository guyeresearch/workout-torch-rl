
import gym
import numpy as np

env = gym.make('MountainCarContinuous-v0')
for i_episode in range(1):
   obs = env.reset()
   t = 0
   done = False
   eps = []
   while not done:
       if (t+1)%100 == 0:
           print(t+1)
        
       a = env.action_space.sample()
       obs_new, r, done, info = env.step(a)
       eps.append([obs,a,r,obs_new])
       obs = obs_new
       t += 1

eps_mat = np.array(eps)
print(t,obs,r,done)
print(eps_mat[:,2])