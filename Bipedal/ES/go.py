import torch
import sys
from lib import *
import gym


obs_dim = 14
action_dim = 4

policy = Policy(obs_dim,action_dim,200)
policy.load_state_dict(torch.load('models/policy_{}.pkl'.format(sys.argv[1])))


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
        a = policy(obs_tensor)
        obs_new, r, done, info = env.step(a.data.tolist())
        obs_new = obs_new[:14]
        obs = obs_new
        t += 1
        r_total += r
    print('{}th run points: {}'.format(i_episode, r_total))