import torch
import sys
from lib import *
import gym


obs_dim = 4
action_dim = 1

policy = Policy(obs_dim,action_dim,64)
policy.load_state_dict(torch.load('models/policy_{}.pkl'.format(sys.argv[1])))


env = gym.make('CartPole-v1')
for i_episode in range(3):
    obs = env.reset()
    t = 0
    r_total = 0
    done = False
    while not done:
        if (t+1)%100 == 0:
            print(t+1)
        env.render()
        obs_tensor = torch.from_numpy(obs.astype('float32'))
        a = policy(obs_tensor).data.tolist()[0]
        a = 0 if a < 0.5 else 1
        # a = env.action_space.sample()
        obs_new, r, done, info = env.step(a)
        obs = obs_new
        t += 1
        r_total += r
    print('{}th run points: {}'.format(i_episode, r_total))