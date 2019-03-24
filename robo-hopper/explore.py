#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:01:54 2019

@author: qn
"""

import gym
import roboschool
import numpy as np
import torch
from lib import Policy, Q, Buffer

import qn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from random import shuffle
from torch.distributions.normal import Normal
import copy


env = gym.make('RoboschoolHopper-v1')


obs = env.reset()
for i in range(100):
    a = env.action_space.sample()
    env.render()
    obs_new, r, done, info = env.step(a)



env = gym.make('MountainCarContinuous-v0')


import roboschool
import gym

env = gym.make('RoboschoolHopper-v1')
env.reset()
while True:
    env.step(env.action_space.sample())
    env.render()