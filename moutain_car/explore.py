import gym
import numpy as np
from model import Critic, Actor
from copy import deepcopy
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable,grad
import torch.nn as nn


critic = Critic(2,1)
actor = Actor(2,1)
actor.eval()
actor(torch.from_numpy(np.array([[1,2]],dtype='float32')))

#critic_ = deepcopy(critic)
#
#critic2 = Critic(2,1)
#
#
#
#params = critic.parameters()
#x = list(params)
#params_ = critic_.parameters()
#x_ = list(params_)
#
#params2 = critic2.parameters()
#x2 = list(params2)


