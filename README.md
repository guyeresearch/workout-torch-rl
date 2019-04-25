# Practice Reinforcement Learning with PyTorch
The code is orgnized based on the OpenAI gym environments. Each solver consists of two files: `model.py/lib.py` and `run.py`. They are self-contained and do not depend on other py files.

`model.py/lib.py` contains the model definitions and some util functions. `run.py` runs the model in the gym envrionments. It has three sections: hyperparameters and initilization, training loop and evaluation loop. 

I usually run the `run.py` in Spyder IDE that supports section-based execution. After running the training loop in Spyder, I run the evaluation loop to visualize and evaluate the result.

I think this way of orgnaization is most friendly to beginners in RL and simpler than the way the spinningup code is strucutured.

### Current progress is as following:

##### CartPole-v1:

VPG (Binomial policy)

VPG (Categorical policy)

[Evolution Stategy](https://arxiv.org/abs/1703.03864) (multi-processes implementation with OpenMPI)



##### MountainCarContinuous-v0:

VPG (not working)

off-policy VPG (not working)

DDPG

DDPG with [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (More efficient for sparse rewards)




##### BipedalWalker-v2:

VPG (Gaussian Policy)

VPG (Multivariate Gaussian Policy)

TRPO

PPO

TD3

SAC

[Evolution Stategy](https://arxiv.org/abs/1703.03864) (multi-processes implementation with OpenMPI)





