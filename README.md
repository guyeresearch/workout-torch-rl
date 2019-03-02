# Practice Reinforcement Learning with PyTorch
The code is orgnized based on the OpenAI gym environments. Each solver should consists of two files: `model.py/lib.py` and `run.py`. They are self-contained and do not depend on other py files.

`model.py/lib.py` contains the model definitions and some util functions. `run.py` runs the model in the gym envrionments. It has three sections: hyperparameters and initilization, training loop and evaluation loop. 

I usually run the `run.py` in Spyder IDE that supports section-based execution. So after running the training loop, I can run the evaluation loop and visuliaze and evaluate the result.

### Current progress is as following:

##### CartPole-v1:

VPG (Binomial policy)

VPG (Categorical policy)




##### MountainCarContinuous-v0:

VPG (not working)

off-policy VPG (not working)

DDPG




##### BipedalWalker-v2:

VPG (Gaussian Policy)

VPG (Multivariate Gaussian Policy)

TRPO

VPG





