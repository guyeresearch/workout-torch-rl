# Practice Reinforcement Learning with PyTorch
The code is orgnized based on the OpenAI gym environments. Each solver generally consists of two files: model.py/lib.py and run.py. They are self-contained and do not depend on other py files to run.

model.py/lib.py contains the model definitions and some util functions. run.py is the script that runs the model in the gym envrionments. It has three sections: hyperparameters and initilization, training loop and evaluation loop. 

I usually run the run.py in Spyder IDE which supports section-based execution. So after running the training loop, I can run the evaluation loop and visuliaze and evaluate the model.

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





