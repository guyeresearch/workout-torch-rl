import torch.nn as nn
import torch
import numpy as np
from lib import *
import torch.nn.functional as F
from torch.autograd import grad


lr = 1e-3
alpha = 0.01
k = 10
batch_size = 10
amp = [0.1,5.]
phase = [0, np.pi]
interval = [-5.,5.]
epochs = 1000

model = Model()
paramReshape = ParamReshape(model)

train_optim = optim.SGD(model.parameters(),lr=alpha)
meta_train_optim = optim.Adam(model.parameters(), lr=lr)

for i in range(epochs):
    grads = 0
    # clone is necessary!!
    theta = [x.data.clone() for x in model.parameters()]
    for j in range(batch_size):
        b_amp = np.random.uniform(low=amp[0],high=amp[1],size=1)[0]
        b_phase = np.random.uniform(low=phase[0],high=phase[1],size=1)[0]
        x = np.random.uniform(low=interval[0],high=interval[1],size=k)
        y = np.sin(x+b_phase)*b_amp
        x_ = np.random.uniform(low=interval[0],high=interval[1],size=k)
        y_ = np.sin(x_+b_phase)*b_amp
        x,y,x_,y_ = torch.tensor(x,dtype=torch.float)[:,None],\
            torch.tensor(y,dtype=torch.float)[:,None],\
            torch.tensor(x_,dtype=torch.float)[:,None],\
            torch.tensor(y_,dtype=torch.float)[:,None]

        # for w_model,w in zip(model.parameters(), theta):
        #     w_model.data = w.data.clone()
        
        pred = model(x)
        loss = F.mse_loss(pred,y)
        train_optim.zero_grad()
        loss.backward(create_graph=True)
        train_optim.step()
        grad_vec = paramReshape.param2vec([x.grad for x in model.parameters()])


        pred = model(x_)
        loss = F.mse_loss(pred,y_)
        train_optim.zero_grad()
        loss.backward(create_graph=False)
        grad_vec_ = paramReshape.param2vec([x.grad for x in model.parameters()])

        for w_model,w in zip(model.parameters(), theta):
            w_model.data = w.data.clone()

        # check if this is the right implementation
        second_deriv = grad(torch.dot(grad_vec_,grad_vec),model.parameters())
        final_grad = grad_vec_ - alpha*second_deriv
        grads += final_grad
    
    for w_model,w in zip(model.parameters(), theta):
        w_model.data = w.data.clone()
    grads_param = paramReshape.vec2param(grads)
    for w_model,g in zip(model.parameters(),grads_param):
        w_model.grad.data = g.data.clone()
    meta_train_optim.step()











