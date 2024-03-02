import torch
import torch.optim as optim
import torch.nn as nn
from torchdiffeq import odeint
from src.utils import  CombinedTrainer

def generate_data(model,controller,x0,ts):
    #Combine model and controller for ode solver
    combine_generator=CombinedTrainer(model,controller)
    xs=odeint(combine_generator,x0.float(),ts.float())
    return xs

DEFAULT_DICS={"lr":.001,
              "weight_decay":0.0}
def fit_model(model:nn.Module,
              xs:torch.Tensor,
              us,
              ts:torch.Tensor,
              loss_func=nn.MSELoss(),
              optimizer:optim.Optimizer=optim.RMSprop,
              optimizer_parameters:dict=DEFAULT_DICS):
    print("DONE")
    combied=CombinedModelTrainer(model,us)
    local_opt=optimizer(combied.parameters(),lr=.4)
    xs.requires_grad=False
    for p in combied.model.parameters():
        p.requires_grad=True
    for p in combied.controller.parameters():
        p.requires_grad=False
    x0 = xs[0, :]
    for e in range(175):
        x0h=odeint(combied,x0,ts)
        loss=loss_func(xs[1:,:],x0h[1:,:])
        local_opt.zero_grad()
        loss.backward()
        local_opt.step()
    return combied.model

