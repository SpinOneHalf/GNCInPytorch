from src.utils import CombinedModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint




def hot_shot(x0:torch.Tensor,
             xf:torch.Tensor,
             model:nn.Module,
             linear_table:nn.Module,
             tf:torch.Tensor,
             optimizer:optim.Optimizer=optim.Adadelta,
             optimizer_dict={"lr":.01},
             loss_func=nn.MSELoss()):
    trainer=CombinedModel(model,linear_table)
    for p in trainer.model.parameters():
        p.requires_grad=False
    for p in trainer.controller.parameters():
        p.requires_grad=True
    local_optimizer=optimizer(trainer.parameters(),**optimizer_dict)
    for e in range(150):
        path=odeint(trainer,x0,tf)
        xfp=path[-1][0:2]
        local_optimizer.zero_grad()
        loss=loss_func(xfp,xf[0:2])
        loss.backward()
        print(loss)
        local_optimizer.step()
    return path,trainer.controller
