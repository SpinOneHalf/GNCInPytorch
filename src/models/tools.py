import torch
import torch.nn as nn
from torchdiffeq import odeint
class CombinedModelTrainer(nn.Module):
    def __init__(self,model:nn.Module,controller:nn.Module):
        super().__init__()
        self.model=model
        self.controller=controller
    def forward(self,t,x):
        return self.model(x,t,self.controller)

def generate_data(model,controller,x0,ts):
    #Combine model and controller for ode solver
    combine_generator=CombinedModelTrainer(model,controller)
    xs=odeint(combine_generator,x0.float(),ts.float())
    print("DONE")
    return xs
def fit_model():
    pass