import torch.nn as nn
class CombinedModel(nn.Module):
    def __init__(self,model:nn.Module,controller:nn.Module):
        super().__init__()
        self.model=model
        self.controller=controller
    def forward(self,t,x):
        return self.model(x,t,self.controller)