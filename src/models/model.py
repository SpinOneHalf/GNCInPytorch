import torch
import torch.nn as nn
class LinearModel(nn.Module):
    def __init__(self,
                 k: float = 1.,
                 gamma: float = 1.):
        super().__init__()
        self.k = nn.Parameter(torch.Tensor([k]))
        self.g = nn.Parameter(torch.Tensor([gamma]))

    def forward(self,x:torch.tensor,t:torch.tensor,u:callable):
        dx = torch.Tensor([0, 0])
        dx[0] = x[1]
        dx[1] =   (-self.k * x[0] - self.g * x[1] + u(x, t))
        return dx



class CartModel(nn.Module):
    def __init__(self,
                 L:float=1.,
                 uk:float=1.):
        super().__init__()
        self.L=nn.Parameter(torch.tensor([L]))
        self.uk=nn.Parameter(torch.tensor([uk]))
    def forward(self,x,t,u):
        f = torch.zeros(4)
        angle_wheel,a=u(x,t)
        _,_,angle_car,s=x
        f[0] = s * torch.cos(angle_car)
        f[1] = s * torch.sin(angle_car)
        f[2] = s / self.L * torch.tan(angle_wheel)
        f[3] = a*self.uk
        return f