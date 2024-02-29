import torch
import torch.nn as nn
class LinearModel(nn.Module):
    def __init__(self,
                 k: float = 1,
                 gamma: float = 1,
                 mass: float = 1):
        super().__init__()
        self.k = nn.Parameter(torch.Tensor([k]))
        self.g = nn.Parameter(torch.Tensor([gamma]))
        self.m = nn.Parameter(torch.Tensor([mass]))

    def forward(self, x: torch.tensor, u: callable, t: torch.tensor):
        dx = torch.Tensor([0, 0])
        dx[0] = x[1]
        dx[1] = (1 / self.m) * (-self.k * x[0] - self.g * x[1] + u(x, t))
        return dx