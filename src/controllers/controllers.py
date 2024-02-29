import torch
import torch.nn as nn


def interpolate_to_support(x: torch.Tensor, y: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    This is a rudimentary implementation of numpy.interp for the 1D case only. If the x values are not unique, this behaves differently to np.interp.
    :param x: The original coordinates.
    :param y: The original values.
    :param support: The support points to which y shall be interpolated.
    :return:
    """
    # Evaluate the forward difference for all except the edge points
    slope = torch.zeros_like(x)
    slope[1:-1] = ((y[1:] - y[:-1]) / (x[1:] - x[:-1]))[1:]

    # Evaluate which of the support points are within the range of x
    support_nonzero_mask = (support >= x.min()) & (support <= x.max())
    # Subset the support points accordingly
    support_nonzero = support[support_nonzero_mask]
    # Get the indices of the closest point to the left for each support point
    support_insert_indices = torch.searchsorted(x, support_nonzero)
    # Get the offset from the point to the left to the support point
    support_nonzero_offset = support_nonzero - x[support_insert_indices]
    # Calculate the value for the nonzero support: value of the point to the left plus slope times offset
    support_nonzero_values = y[support_insert_indices] + slope[support_insert_indices - 1] * support_nonzero_offset

    # Create the output tensor and place the nonzero support
    support_values = torch.zeros_like(support).float()
    support_values[support_nonzero_mask] = support_nonzero_values
    return support_values


class LookUpTable(nn.Module):
    def __init__(self, us: torch.Tensor, t_range: torch.Tensor):
        super().__init__()
        self.us = nn.Parameter(us)
        self.t_range = nn.Parameter(t_range,requires_grad=False)
    def forward(self,x,t):
        return interpolate_to_support(self.t_range,self.us,t)
