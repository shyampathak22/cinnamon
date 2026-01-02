import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Uses rsqrt for efficiency (1/sqrt is faster than sqrt + division).
    """
    def __init__(self, d_model, eps):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # rsqrt = 1/sqrt, more efficient than sqrt + divide
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma