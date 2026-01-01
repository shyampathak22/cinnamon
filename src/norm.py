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
    
if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()
    rms = RMSNorm(cfg.d_model, cfg.rms_eps)
    x = torch.randn((32, cfg.max_seq_len, cfg.d_model))
    norm = rms(x)
    print(f"RMSNorm output shape: {norm.shape}")