import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        # init gain and epsilon values
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

        # define rms function
        self.rms = lambda x: ((x**2).mean(-1, keepdim=True) + self.eps).sqrt()

    def forward(self, x):
        # apply RMSNorm
        return x / self.rms(x) * self.gamma
    
if __name__ == "__main__":
    vocab_size = 50257
    d_model = 512
    max_seq_len = 1024
    batch_size = 32
    rms = RMSNorm(d_model)
    x = torch.randn((batch_size, max_seq_len, d_model))
    norm = rms(x)
    print(f"RMS Weights: {(rms.parameters())}\nRMS Output Shape: {norm.shape}")