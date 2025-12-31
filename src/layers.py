import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import RMSNorm
from attention import MultiHeadAttention

class SwiGLU(nn.Module):

    def __init__(self, d_model, hidden_dim):
        super().__init__()

        # up projection 1 (d_model -> hidden_dim, SiLU applied here)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        
        # up projection 2 (d_model -> hidden_dim, no SiLU applied)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)

        # final down projection (hidden_dim -> d_model, acts as gate)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # return calcualted SwiGLU
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    
class Transformer(nn.Module):

    def __init__(self, d_model, hidden_dim, max_seq_len, n_heads):
        super().__init__()
        
        # init layers
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.swiglu = SwiGLU(d_model, hidden_dim)

    def forward(self, x):
        # compute norm of input
        normx = self.rms1(x)

        # pass norms to MHA and add to residual stream
        x = x + self.mha(normx, normx, normx)
        
        # pass norms to SwiGLU and add to residual stream
        x = x +  self.swiglu(self.rms2(x))
        return x
    
if __name__ == "__main__":
    vocab_size = 50257
    d_model = 512
    hidden_dim = 2048
    max_seq_len = 1024
    batch_size = 32
    n_heads = 8
    transformer = Transformer(d_model, hidden_dim, max_seq_len, n_heads)
    x = torch.randn(batch_size, max_seq_len, d_model)
    print(f"output shape: {transformer(x).shape}")