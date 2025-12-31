import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import RMSNorm
from attention import MultiheadLatentAttention

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
    
class MoE(nn.Module):
    def __init__(self, n_routed, n_shared, top_k, d_model, hidden_dim, expert_scale, gamma):
        super().__init__()
        routed_hidden = hidden_dim // expert_scale
        self.routed_experts = nn.ModuleList([SwiGLU(d_model, routed_hidden) for _ in range(n_routed)])
        self.shared_experts = nn.ModuleList([SwiGLU(d_model, hidden_dim) for _ in range(n_shared)])
        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.top_k = top_k
        self.n_routed = n_routed
        self.gamma = gamma
        self.register_buffer("expert_bias", torch.zeros(n_routed))

    def update_bias(self, top_idx):
        flat_idx = top_idx.flatten()
        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed)
        total_selections = top_idx.numel()
        expected = total_selections / self.n_routed
        self.expert_bias += self.gamma * (expected - expert_counts) / expected
        self.expert_bias.clamp_(-1, 1)

    def forward(self, x):
        batch, seq, d = x.shape
        shared_out = sum([expert(x) for expert in self.shared_experts])
        scores = torch.sigmoid(self.router(x))
        selection_scores = scores + self.expert_bias
        _, top_idx = selection_scores.topk(self.top_k, dim=-1)
        top_scores = scores.gather(-1, top_idx)
        gates = torch.zeros_like(scores).scatter(-1, top_idx, top_scores)
        routed_out = torch.zeros_like(x)
        for i, expert in enumerate(self.routed_experts):
            mask = (top_idx == i).any(dim=-1)

            if mask.any():
                gate_values = gates[:, :, i][mask]
                routed_out[mask] += expert(x[mask]) * gate_values.unsqueeze(-1)
        if self.training:
            self.update_bias(top_idx)
        return shared_out + routed_out

class Transformer(nn.Module):

    def __init__(self, d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_rope, n_routed, n_shared, top_k, expert_scale, gamma):
        super().__init__()
        
        # init layers
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.attn = MultiheadLatentAttention(d_model, d_ckv, d_cq, n_heads, d_head, d_rope, max_seq_len)
        self.moe = MoE(n_routed, n_shared, top_k, d_model, hidden_dim, expert_scale, gamma)

    def forward(self, x):
        # compute norm of input
        normx = self.rms1(x)

        # pass norms to MHA and add to residual stream
        x = x + self.attn(normx)
        
        # pass norms to SwiGLU and add to residual stream
        x = x +  self.moe(self.rms2(x))
        return x
    
if __name__ == "__main__":
    vocab_size = 50257
    d_model = 512
    hidden_dim = 2048
    max_seq_len = 1024
    batch_size = 32
    n_heads = 8
    d_ckv=256
    d_cq=256
    d_head=64
    d_rope=32
    n_routed=8
    n_shared=1
    top_k=2
    expert_scale=4
    gamma=0.001
    transformer = Transformer(d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_rope, n_routed, n_shared, top_k, expert_scale, gamma)
    x = torch.randn(batch_size, max_seq_len, d_model)
    print(f"output shape: {transformer(x).shape}")
    moe = MoE(n_routed=8, n_shared=1, top_k=2, d_model=512, hidden_dim=2048, expert_scale=4, gamma=0.001)
    x = torch.randn(4, 128, 512)
    out = moe(x)
    print(f"MoE output shape: {out.shape}")
    print(f"MoE params: {sum(p.numel() for p in moe.parameters()):,}")