import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Transformer, MultiTokenPrediction
from norm import RMSNorm

class Cinnamon(nn.Module):

    def __init__(self,
                 d_model,
                 n_layers,
                 vocab_size, 
                 hidden_dim, 
                 n_heads, 
                 max_seq_len,
                 d_ckv, 
                 d_cq, 
                 d_head, 
                 d_rope, 
                 n_routed, 
                 n_shared, 
                 top_k, 
                 expert_scale,
                 gamma,
                 k_ts,
                 local_window):
        super().__init__()
        self.n_layers = n_layers
        
        # init nn.Sequential using n_layers to create transformer blocks
        self.transformer_layers = nn.Sequential(*[Transformer(d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_rope, n_routed, n_shared, top_k, expert_scale, gamma, k_ts, local_window) for _ in range(self.n_layers)])

        # init embedding, norm, and lm_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # tie lm_head and embedding weights together (save memory/compute and improve generalization)
        self.lm_head.weight = self.embedding.weight

        self.mtp = MultiTokenPrediction(d_model)

        # apply weight inits
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                
                # scale down the residual streams to reduce variance in grads
                if name.endswith('o.weight') or name.endswith('w3.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2*self.n_layers)**0.5)

                # init all others to low value with low std.
                elif p.dim() >= 2:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids):
        # pass through layers
        x = self.embedding(input_ids)
        x = self.transformer_layers(x)
        x_2 = self.mtp(x[:, :-1], self.embedding(input_ids[:, 1:]))
        x = self.norm(x)
        x = self.lm_head(x)
        x_2 = self.lm_head(self.norm(x_2))
        return x, x_2
    
if __name__ == "__main__":
    vocab_size = 50257
    d_model = 512
    max_seq_len = 1024
    batch_size = 32
    n_heads = 8
    n_layers = 4
    hidden_dim = 1024
    d_ckv=256
    d_cq=256
    d_head=64
    d_rope=32
    n_routed=8
    n_shared=1
    top_k=2
    expert_scale=4
    gamma=0.001
    k_ts=256
    local_window=128
    model = Cinnamon(d_model,
                 n_layers,
                 vocab_size, 
                 hidden_dim, 
                 n_heads, 
                 max_seq_len,
                 d_ckv, 
                 d_cq, 
                 d_head, 
                 d_rope, 
                 n_routed, 
                 n_shared, 
                 top_k, 
                 expert_scale,
                 gamma,
                 k_ts,
                 local_window)
    x = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    out, mtp_out = model(x)
    print(f"output shape: {out.shape}, mtp: {mtp_out.shape}")
    print(f"Total: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")