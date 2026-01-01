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
                 local_window,
                 n_indexer_heads,
                 rms_eps,
                 rope_base,
                 d_inner):
        super().__init__()
        self.n_layers = n_layers

        # init nn.Sequential using n_layers to create transformer blocks
        self.transformer_layers = nn.Sequential(*[Transformer(d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_rope, n_routed, n_shared, top_k, expert_scale, gamma, k_ts, local_window, n_indexer_heads, rms_eps, rope_base, d_inner) for _ in range(self.n_layers)])

        # init embedding, norm, and lm_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model, eps=rms_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # tie lm_head and embedding weights together (save memory/compute and improve generalization)
        self.lm_head.weight = self.embedding.weight

        self.mtp = MultiTokenPrediction(d_model, rms_eps)

        # apply weight inits
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                # Scale down residual stream projections to reduce variance in grads
                # w_out: MLA output projection, w3: SwiGLU down projection
                if name.endswith('w_out.weight') or name.endswith('w3.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2*self.n_layers)**0.5)
                # Standard init for all other weights
                elif p.dim() >= 2:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids):
        # embed once, reuse for MTP (avoid redundant embedding lookup)
        emb = self.embedding(input_ids)
        x = self.transformer_layers(emb)
        x_2 = self.mtp(x[:, :-1], emb[:, 1:])  # reuse embeddings instead of re-embedding

        # Batch LM head calls: single norm + linear pass
        seq_len_main, seq_len_mtp = x.size(1), x_2.size(1)
        combined = torch.cat([x, x_2], dim=1)
        combined = self.lm_head(self.norm(combined))
        x, x_2 = combined[:, :seq_len_main], combined[:, seq_len_main:]
        return x, x_2
    
if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()
    cfg.n_layers = 4  # reduced for testing
    model = Cinnamon(
        cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.hidden_dim, cfg.n_heads,
        cfg.max_seq_len, cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_rope,
        cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.expert_scale, cfg.gamma,
        cfg.k_ts, cfg.local_window, cfg.n_indexer_heads, cfg.rms_eps, cfg.rope_base,
        cfg.d_inner
    )
    x = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))
    out, mtp_out = model(x)
    print(f"output shape: {out.shape}, mtp: {mtp_out.shape}")
    print(f"Total: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")