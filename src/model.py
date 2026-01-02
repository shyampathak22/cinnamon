import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Transformer, MTPModule
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
                 d_v,
                 d_rope,
                 n_routed,
                 n_shared,
                 top_k,
                 expert_scale,
                 gamma,
                 balance_alpha,
                 dsa_topk,
                 local_window,
                 n_indexer_heads,
                 d_indexer_head,
                 rms_eps,
                 rope_base,
                 rope_type='rope',
                 mtp_depth=1,
                 pope_delta_init='zero',
                 original_seq_len=None,
                 rope_factor=1.0,
                 beta_fast=32,
                 beta_slow=1,
                 mscale=1.0,
                 indexer_use_fp8=True,
                 indexer_use_hadamard=True):
        super().__init__()
        self.n_layers = n_layers
        self.mtp_depth = mtp_depth

        # init transformer blocks
        self.transformer_layers = nn.ModuleList([
            Transformer(
                d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_v, d_rope,
                n_routed, n_shared, top_k, expert_scale, gamma, balance_alpha, dsa_topk, local_window,
                n_indexer_heads, d_indexer_head, rms_eps, rope_base, rope_type, pope_delta_init,
                original_seq_len, rope_factor, beta_fast, beta_slow, mscale,
                indexer_use_fp8, indexer_use_hadamard
            )
            for _ in range(self.n_layers)
        ])

        # init embedding, norm, and lm_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model, eps=rms_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # tie lm_head and embedding weights together (save memory/compute and improve generalization)
        self.lm_head.weight = self.embedding.weight

        self.mtp_modules = nn.ModuleList([
            MTPModule(
                d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_v, d_rope,
                n_routed, n_shared, top_k, expert_scale, gamma, balance_alpha, dsa_topk, local_window,
                n_indexer_heads, d_indexer_head, rms_eps, rope_base, rope_type, pope_delta_init,
                original_seq_len, rope_factor, beta_fast, beta_slow, mscale,
                indexer_use_fp8, indexer_use_hadamard
            )
            for _ in range(self.mtp_depth)
        ])

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

    def forward(self, input_ids, dsa_warmup=False, compute_aux=False):
        # embed once, reuse for MTP (avoid redundant embedding lookup)
        emb = self.embedding(input_ids)
        x = emb
        dsa_kl = None
        moe_balance = None
        for block in self.transformer_layers:
            x, attn_aux, moe_aux = block(x, dsa_warmup=dsa_warmup, compute_aux=compute_aux)
            if compute_aux:
                if attn_aux is not None:
                    dsa_kl = attn_aux if dsa_kl is None else dsa_kl + attn_aux
                if moe_aux is not None:
                    moe_balance = moe_aux if moe_balance is None else moe_balance + moe_aux

        main_logits = self.lm_head(self.norm(x))

        mtp_logits = []
        if self.mtp_modules:
            h_prev = x
            for depth, mtp in enumerate(self.mtp_modules, start=1):
                h_prev = h_prev[:, :-1]
                emb_next = emb[:, depth:]
                h_prev, attn_aux, moe_aux = mtp(h_prev, emb_next, dsa_warmup=dsa_warmup, compute_aux=compute_aux)
                if compute_aux:
                    if attn_aux is not None:
                        dsa_kl = attn_aux if dsa_kl is None else dsa_kl + attn_aux
                    if moe_aux is not None:
                        moe_balance = moe_aux if moe_balance is None else moe_balance + moe_aux
                mtp_logits.append(self.lm_head(self.norm(h_prev)))

        if compute_aux:
            return main_logits, mtp_logits, dsa_kl, moe_balance
        return main_logits, mtp_logits
    
if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()
    cfg.n_layers = 4  # reduced for testing
    model = Cinnamon(
        cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.hidden_dim, cfg.n_heads,
        cfg.max_seq_len, cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_v, cfg.d_rope,
        cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.expert_scale, cfg.gamma,
        0.0, cfg.dsa_topk, cfg.local_window, cfg.n_indexer_heads, cfg.d_indexer_head,
        cfg.rms_eps, cfg.rope_base, cfg.rope_type, cfg.mtp_depth, cfg.pope_delta_init,
        cfg.original_seq_len, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow, cfg.mscale,
        cfg.indexer_use_fp8, cfg.indexer_use_hadamard
    )
    x = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))
    out, mtp_out = model(x)
    mtp_shapes = [t.shape for t in mtp_out]
    print(f"output shape: {out.shape}, mtp: {mtp_shapes}")
    print(f"Total: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
