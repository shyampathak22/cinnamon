import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class PoPE(nn.Module):
    """
    Polar Positional Encoding (PoPE) with learnable phase offset.

    Uses d/2 frequencies to preserve dot-product compatibility:
    - Query at pos t: [μ·cos(tθ), μ·sin(tθ)]
    - Key at pos s:   [μ·cos(sθ + δ), μ·sin(sθ + δ)]
    - q·k = Σ μ_q·μ_k·cos((s-t)θ + δ)
    """
    def __init__(self, d_k, max_seq_len, base):
        super().__init__()
        self.d_k = d_k
        self.n_freq = d_k // 2
        self.base = base
        self.register_buffer('theta', base ** (-(torch.arange(self.n_freq, dtype=torch.float32) / self.n_freq)))

        # Cache base angles (without delta) - shape: (max_seq_len, n_freq)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer('base_angles', torch.outer(pos, self.theta))

        # Learnable phase offset δ ∈ [-2π, 0]
        self.raw_delta = nn.Parameter(torch.zeros(self.n_freq))

    def _extend_base_angles(self, seq_len):
        """Extend cache for longer sequences."""
        if seq_len <= self.base_angles.size(0):
            return self.base_angles[:seq_len]
        pos = torch.arange(seq_len, device=self.theta.device, dtype=self.theta.dtype)
        return torch.outer(pos, self.theta)

    def forward(self, x, apply_delta=True, positions=None):
        """
        Args:
            x: (batch, seq, heads, d_k) or (batch, seq, k, d_k)
            apply_delta: if True, add learnable phase offset (use for keys only)
            positions: optional positions tensor with shape (seq,), (batch, seq), or (batch, seq, k)
        """
        seq_len = x.size(1)

        if positions is None:
            phases = self._extend_base_angles(seq_len).unsqueeze(0).unsqueeze(2)
        else:
            pos = positions
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            if pos.dim() == 2:
                pos = pos.unsqueeze(-1)
            pos = pos.to(device=self.theta.device, dtype=self.theta.dtype)
            phases = pos[..., None] * self.theta

        if apply_delta:
            delta = -2 * torch.pi * torch.sigmoid(self.raw_delta)
            phases = phases + delta

        # Single magnitude per frequency (shared across cos/sin)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        mu = F.softplus(x1 + x2)
        cos_out = mu * phases.cos()
        sin_out = mu * phases.sin()
        return torch.cat((cos_out, sin_out), dim=-1)


class LightningIndexer(nn.Module):
    """
    Ultra-light scorer for sparse attention token selection (DeepSeek V3.2 style).

    Uses a fused Triton kernel that:
    - Avoids materializing O(L² * n_head) intermediate tensor
    - Computes in FP16 for memory efficiency
    - Fuses ReLU + weighted sum into single pass

    Memory: O(L²) in FP16 output only, vs O(L² * n_head) in FP32 intermediate
    """
    def __init__(self, d_model, n_head, d_head, n_indexer_heads):
        super().__init__()
        # Use fewer heads for indexer (DeepSeek style)
        self.n_head = n_indexer_heads
        self.d_head = d_head
        self.w_q = nn.Linear(d_model, n_indexer_heads * d_head, bias=False)
        self.w_k = nn.Linear(d_model, d_head, bias=False)
        self.w = nn.Parameter(torch.ones(n_indexer_heads))

        # Try to import fused kernel
        self._use_fused = False
        self._warned_fallback = False
        try:
            from kernels import lightning_indexer_fused
            self._fused_fn = lightning_indexer_fused
            self._use_fused = True
        except ImportError:
            pass

    def forward(self, h):
        batch, seq, _ = h.shape

        q = self.w_q(h).view(batch, seq, self.n_head, self.d_head)
        k = self.w_k(h)

        if self._use_fused and h.is_cuda:
            # Use fused Triton kernel (avoids O(L² * n_head) intermediate)
            I = self._fused_fn(q, k, self.w)
        else:
            # Fallback: standard PyTorch (for CPU or if kernel unavailable)
            if h.is_cuda and not self._use_fused and not self._warned_fallback:
                warnings.warn(
                    "LightningIndexer: fused kernel unavailable, using O(L² × n_head) fallback. "
                    "This may cause OOM on long sequences.",
                    RuntimeWarning
                )
                self._warned_fallback = True
            scores = F.relu(torch.einsum('bthd,bsd->bths', q, k))
            I = torch.einsum('bths,h->bts', scores, self.w)

        return I
    
class TokenSelector(nn.Module):
    def __init__(self, k, local_window):
        super().__init__()
        self.k = k
        self.lw = local_window

    def forward(self, I):
        _, q_len, k_len = I.shape
        device = I.device

        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)

        local_mask = (k_pos >= q_pos - self.lw + 1) & (k_pos <= q_pos)
        causal_mask = k_pos > q_pos

        # Non-in-place masking to preserve gradients
        I = I.masked_fill(local_mask, float('inf'))
        I = I.masked_fill(causal_mask, float('-inf'))

        k = min(self.k, k_len, q_len)
        _, indices = I.topk(k, dim=-1)
        return indices

class MultiheadLatentAttention(nn.Module):
    def __init__(self, d_model, d_ckv, d_cq, n_head, d_head, d_rope, max_seq_len, k_ts, local_window, n_indexer_heads, rope_base):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.w_dkv = nn.Linear(d_model, d_ckv, bias=False)
        self.w_uk = nn.Linear(d_ckv, d_head*n_head, bias=False)
        self.w_uv = nn.Linear(d_ckv, d_head*n_head, bias=False)
        self.w_dq = nn.Linear(d_model, d_cq, bias=False)
        self.w_uq = nn.Linear(d_cq, d_head*n_head, bias=False)
        self.d_rope = d_rope
        self.w_qr = nn.Linear(d_cq, d_rope*n_head, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)
        self.pope = PoPE(d_rope, max_seq_len, rope_base)
        self.w_out = nn.Linear(d_head*n_head, d_model, bias=False)
        self.light_sel = LightningIndexer(d_model, n_head, d_head, n_indexer_heads)
        self.tok_sel = TokenSelector(k_ts, local_window)

    def sparse_attention(self, query, key, value, scale):
        scores = torch.einsum('bhld,bhlkd->bhlk', query, key) * scale
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        # Cast back to value dtype for einsum compatibility
        attn_weights = attn_weights.to(value.dtype)
        output = torch.einsum('bhlk,bhlkd->bhld', attn_weights, value)
        return output

    def forward(self, x):
        batch_size = x.size(0)
        c_kv = self.w_dkv(x)
        I = self.light_sel(x)
        idx = self.tok_sel(I)

        # Single gather for both c_kv and x (same indices, halves memory bandwidth)
        batch_idx = torch.arange(batch_size, device=x.device)[:, None, None]
        combined = torch.cat([c_kv, x], dim=-1)  # (batch, L, d_ckv + d_model)
        combined_selected = combined[batch_idx, idx]  # (batch, L, k, d_ckv + d_model)
        ckv_selected = combined_selected[..., :c_kv.size(-1)]  # (batch, L, k, d_ckv)
        x_selected = combined_selected[..., c_kv.size(-1):]    # (batch, L, k, d_model)

        key_c = self.w_uk(ckv_selected).view(batch_size, ckv_selected.size(1), ckv_selected.size(2), self.n_head, self.d_head).permute(0, 3, 1, 2, 4)
        value = self.w_uv(ckv_selected).view(batch_size, ckv_selected.size(1), ckv_selected.size(2), self.n_head, self.d_head).permute(0, 3, 1, 2, 4)
        c_q = self.w_dq(x)
        query_c = self.w_uq(c_q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        query_r = self.pope(
            self.w_qr(c_q).view(batch_size, -1, self.n_head, self.d_rope),
            apply_delta=False
        ).transpose(1, 2)
        ker_r_pre = self.w_kr(x_selected)
        key_r = self.pope(ker_r_pre, apply_delta=True, positions=idx)
        key_r = key_r.unsqueeze(1).expand(-1, self.n_head, -1, -1, -1)
        query = torch.cat((query_c, query_r), dim=-1)
        key = torch.cat((key_c, key_r), dim=-1)

        attn_o = self.sparse_attention(query, key, value, scale=(self.d_head + self.d_rope)**-0.5)
        attn_o = attn_o.transpose(1, 2).reshape(batch_size, -1, self.d_head*self.n_head)
        return self.w_out(attn_o)
    
if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()

    # Test MLA
    mla = MultiheadLatentAttention(
        cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head,
        cfg.d_rope, cfg.max_seq_len, cfg.k_ts, cfg.local_window,
        cfg.n_indexer_heads, cfg.rope_base
    )
    x = torch.randn(4, 128, cfg.d_model)
    out = mla(x)
    print(f"MLA output shape: {out.shape}")

    # Test PoPE extrapolation
    pope = PoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
    pope.cuda()
    for seq_len in [1024, 4096, 16384]:
        x = torch.randn(4, seq_len, 1, cfg.d_rope).cuda()
        out = pope(x)
        print(f"PoPE seq_len={seq_len}: {out.shape}")

    # Test LightningIndexer
    indexer = LightningIndexer(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_indexer_heads)
    h = torch.randn(4, 128, cfg.d_model)
    relevance = indexer(h)
    print(f"LightningIndexer relevance shape: {relevance.shape}")

    # Test TokenSelector
    selector = TokenSelector(k=32, local_window=8)
    I = torch.randn(4, 64, 64)
    indices = selector(I)
    print(f"TokenSelector indices shape: {indices.shape}")

    # Benchmark MLA at different sequence lengths
    import time
    torch.cuda.empty_cache()
    mla_sparse = MultiheadLatentAttention(
        cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head,
        cfg.d_rope, 16384, cfg.k_ts, cfg.local_window,
        cfg.n_indexer_heads, cfg.rope_base
    )
    mla_sparse.cuda()

    for seq_len in [1024, 2048, 4096]:
        torch.cuda.empty_cache()
        x = torch.randn(1, seq_len, cfg.d_model).cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = mla_sparse(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"seq_len={seq_len}: {elapsed*1000:.1f}ms, shape={out.shape}")
        del x, out
