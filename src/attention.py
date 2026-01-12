import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import RMSNorm
from kernels import sparse_attention


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in DeepSeek V3.

    Standard RoPE rotates pairs of dimensions by position-dependent angles:
    - For pair (x_0, x_1) at position t with frequency θ:
      output = (x_0·cos(tθ) - x_1·sin(tθ), x_0·sin(tθ) + x_1·cos(tθ))

    This is decoupled from content in MLA - applied only to separate
    d_rope-dimensional key/query components, not the full representations.
    """
    def __init__(self, d_k, max_seq_len, base, original_seq_len=None, rope_factor=1.0,
                 beta_fast=32, beta_slow=1):
        super().__init__()
        self.d_k = d_k
        self.n_freq = d_k // 2
        self.base = base
        self.max_seq_len = max_seq_len
        self.original_seq_len = original_seq_len if original_seq_len is not None else max_seq_len
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Frequencies: θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        # This gives decreasing frequencies (slower rotation for higher dims)
        freqs = base ** (-(torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        if max_seq_len > self.original_seq_len and self.rope_factor != 1.0:
            low, high = self._find_correction_range(
                self.beta_fast, self.beta_slow, d_k, base, self.original_seq_len
            )
            smooth = 1 - self._linear_ramp_factor(low, high, d_k // 2)
            freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth
        self.register_buffer('freqs', freqs)

        # Pre-compute angles for positions 0..max_seq_len-1
        # Shape: (max_seq_len, n_freq)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    @staticmethod
    def _find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    @classmethod
    def _find_correction_range(cls, low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(cls._find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(cls._find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear, 0, 1)

    def _extend_cache(self, seq_len):
        """Extend cache for sequences longer than max_seq_len."""
        if seq_len <= self.cos_cached.size(0):
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        pos = torch.arange(seq_len, device=self.freqs.device, dtype=self.freqs.dtype)
        angles = torch.outer(pos, self.freqs)
        return angles.cos(), angles.sin()

    def forward(self, x, positions=None, interleaved=True):
        """
        Apply rotary embedding to input tensor.

        Args:
            x: (batch, seq, heads, d_k) or (batch, seq, k, d_k) for sparse keys
            positions: optional position indices, shape (seq,), (batch, seq), or (batch, seq, k)
            interleaved: if False, treat last dim as [real...imag...] instead of interleaved pairs

        Returns:
            Rotated tensor of same shape as input
        """
        seq_len = x.size(1)

        if positions is None:
            # Standard sequential positions
            cos, sin = self._extend_cache(seq_len)
            # Reshape for broadcasting: (seq, n_freq) -> (1, seq, 1, n_freq)
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            # Custom positions (for sparse attention gathered keys)
            pos = positions
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            if pos.dim() == 2:
                pos = pos.unsqueeze(-1)
            # Add trailing dimension for frequency broadcasting
            pos = pos.unsqueeze(-1)  # (..., 1)
            pos = pos.to(device=self.freqs.device, dtype=self.freqs.dtype)
            # pos shape: (batch, seq, k, 1) or (batch, seq, 1, 1)
            angles = pos * self.freqs  # broadcast to (..., n_freq)
            cos = angles.cos()
            sin = angles.sin()

        # Split into real/imag pairs and rotate
        # x shape: (..., d_k) where d_k = 2 * n_freq
        if interleaved:
            x1, x2 = x[..., ::2], x[..., 1::2]  # Even and odd indices
        else:
            half = x.size(-1) // 2
            x1, x2 = x[..., :half], x[..., half:]

        # Rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        if interleaved:
            # Interleave back: [out1_0, out2_0, out1_1, out2_1, ...]
            out = torch.stack([out1, out2], dim=-1).flatten(-2)
        else:
            # Concatenate real/imag blocks: [out_real..., out_imag...]
            out = torch.cat([out1, out2], dim=-1)
        return out


class PoPE(nn.Module):
    """
    Polar Positional Encoding (PoPE) with learnable phase offset.

    Uses d frequencies and outputs 2d dimensions (cos/sin):
    - Query at pos t: [μ·cos(tθ), μ·sin(tθ)]
    - Key at pos s:   [μ·cos(sθ + δ), μ·sin(sθ + δ)]
    - q·k = Σ μ_q·μ_k·cos((s-t)θ + δ)
    """
    def __init__(self, d_k, max_seq_len, base, delta_init="zero"):
        super().__init__()
        self.d_k = d_k
        self.n_freq = d_k
        self.base = base
        self.register_buffer('theta', base ** (-(torch.arange(self.n_freq, dtype=torch.float32) / self.n_freq)))

        # Cache base angles (without delta) - shape: (max_seq_len, n_freq)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer('base_angles', torch.outer(pos, self.theta))

        # Learnable phase offset δ ∈ [-2π, 0]
        self.delta = nn.Parameter(torch.zeros(self.n_freq))
        if delta_init == "uniform":
            self.delta.data.uniform_(-2 * math.pi, 0.0)

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
            delta = self.delta.clamp(-2 * math.pi, 0.0)
            phases = phases + delta

        # Magnitude per component (softplus), then polar -> Cartesian
        mu = F.softplus(x)
        cos_out = mu * phases.cos()
        sin_out = mu * phases.sin()
        return torch.cat((cos_out, sin_out), dim=-1)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dimension."""
    n = x.size(-1)
    if n & (n - 1) != 0:
        return x
    h = x
    prefix = x.shape[:-1]
    step = 1
    while step < n:
        h = h.reshape(*prefix, -1, 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.cat([a + b, a - b], dim=-1)
        h = h.reshape(*prefix, n)
        step *= 2
    return h * (n ** -0.5)


class DSAIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) lightning indexer.

    Computes index scores:
        I_{t,s} = sum_j w_{t,j} * ReLU(q^I_{t,j} · k^I_s)
    """
    def __init__(self, d_model, d_cq, n_heads, d_head, d_rope, max_seq_len, rope_base, rope_type,
                 topk, use_fp8=True, use_hadamard=True, original_seq_len=None, rope_factor=1.0,
                 beta_fast=32, beta_slow=1, pope_delta_init='zero'):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_rope = d_rope
        self.topk = topk
        self.use_fp8 = use_fp8
        self.use_hadamard = use_hadamard
        self.softmax_scale = d_head ** -0.5
        self.rope_type = rope_type

        self.w_q = nn.Linear(d_cq, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d_model, d_head, bias=False)
        self.k_norm = nn.LayerNorm(d_head)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

        # Indexer positional encoding - use min(d_rope, d_head) to match indexer dimensions
        # The indexer has smaller heads than main attention, so we can't use the full d_rope
        indexer_rope_dim = min(d_rope, d_head)
        self.indexer_rope_dim = indexer_rope_dim

        if rope_type == 'pope':
            self.pos_enc = PoPE(indexer_rope_dim, max_seq_len, rope_base, delta_init=pope_delta_init)
            self.pos_enc_dim = indexer_rope_dim * 2  # PoPE outputs cos + sin
        elif rope_type == 'none':
            self.pos_enc = None  # DroPE: no positional embeddings
            self.pos_enc_dim = 0
        else:  # rope
            self.pos_enc = RoPE(
                indexer_rope_dim,
                max_seq_len=max_seq_len,
                base=rope_base,
                original_seq_len=original_seq_len,
                rope_factor=rope_factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
            self.pos_enc_dim = indexer_rope_dim  # RoPE preserves dimension

        try:
            from kernels import quantize_fp8_rowwise
            self._quantize_fp8_rowwise = quantize_fp8_rowwise
        except Exception:
            self._quantize_fp8_rowwise = None

    def _apply_pos_enc(self, x, positions=None, apply_delta=False):
        """Apply positional encoding (RoPE or PoPE) to indexer queries/keys."""
        if self.rope_type == 'pope':
            return self.pos_enc(x, apply_delta=apply_delta, positions=positions)
        else:
            return self.pos_enc(x, positions=positions, interleaved=False)

    def forward(self, x, q_latent, positions=None, mask=None, return_scores=False):
        bsz, seqlen, _ = x.shape
        q = self.w_q(q_latent).view(bsz, seqlen, self.n_heads, self.d_head)
        k = self.k_norm(self.w_k(x))

        # Apply positional encoding to indexer queries/keys (skip for DroPE)
        if self.pos_enc is not None:
            rope_dim = self.indexer_rope_dim
            if rope_dim > 0:
                q_rope, q_nope = q[..., :rope_dim], q[..., rope_dim:]
                k_rope, k_nope = k[..., :rope_dim], k[..., rope_dim:]
                # For PoPE: queries don't get delta, keys do
                q_rope = self._apply_pos_enc(q_rope, positions=None, apply_delta=False)
                k_rope = self._apply_pos_enc(k_rope.unsqueeze(2), positions=None, apply_delta=True).squeeze(2)
                q = torch.cat([q_rope, q_nope], dim=-1)
                k = torch.cat([k_rope, k_nope], dim=-1)

        if self.use_hadamard:
            q = hadamard_transform(q)
            k = hadamard_transform(k)

        weights = self.weights_proj(x.float()) * (self.n_heads ** -0.5)

        if self.use_fp8 and x.is_cuda and self._quantize_fp8_rowwise is not None:
            # Use actual dimension after positional encoding (PoPE expands d_rope -> 2*d_rope)
            actual_dim = q.size(-1)
            q_flat = q.reshape(-1, actual_dim).contiguous()
            k_flat = k.reshape(-1, actual_dim).contiguous()
            q_fp8, q_scale = self._quantize_fp8_rowwise(q_flat, block_size=128)
            k_fp8, k_scale = self._quantize_fp8_rowwise(k_flat, block_size=128)
            q_fp8 = q_fp8.view(bsz, seqlen, self.n_heads, actual_dim)
            k_fp8 = k_fp8.view(bsz, seqlen, actual_dim)
            q_scale = q_scale.view(bsz, seqlen, self.n_heads, -1)[..., 0]
            k_scale = k_scale.view(bsz, seqlen, -1)[..., 0]

            logits = torch.einsum('bthd,bsd->bths', q_fp8.float(), k_fp8.float())
            logits = F.relu(logits) * (q_scale * weights * self.softmax_scale).unsqueeze(-1)
            scores = logits.sum(dim=2) * k_scale.unsqueeze(1)
        else:
            logits = F.relu(torch.einsum('bthd,bsd->bths', q, k))
            scores = torch.einsum('bths,bth->bts', logits, weights * self.softmax_scale)

        if mask is not None:
            scores = scores + mask

        if return_scores:
            return scores

        topk = seqlen if self.topk <= 0 else min(seqlen, max(1, self.topk))
        return scores.topk(topk, dim=-1).indices

class MultiheadLatentAttention(nn.Module):
    def __init__(self, d_model, d_ckv, d_cq, n_head, d_head, d_v, d_rope, max_seq_len,
                 dsa_topk, local_window, n_indexer_heads, d_indexer_head, rms_eps,
                 rope_base, rope_type='rope', pope_delta_init='zero',
                 original_seq_len=None, rope_factor=1.0, beta_fast=32, beta_slow=1, mscale=1.0,
                 indexer_use_fp8=True, indexer_use_hadamard=True, use_sparse_kernel=True):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.d_v = d_v
        self.d_rope = d_rope
        self.rope_dim = 0 if rope_type == 'none' else d_rope * (2 if rope_type == 'pope' else 1)
        self.rope_type = rope_type
        self.dsa_topk = dsa_topk
        self.local_window = local_window
        self.use_sparse_kernel = use_sparse_kernel
        self.original_seq_len = original_seq_len if original_seq_len is not None else max_seq_len
        self.rope_factor = rope_factor

        # Compressed KV and Q projections
        self.w_dkv = nn.Linear(d_model, d_ckv, bias=False)
        self.kv_norm = RMSNorm(d_ckv, eps=rms_eps)
        self.w_uk = nn.Linear(d_ckv, d_head * n_head, bias=False)
        self.w_uv = nn.Linear(d_ckv, d_v * n_head, bias=False)
        self.w_dq = nn.Linear(d_model, d_cq, bias=False)
        self.q_norm = RMSNorm(d_cq, eps=rms_eps)
        self.w_uq = nn.Linear(d_cq, d_head * n_head, bias=False)

        # Decoupled positional projections (with normalization for PoPE stability)
        # These are None when rope_type='none' (DroPE - no positional embeddings)
        if rope_type != 'none':
            self.w_qr = nn.Linear(d_cq, d_rope * n_head, bias=False)
            self.w_kr = nn.Linear(d_model, d_rope, bias=False)
            self.qr_norm = RMSNorm(d_rope * n_head, eps=rms_eps)
            self.kr_norm = RMSNorm(d_rope, eps=rms_eps)
        else:
            self.w_qr = None
            self.w_kr = None
            self.qr_norm = None
            self.kr_norm = None

        # Positional encoding: RoPE, PoPE, or None (DroPE)
        if rope_type == 'rope':
            self.pos_enc = RoPE(
                d_rope,
                max_seq_len,
                rope_base,
                original_seq_len=self.original_seq_len,
                rope_factor=rope_factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
        elif rope_type == 'pope':
            self.pos_enc = PoPE(d_rope, max_seq_len, rope_base, delta_init=pope_delta_init)
        elif rope_type == 'none':
            self.pos_enc = None  # DroPE: no positional embeddings
        else:
            raise ValueError(f"Unknown rope_type: {rope_type}. Use 'rope', 'pope', or 'none'.")

        self.w_out = nn.Linear(d_v * n_head, d_model, bias=False)
        self.indexer = DSAIndexer(
            d_model, d_cq, n_indexer_heads, d_indexer_head, d_rope, max_seq_len,
            rope_base, rope_type, dsa_topk, use_fp8=indexer_use_fp8,
            use_hadamard=indexer_use_hadamard, original_seq_len=self.original_seq_len,
            rope_factor=rope_factor, beta_fast=beta_fast, beta_slow=beta_slow,
            pope_delta_init=pope_delta_init
        )
        self.softmax_scale = (self.d_head + self.rope_dim) ** -0.5
        if rope_type == 'rope' and max_seq_len > self.original_seq_len and rope_factor != 1.0:
            scale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.softmax_scale *= scale * scale

    def _create_sparse_mask(self, idx, seq_len, device):
        """
        Create sparse attention mask from top-k indices, combined with causal mask.

        Args:
            idx: [B, L, K] - selected key indices for each query position
            seq_len: sequence length L
            device: torch device

        Returns:
            mask: [B, 1, L, L] - True where attention is allowed (sparse + causal)
        """
        B, L, K = idx.shape
        # Create [B, L, L] mask initialized to False
        mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        # Scatter True at selected positions for each query
        mask.scatter_(2, idx, True)
        # Combine with causal mask (can only attend to positions <= current)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        mask = mask & causal
        return mask.unsqueeze(1)  # [B, 1, L, L] for head broadcasting

    def forward(self, x, dsa_warmup=False, compute_aux=False):
        batch_size, seq_len, _ = x.size()
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1
        ).unsqueeze(0)

        c_kv = self.kv_norm(self.w_dkv(x))
        c_q = self.q_norm(self.w_dq(x))

        index_scores = self.indexer(
            x.detach(),
            c_q.detach(),
            mask=causal_mask,
            return_scores=True
        )

        idx = None
        if not dsa_warmup:
            topk = seq_len if self.dsa_topk <= 0 else min(self.dsa_topk, seq_len)
            if self.local_window > 0:
                topk = min(seq_len, max(topk, self.local_window))
                offsets = torch.arange(seq_len, device=x.device)
                local_mask = (offsets[None, :] - offsets[:, None])
                local_mask = (local_mask <= 0) & (local_mask > -self.local_window)
                index_scores = index_scores + local_mask.unsqueeze(0).to(index_scores.dtype) * 1e4
            idx = index_scores.topk(topk, dim=-1).indices

        # Query projections (same for both paths)
        query_c = self.w_uq(c_q).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # Key/value projections - same for both dense warmup and sparse (memory-efficient!)
        # Keys/values are [B, H, L, D] - no expansion to [B, H, L, K, D]
        key_c = self.w_uk(c_kv).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        value = self.w_uv(c_kv).view(batch_size, seq_len, self.n_head, self.d_v).transpose(1, 2)

        # Positional components (skip for rope_type='none' / DroPE)
        if self.rope_type == 'none':
            # DroPE: content-only attention, no positional embeddings
            query = query_c
            key = key_c
        else:
            q_rope_input = self.qr_norm(self.w_qr(c_q)).view(batch_size, seq_len, self.n_head, self.d_rope)
            k_rope_input = self.kr_norm(self.w_kr(x)).unsqueeze(2)

            if self.rope_type == 'rope':
                query_r = self.pos_enc(q_rope_input, interleaved=True).transpose(1, 2)
                key_r = self.pos_enc(k_rope_input, interleaved=True).squeeze(2)
            else:  # pope
                query_r = self.pos_enc(q_rope_input, apply_delta=False).transpose(1, 2)
                key_r = self.pos_enc(k_rope_input, apply_delta=True).squeeze(2)
            key_r = key_r.unsqueeze(1).expand(-1, self.n_head, -1, -1)

            query = torch.cat((query_c, query_r), dim=-1)
            key = torch.cat((key_c, key_r), dim=-1)

        scale = self.softmax_scale

        if dsa_warmup:
            # Dense attention (warmup phase) - O(L^2)
            scores = torch.einsum('bhld,bhsd->bhls', query, key) * scale
            scores = scores + causal_mask.unsqueeze(1)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
            attn_o = torch.einsum('bhls,bhsd->bhld', attn_weights, value)
            sparse_attn_weights = None
        elif self.use_sparse_kernel:
            # TRUE sparse attention - O(L*K) compute and memory
            attn_o, sparse_attn_weights = sparse_attention(query, key, value, idx, scale)
        else:
            # Fallback: dense + mask (O(L^2) compute, for testing/comparison)
            scores = torch.einsum('bhld,bhsd->bhls', query, key) * scale
            sparse_mask = self._create_sparse_mask(idx, seq_len, x.device)
            scores = scores.masked_fill(~sparse_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
            attn_o = torch.einsum('bhls,bhsd->bhld', attn_weights, value)
            sparse_attn_weights = None

        attn_o = attn_o.transpose(1, 2).reshape(batch_size, seq_len, self.n_head * self.d_v)
        out = self.w_out(attn_o)

        aux_loss = None
        if compute_aux:
            if dsa_warmup:
                # Dense: attention weights are [B, H, L, L]
                p = attn_weights.sum(dim=1)  # [B, L, L]
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
                log_q = F.log_softmax(index_scores, dim=-1)
                log_q = log_q.masked_fill(~torch.isfinite(index_scores), 0.0)
            elif self.use_sparse_kernel:
                # TRUE sparse: sparse_attn_weights is already [B, H, L, K]
                p = sparse_attn_weights.float().sum(dim=1)  # [B, L, K]
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
                log_q = F.log_softmax(index_scores.gather(-1, idx), dim=-1)
                positions = torch.arange(seq_len, device=x.device)
                valid = idx <= positions[None, :, None]
                log_q = log_q.masked_fill(~valid, 0.0)
            else:
                # Fallback sparse: gather attention weights at selected positions
                # attn_weights is [B, H, L, L], gather to [B, H, L, K] using idx
                idx_expanded = idx.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # [B, H, L, K]
                p = attn_weights.gather(-1, idx_expanded)  # [B, H, L, K]
                p = p.sum(dim=1)  # [B, L, K]
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
                log_q = F.log_softmax(index_scores.gather(-1, idx), dim=-1)
                positions = torch.arange(seq_len, device=x.device)
                valid = idx <= positions[None, :, None]
                log_q = log_q.masked_fill(~valid, 0.0)
            p = p.detach()
            aux_loss = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1).mean()

        return out, aux_loss
