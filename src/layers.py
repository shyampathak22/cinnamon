import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import RMSNorm
from attention import MultiheadLatentAttention

# Try to import grouped GEMM kernel
try:
    from kernels import moe_grouped_gemm
    HAS_GROUPED_GEMM = True
except ImportError:
    HAS_GROUPED_GEMM = False


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


class GroupedExperts(nn.Module):
    """
    Stacked expert weights for grouped GEMM execution.

    Optimization: w1 and w2 are fused into w12 (2*hidden_dim output).
    This reduces 3 GEMMs to 2 GEMMs per forward pass.
    """
    def __init__(self, n_experts, d_model, hidden_dim):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Fused up-projection: w12 = [w1; w2] stacked
        # Single GEMM outputs (total_tokens, 2*hidden_dim), then split
        self.w12 = nn.Parameter(torch.empty(n_experts, 2 * hidden_dim, d_model))
        self.w3 = nn.Parameter(torch.empty(n_experts, d_model, hidden_dim))

        # Initialize
        nn.init.normal_(self.w12, mean=0.0, std=0.02)
        nn.init.normal_(self.w3, mean=0.0, std=0.02)

    def forward(self, sorted_x, expert_starts, expert_ends, max_expert_tokens=None):
        """
        Apply SwiGLU to sorted tokens using grouped GEMM.

        Args:
            sorted_x: (total_tokens, d_model) - tokens sorted by expert
            expert_starts: (n_experts,) - start indices
            expert_ends: (n_experts,) - end indices
            max_expert_tokens: max tokens assigned to any single expert (for grid sizing)

        Returns:
            (total_tokens, d_model) - output
        """
        if HAS_GROUPED_GEMM and sorted_x.is_cuda:
            # Fused up-projection: one GEMM instead of two
            h12 = moe_grouped_gemm(sorted_x, self.w12, expert_starts, expert_ends, max_expert_tokens)
            h1, h2 = h12.float().chunk(2, dim=-1)
            h = F.silu(h1) * h2
            out = moe_grouped_gemm(h.to(sorted_x.dtype), self.w3, expert_starts, expert_ends, max_expert_tokens)
            return out.float()
        else:
            # Fallback: sequential expert processing (GPU without kernel or CPU)
            out = torch.zeros(sorted_x.size(0), self.d_model, device=sorted_x.device, dtype=sorted_x.dtype)
            for i in range(self.n_experts):
                start, end = expert_starts[i].item(), expert_ends[i].item()
                if end > start:
                    x_i = sorted_x[start:end]
                    h12 = F.linear(x_i, self.w12[i])
                    h1, h2 = h12.chunk(2, dim=-1)
                    h = F.silu(h1) * h2
                    out[start:end] = F.linear(h, self.w3[i])
            return out
    
class MoE(nn.Module):
    """
    Mixture of Experts with grouped GEMM for parallel expert execution.

    Uses GroupedExperts for routed experts (all experts in one kernel launch)
    and standard SwiGLU for shared experts.
    """
    def __init__(self, n_routed, n_shared, top_k, d_model, hidden_dim, expert_scale, gamma, balance_alpha=0.0):
        super().__init__()
        routed_hidden = hidden_dim // expert_scale

        # Grouped experts for parallel execution
        self.routed_experts = GroupedExperts(n_routed, d_model, routed_hidden)
        self.shared_experts = nn.ModuleList([SwiGLU(d_model, hidden_dim) for _ in range(n_shared)])
        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.top_k = top_k
        self.n_routed = n_routed
        self.d_model = d_model
        self.gamma = gamma
        self.balance_alpha = balance_alpha
        self.register_buffer("expert_bias", torch.zeros(n_routed))

    def _update_bias(self, expert_counts: torch.Tensor, total_selections: int):
        """Update expert bias using precomputed counts (avoids duplicate bincount)."""
        expected = total_selections / self.n_routed
        delta = torch.where(expert_counts.float() > expected, -self.gamma, self.gamma)
        self.expert_bias += delta
        self.expert_bias.clamp_(-1, 1)

    def forward(self, x):
        batch, seq, d = x.shape
        # Fast path for single shared expert (common case)
        shared_out = self.shared_experts[0](x) if len(self.shared_experts) == 1 else sum(expert(x) for expert in self.shared_experts)
        scores = torch.sigmoid(self.router(x))
        selection_scores = scores + self.expert_bias
        _, top_idx = selection_scores.topk(self.top_k, dim=-1)
        top_scores = scores.gather(-1, top_idx)
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Efficient MoE: permute-process-unpermute with grouped GEMM
        flat_x = x.view(-1, d)
        n_tokens = flat_x.size(0)

        # Flatten indices and scores
        expanded_idx = top_idx.view(-1)  # (n_tokens * top_k,)
        expanded_scores = top_scores.view(-1)

        # Sort by expert for grouped processing
        sorted_idx, sort_order = expanded_idx.sort()

        # Memory-efficient gather: map sorted position -> original token index
        # sort_order[i] is the original expanded position, divide by top_k to get token
        token_indices = sort_order // self.top_k
        sorted_x = flat_x[token_indices]  # Gather directly, no expanded_x allocation
        sorted_scores = expanded_scores[sort_order]

        # Expert boundaries (compute bincount once, reuse for bias update)
        expert_counts = torch.bincount(sorted_idx, minlength=self.n_routed)
        expert_ends = expert_counts.cumsum(0)
        expert_starts = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), expert_ends[:-1]])

        # Max tokens per expert for grid sizing (sync is worth it vs 2x over-launch)
        max_expert_tokens = expert_counts.max().item()

        # Grouped GEMM: all experts in parallel
        sorted_out = self.routed_experts(sorted_x, expert_starts, expert_ends, max_expert_tokens)

        # Apply gating and scatter-add directly to output (avoids expanded_out allocation)
        weighted_out = sorted_out * sorted_scores[:, None]
        routed_out = torch.zeros(n_tokens, d, device=x.device, dtype=weighted_out.dtype)
        routed_out.scatter_add_(0, token_indices.unsqueeze(1).expand_as(weighted_out), weighted_out)
        routed_out = routed_out.view(batch, seq, d)

        # Update bias using already-computed expert_counts
        if self.training:
            self._update_bias(expert_counts, expanded_idx.numel())

        balance_loss = None
        if self.balance_alpha > 0:
            # Sequence-wise balance loss to avoid per-sequence collapse
            flat_top_idx = top_idx.view(batch, -1)
            counts = torch.zeros(batch, self.n_routed, device=x.device, dtype=x.dtype)
            counts.scatter_add_(1, flat_top_idx, torch.ones_like(flat_top_idx, dtype=x.dtype))
            f_i = counts * (self.n_routed / (self.top_k * seq))

            s_norm = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
            p_i = s_norm.mean(dim=1)

            balance_loss = self.balance_alpha * (f_i * p_i).sum(dim=-1).mean()

        # Expert utilization stats (computed during training for monitoring)
        moe_stats = None
        if self.training:
            with torch.no_grad():
                # Normalize expert counts to fractions
                expert_fracs = expert_counts.float() / expert_counts.sum()
                # Router entropy: higher = more exploration
                router_probs = scores.mean(dim=(0, 1))  # avg routing probs per expert
                router_probs = router_probs / (router_probs.sum() + 1e-9)
                router_entropy = -(router_probs * (router_probs + 1e-9).log()).sum()

                moe_stats = {
                    "load_std": expert_fracs.std().item(),
                    "load_min": expert_fracs.min().item(),
                    "load_max": expert_fracs.max().item(),
                    "router_entropy": router_entropy.item(),
                    "expert_counts": expert_counts.detach(),  # for histogram
                }

        return shared_out + routed_out, balance_loss, moe_stats

class Transformer(nn.Module):

    def __init__(self, d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_v, d_rope,
                 n_routed, n_shared, top_k, expert_scale, gamma, balance_alpha, dsa_topk, local_window,
                 n_indexer_heads, d_indexer_head, rms_eps, rope_base, rope_type='rope',
                 pope_delta_init='zero', original_seq_len=None, rope_factor=1.0, beta_fast=32,
                 beta_slow=1, mscale=1.0, indexer_use_fp8=True, indexer_use_hadamard=True,
                 use_sparse_kernel=True):
        super().__init__()

        # init layers
        self.rms1 = RMSNorm(d_model, eps=rms_eps)
        self.rms2 = RMSNorm(d_model, eps=rms_eps)
        self.attn = MultiheadLatentAttention(
            d_model, d_ckv, d_cq, n_heads, d_head, d_v, d_rope, max_seq_len,
            dsa_topk, local_window, n_indexer_heads, d_indexer_head, rms_eps,
            rope_base, rope_type, pope_delta_init, original_seq_len, rope_factor, beta_fast,
            beta_slow, mscale, indexer_use_fp8, indexer_use_hadamard, use_sparse_kernel
        )
        self.moe = MoE(n_routed, n_shared, top_k, d_model, hidden_dim, expert_scale, gamma, balance_alpha)

    def forward(self, x, dsa_warmup=False, compute_aux=False):
        # compute norm of input
        normx = self.rms1(x)

        # pass norms to MHA and add to residual stream
        attn_out, attn_aux = self.attn(normx, dsa_warmup=dsa_warmup, compute_aux=compute_aux)
        x = x + attn_out

        # pass norms to SwiGLU and add to residual stream
        moe_out, moe_balance, moe_stats = self.moe(self.rms2(x))
        x = x + moe_out
        return x, attn_aux, moe_balance, moe_stats
    
class MTPModule(nn.Module):
    """
    DeepSeek-V3 style Multi-Token Prediction module.
    """
    def __init__(self, d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_v, d_rope,
                 n_routed, n_shared, top_k, expert_scale, gamma, balance_alpha, dsa_topk, local_window,
                 n_indexer_heads, d_indexer_head, rms_eps, rope_base, rope_type='rope',
                 pope_delta_init='zero', original_seq_len=None, rope_factor=1.0, beta_fast=32,
                 beta_slow=1, mscale=1.0, indexer_use_fp8=True, indexer_use_hadamard=True,
                 use_sparse_kernel=True):
        super().__init__()
        self.norm_h = RMSNorm(d_model, eps=rms_eps)
        self.norm_emb = RMSNorm(d_model, eps=rms_eps)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = Transformer(
            d_model, hidden_dim, max_seq_len, n_heads, d_ckv, d_cq, d_head, d_v, d_rope,
            n_routed, n_shared, top_k, expert_scale, gamma, balance_alpha, dsa_topk, local_window,
            n_indexer_heads, d_indexer_head, rms_eps, rope_base, rope_type, pope_delta_init,
            original_seq_len, rope_factor, beta_fast, beta_slow, mscale,
            indexer_use_fp8, indexer_use_hadamard, use_sparse_kernel
        )

    def forward(self, h_prev, emb_next, dsa_warmup=False, compute_aux=False):
        h = self.norm_h(h_prev)
        e = self.norm_emb(emb_next)
        h_proj = self.proj(torch.cat((h, e), dim=-1))
        x, attn_aux, moe_balance, moe_stats = self.block(h_proj, dsa_warmup=dsa_warmup, compute_aux=compute_aux)
        return x, attn_aux, moe_balance, moe_stats
