import torch
import pytest

from attention import RoPE, PoPE, DSAIndexer, MultiheadLatentAttention


class TestRoPE:
    def test_output_shape(self, cfg):
        """RoPE preserves input shape."""
        rope = RoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
        x = torch.randn(4, 128, 8, cfg.d_rope)
        out = rope(x)
        assert out.shape == x.shape

    def test_sparse_positions(self, cfg):
        """RoPE handles custom positions for sparse attention."""
        rope = RoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
        positions = torch.randint(0, 128, (4, 128, 32))
        x = torch.randn(4, 128, 32, cfg.d_rope)
        out = rope(x, positions=positions)
        assert out.shape == x.shape

    def test_interleaved_vs_split(self, cfg):
        """RoPE supports both interleaved and split formats."""
        rope = RoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
        x = torch.randn(4, 128, 8, cfg.d_rope)
        out_interleaved = rope(x, interleaved=True)
        out_split = rope(x, interleaved=False)
        assert out_interleaved.shape == out_split.shape

    def test_cache_extension(self, cfg):
        """RoPE extends cache for longer sequences."""
        rope = RoPE(cfg.d_rope, max_seq_len=64, base=cfg.rope_base)
        x = torch.randn(4, 128, 8, cfg.d_rope)  # longer than cached
        out = rope(x)
        assert out.shape == x.shape


class TestPoPE:
    def test_output_shape(self, cfg):
        """PoPE outputs 2x input dimension (cos + sin)."""
        pope = PoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
        x = torch.randn(4, 128, 8, cfg.d_rope)
        out = pope(x)
        assert out.shape == (*x.shape[:-1], cfg.d_rope * 2)

    def test_delta_range(self, cfg):
        """PoPE delta is clamped to [-2pi, 0]."""
        import math
        pope = PoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base, delta_init="uniform")
        delta = pope.delta.clamp(-2 * math.pi, 0.0)
        assert (delta >= -2 * math.pi).all()
        assert (delta <= 0).all()

    def test_query_vs_key(self, cfg):
        """PoPE applies delta only to keys."""
        pope = PoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
        pope.delta.data.fill_(-1.0)  # non-zero delta
        x = torch.randn(4, 128, 8, cfg.d_rope)
        out_query = pope(x, apply_delta=False)
        out_key = pope(x, apply_delta=True)
        assert not torch.allclose(out_query, out_key)

    def test_custom_positions(self, cfg):
        """PoPE handles custom positions."""
        pope = PoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base)
        positions = torch.randint(0, 128, (4, 128, 32))
        x = torch.randn(4, 128, 32, cfg.d_rope)
        out = pope(x, positions=positions)
        assert out.shape == (*x.shape[:-1], cfg.d_rope * 2)

    @pytest.mark.cuda
    def test_extrapolation(self, cfg):
        """PoPE handles sequences beyond training length."""
        pope = PoPE(cfg.d_rope, cfg.max_seq_len, cfg.rope_base).cuda()
        for seq_len in [1024, 4096, 16384]:
            x = torch.randn(4, seq_len, 1, cfg.d_rope).cuda()
            out = pope(x)
            assert out.shape == (4, seq_len, 1, cfg.d_rope * 2)
            assert not torch.isnan(out).any()


class TestDSAIndexer:
    def test_output_shapes(self, cfg):
        """DSAIndexer returns correct shapes for scores and indices."""
        indexer = DSAIndexer(
            cfg.d_model, cfg.d_cq, cfg.n_indexer_heads, cfg.d_indexer_head,
            cfg.d_rope, cfg.max_seq_len, cfg.rope_base, cfg.rope_type,
            cfg.dsa_topk, use_fp8=False, use_hadamard=False
        )
        h = torch.randn(4, 128, cfg.d_model)
        q_latent = torch.randn(4, 128, cfg.d_cq)

        scores = indexer(h, q_latent, return_scores=True)
        assert scores.shape == (4, 128, 128)

        indices = indexer(h, q_latent, return_scores=False)
        expected_k = min(128, cfg.dsa_topk) if cfg.dsa_topk > 0 else 128
        assert indices.shape == (4, 128, expected_k)

    def test_causal_mask(self, cfg):
        """DSAIndexer respects causal mask."""
        indexer = DSAIndexer(
            cfg.d_model, cfg.d_cq, cfg.n_indexer_heads, cfg.d_indexer_head,
            cfg.d_rope, cfg.max_seq_len, cfg.rope_base, cfg.rope_type,
            cfg.dsa_topk, use_fp8=False, use_hadamard=False
        )
        h = torch.randn(2, 64, cfg.d_model)
        q_latent = torch.randn(2, 64, cfg.d_cq)
        causal = torch.triu(torch.full((64, 64), float("-inf")), diagonal=1).unsqueeze(0)

        scores = indexer(h, q_latent, mask=causal, return_scores=True)
        # Future positions should have -inf scores
        assert (scores[:, 0, 1:] == float("-inf")).all()


class TestMultiheadLatentAttention:
    def test_rope_output_shape(self, cfg):
        """MLA with RoPE produces correct output shape."""
        mla = MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, cfg.max_seq_len, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='rope',
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(4, 128, cfg.d_model)
        out, aux = mla(x)
        assert out.shape == x.shape
        assert aux is None  # No aux loss when compute_aux=False

    def test_pope_output_shape(self, cfg):
        """MLA with PoPE produces correct output shape."""
        mla = MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, cfg.max_seq_len, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='pope',
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(4, 128, cfg.d_model)
        out, aux = mla(x)
        assert out.shape == x.shape

    def test_dsa_warmup_mode(self, cfg):
        """MLA in DSA warmup mode uses dense attention."""
        mla = MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, cfg.max_seq_len, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='rope',
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(4, 128, cfg.d_model)
        out_warmup, _ = mla(x, dsa_warmup=True)
        out_sparse, _ = mla(x, dsa_warmup=False)
        assert out_warmup.shape == out_sparse.shape

    def test_aux_loss_computation(self, cfg):
        """MLA computes auxiliary KL loss when requested."""
        mla = MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, cfg.max_seq_len, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='rope',
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(4, 128, cfg.d_model)
        _, aux = mla(x, compute_aux=True)
        assert aux is not None
        assert aux.ndim == 0  # scalar loss

    def test_gradient_flow(self, small_cfg):
        """Gradients flow through MLA."""
        mla = MultiheadLatentAttention(
            small_cfg.d_model, small_cfg.d_ckv, small_cfg.d_cq, small_cfg.n_heads,
            small_cfg.d_head, small_cfg.d_v, small_cfg.d_rope, small_cfg.max_seq_len,
            small_cfg.dsa_topk, small_cfg.local_window, small_cfg.n_indexer_heads,
            small_cfg.d_indexer_head, small_cfg.rms_eps, small_cfg.rope_base,
            rope_type='rope', indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(2, 32, small_cfg.d_model, requires_grad=True)
        out, _ = mla(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
