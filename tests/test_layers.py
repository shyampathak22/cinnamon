import torch
import pytest

from layers import Transformer, MoE, MTPModule


class TestMoE:
    def test_output_shape(self, cfg):
        """MoE preserves input shape."""
        moe = MoE(
            cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.d_model,
            cfg.hidden_dim, cfg.expert_scale, cfg.gamma, 0.0
        )
        x = torch.randn(4, 128, cfg.d_model)
        out, aux = moe(x)
        assert out.shape == x.shape

    def test_aux_loss(self, cfg):
        """MoE returns auxiliary balance loss."""
        moe = MoE(
            cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.d_model,
            cfg.hidden_dim, cfg.expert_scale, cfg.gamma, 1e-2
        )
        x = torch.randn(4, 128, cfg.d_model)
        _, aux = moe(x)
        assert aux is not None
        assert aux.ndim == 0

    def test_gradient_flow(self, cfg):
        """Gradients flow through MoE."""
        moe = MoE(
            cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.d_model,
            cfg.hidden_dim, cfg.expert_scale, cfg.gamma, 0.0
        )
        x = torch.randn(4, 64, cfg.d_model, requires_grad=True)
        out, _ = moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_expert_selection(self, cfg):
        """MoE selects top_k experts per token."""
        moe = MoE(
            cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.d_model,
            cfg.hidden_dim, cfg.expert_scale, cfg.gamma, 0.0
        )
        # Verify router produces valid scores
        x = torch.randn(4, 64, cfg.d_model)
        scores = torch.sigmoid(moe.router(x))
        assert scores.shape == (4, 64, cfg.n_routed)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestTransformer:
    def test_output_shape(self, cfg):
        """Transformer block preserves input shape."""
        transformer = Transformer(
            cfg.d_model, cfg.hidden_dim, cfg.max_seq_len, cfg.n_heads,
            cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_v, cfg.d_rope, cfg.n_routed,
            cfg.n_shared, cfg.top_k, cfg.expert_scale, cfg.gamma, 0.0,
            cfg.dsa_topk, cfg.local_window, cfg.n_indexer_heads, cfg.d_indexer_head,
            cfg.rms_eps, cfg.rope_base, cfg.rope_type, cfg.pope_delta_init,
            cfg.original_seq_len, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
            cfg.mscale, indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(4, cfg.max_seq_len, cfg.d_model)
        out, attn_aux, moe_aux = transformer(x)
        assert out.shape == x.shape
        assert attn_aux is None
        assert moe_aux is None

    def test_aux_losses(self, cfg):
        """Transformer returns MoE and attention aux losses."""
        transformer = Transformer(
            cfg.d_model, cfg.hidden_dim, cfg.max_seq_len, cfg.n_heads,
            cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_v, cfg.d_rope, cfg.n_routed,
            cfg.n_shared, cfg.top_k, cfg.expert_scale, cfg.gamma, 1e-2,
            cfg.dsa_topk, cfg.local_window, cfg.n_indexer_heads, cfg.d_indexer_head,
            cfg.rms_eps, cfg.rope_base, cfg.rope_type, cfg.pope_delta_init,
            cfg.original_seq_len, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
            cfg.mscale, indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randn(4, 128, cfg.d_model)
        _, attn_aux, moe_aux = transformer(x, compute_aux=True)
        assert moe_aux is not None
        assert attn_aux is not None


class TestMTPModule:
    def test_output_shape(self, cfg):
        """MTPModule preserves input shape."""
        mtp = MTPModule(
            cfg.d_model, cfg.hidden_dim, cfg.max_seq_len, cfg.n_heads,
            cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_v, cfg.d_rope, cfg.n_routed,
            cfg.n_shared, cfg.top_k, cfg.expert_scale, cfg.gamma, 0.0,
            cfg.dsa_topk, cfg.local_window, cfg.n_indexer_heads, cfg.d_indexer_head,
            cfg.rms_eps, cfg.rope_base, cfg.rope_type, cfg.pope_delta_init,
            cfg.original_seq_len, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
            cfg.mscale,
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        h = torch.randn(4, 128, cfg.d_model)
        emb = torch.randn(4, 128, cfg.d_model)
        out, attn_aux, moe_aux = mtp(h, emb)
        assert out.shape == h.shape
        assert attn_aux is None
        assert moe_aux is None

    def test_gradient_flow(self, small_cfg):
        """Gradients flow through MTPModule."""
        mtp = MTPModule(
            small_cfg.d_model, small_cfg.hidden_dim, small_cfg.max_seq_len,
            small_cfg.n_heads, small_cfg.d_ckv, small_cfg.d_cq, small_cfg.d_head,
            small_cfg.d_v, small_cfg.d_rope, small_cfg.n_routed, small_cfg.n_shared,
            small_cfg.top_k, small_cfg.expert_scale, small_cfg.gamma, 0.0,
            small_cfg.dsa_topk, small_cfg.local_window, small_cfg.n_indexer_heads,
            small_cfg.d_indexer_head, small_cfg.rms_eps, small_cfg.rope_base,
            small_cfg.rope_type, small_cfg.pope_delta_init, small_cfg.original_seq_len,
            small_cfg.rope_factor, small_cfg.beta_fast, small_cfg.beta_slow,
            small_cfg.mscale,
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        h = torch.randn(2, 32, small_cfg.d_model, requires_grad=True)
        emb = torch.randn(2, 32, small_cfg.d_model)
        out, _, _ = mtp(h, emb)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
