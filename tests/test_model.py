import torch
import pytest

from model import Cinnamon


class TestCinnamon:
    def test_output_shapes(self, small_cfg):
        """Cinnamon produces correct output shapes."""
        model = Cinnamon(
            small_cfg.d_model, small_cfg.n_layers, small_cfg.vocab_size,
            small_cfg.hidden_dim, small_cfg.n_heads, small_cfg.max_seq_len,
            small_cfg.d_ckv, small_cfg.d_cq, small_cfg.d_head, small_cfg.d_v,
            small_cfg.d_rope, small_cfg.n_routed, small_cfg.n_shared,
            small_cfg.top_k, small_cfg.expert_scale, small_cfg.gamma, 1e-2,
            small_cfg.dsa_topk, small_cfg.local_window, small_cfg.n_indexer_heads,
            small_cfg.d_indexer_head, small_cfg.rms_eps, small_cfg.rope_base,
            small_cfg.rope_type, small_cfg.mtp_depth, small_cfg.pope_delta_init,
            small_cfg.original_seq_len, small_cfg.rope_factor, small_cfg.beta_fast,
            small_cfg.beta_slow, small_cfg.mscale,
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randint(0, small_cfg.vocab_size, (4, small_cfg.max_seq_len))
        out, mtp_out = model(x)

        assert out.shape == (4, small_cfg.max_seq_len, small_cfg.vocab_size)
        assert len(mtp_out) == small_cfg.mtp_depth
        for depth, mtp in enumerate(mtp_out, start=1):
            assert mtp.shape == (4, small_cfg.max_seq_len - depth, small_cfg.vocab_size)

    def test_parameter_count(self, small_cfg):
        """Model has expected parameter structure."""
        model = Cinnamon(
            small_cfg.d_model, small_cfg.n_layers, small_cfg.vocab_size,
            small_cfg.hidden_dim, small_cfg.n_heads, small_cfg.max_seq_len,
            small_cfg.d_ckv, small_cfg.d_cq, small_cfg.d_head, small_cfg.d_v,
            small_cfg.d_rope, small_cfg.n_routed, small_cfg.n_shared,
            small_cfg.top_k, small_cfg.expert_scale, small_cfg.gamma, 1e-2,
            small_cfg.dsa_topk, small_cfg.local_window, small_cfg.n_indexer_heads,
            small_cfg.d_indexer_head, small_cfg.rms_eps, small_cfg.rope_base,
            small_cfg.rope_type, small_cfg.mtp_depth, small_cfg.pope_delta_init,
            small_cfg.original_seq_len, small_cfg.rope_factor, small_cfg.beta_fast,
            small_cfg.beta_slow, small_cfg.mscale,
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0
        assert trainable == total  # All params should be trainable

    def test_gradient_flow(self, small_cfg):
        """Gradients flow through entire model."""
        small_cfg.n_layers = 1
        small_cfg.mtp_depth = 1
        model = Cinnamon(
            small_cfg.d_model, small_cfg.n_layers, small_cfg.vocab_size,
            small_cfg.hidden_dim, small_cfg.n_heads, small_cfg.max_seq_len,
            small_cfg.d_ckv, small_cfg.d_cq, small_cfg.d_head, small_cfg.d_v,
            small_cfg.d_rope, small_cfg.n_routed, small_cfg.n_shared,
            small_cfg.top_k, small_cfg.expert_scale, small_cfg.gamma, 0.0,
            small_cfg.dsa_topk, small_cfg.local_window, small_cfg.n_indexer_heads,
            small_cfg.d_indexer_head, small_cfg.rms_eps, small_cfg.rope_base,
            small_cfg.rope_type, small_cfg.mtp_depth, small_cfg.pope_delta_init,
            small_cfg.original_seq_len, small_cfg.rope_factor, small_cfg.beta_fast,
            small_cfg.beta_slow, small_cfg.mscale,
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randint(0, small_cfg.vocab_size, (2, 32))
        out, mtp_out = model(x)
        loss = out.sum() + sum(m.sum() for m in mtp_out)
        loss.backward()

        # Check embedding gradients
        assert model.embedding.weight.grad is not None
        assert not torch.isnan(model.embedding.weight.grad).any()

    def test_aux_loss_aggregation(self, small_cfg):
        """Model aggregates auxiliary losses from all layers."""
        small_cfg.n_layers = 2
        model = Cinnamon(
            small_cfg.d_model, small_cfg.n_layers, small_cfg.vocab_size,
            small_cfg.hidden_dim, small_cfg.n_heads, small_cfg.max_seq_len,
            small_cfg.d_ckv, small_cfg.d_cq, small_cfg.d_head, small_cfg.d_v,
            small_cfg.d_rope, small_cfg.n_routed, small_cfg.n_shared,
            small_cfg.top_k, small_cfg.expert_scale, small_cfg.gamma, 1e-2,
            small_cfg.dsa_topk, small_cfg.local_window, small_cfg.n_indexer_heads,
            small_cfg.d_indexer_head, small_cfg.rms_eps, small_cfg.rope_base,
            small_cfg.rope_type, small_cfg.mtp_depth, small_cfg.pope_delta_init,
            small_cfg.original_seq_len, small_cfg.rope_factor, small_cfg.beta_fast,
            small_cfg.beta_slow, small_cfg.mscale,
            indexer_use_fp8=False, indexer_use_hadamard=False
        )
        x = torch.randint(0, small_cfg.vocab_size, (2, 32))
        out, mtp_out, dsa_kl, moe_balance = model(x, compute_aux=True)

        assert dsa_kl is not None
        assert moe_balance is not None
        assert dsa_kl.ndim == 0
        assert moe_balance.ndim == 0
