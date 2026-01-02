import torch
import pytest

from norm import RMSNorm


class TestRMSNorm:
    def test_output_shape(self, cfg):
        """RMSNorm preserves input shape."""
        rms = RMSNorm(cfg.d_model, cfg.rms_eps)
        x = torch.randn(32, cfg.max_seq_len, cfg.d_model)
        out = rms(x)
        assert out.shape == x.shape

    def test_normalization(self, cfg):
        """RMSNorm produces normalized output."""
        rms = RMSNorm(cfg.d_model, cfg.rms_eps)
        x = torch.randn(4, 16, cfg.d_model)
        out = rms(x)
        rms_values = torch.sqrt((out ** 2).mean(dim=-1))
        assert rms_values.mean().item() == pytest.approx(1.0, abs=0.1)

    def test_gradient_flow(self, cfg):
        """Gradients flow through RMSNorm."""
        rms = RMSNorm(cfg.d_model, cfg.rms_eps)
        x = torch.randn(4, 16, cfg.d_model, requires_grad=True)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
