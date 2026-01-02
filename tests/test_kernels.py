import torch
import torch.nn.functional as F
import pytest


@pytest.fixture
def kernel_imports():
    """Import kernels, skip if not available."""
    try:
        from kernels import (
            quantize_fp8_rowwise,
            quantize_fp8_blockwise,
            dequantize_fp8,
            lightning_indexer_fused,
            moe_grouped_gemm,
        )
        return {
            "quantize_fp8_rowwise": quantize_fp8_rowwise,
            "quantize_fp8_blockwise": quantize_fp8_blockwise,
            "dequantize_fp8": dequantize_fp8,
            "lightning_indexer_fused": lightning_indexer_fused,
            "moe_grouped_gemm": moe_grouped_gemm,
        }
    except Exception as e:
        pytest.skip(f"Kernels not available: {e}")


@pytest.mark.cuda
class TestFP8Quantization:
    def test_rowwise_shapes(self, kernel_imports):
        """Row-wise quantization produces correct shapes."""
        quantize = kernel_imports["quantize_fp8_rowwise"]
        x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        x_fp8, scales = quantize(x, block_size=128)

        assert x_fp8.shape == x.shape
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert scales.shape == (256, 4)  # 512/128 = 4 blocks per row

    def test_blockwise_shapes(self, kernel_imports):
        """Block-wise quantization produces correct shapes."""
        quantize = kernel_imports["quantize_fp8_blockwise"]
        w = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
        w_fp8, scales = quantize(w, block_m=128, block_n=128)

        assert w_fp8.shape == w.shape
        assert w_fp8.dtype == torch.float8_e4m3fn
        assert scales.shape == (4, 4)  # 512/128 = 4 blocks each dim

    def test_roundtrip_accuracy(self, kernel_imports):
        """Quantize-dequantize preserves values within tolerance."""
        quantize = kernel_imports["quantize_fp8_rowwise"]
        dequantize = kernel_imports["dequantize_fp8"]

        x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        x_fp8, scales = quantize(x, block_size=128)
        x_reconstructed = dequantize(x_fp8, scales, block_m=1, block_n=128)

        rel_error = (x.float() - x_reconstructed.float()).abs() / (x.float().abs() + 1e-6)
        assert rel_error.mean().item() < 0.1  # Mean error < 10%

    def test_outlier_handling(self, kernel_imports):
        """Row-wise scaling handles outliers gracefully."""
        quantize = kernel_imports["quantize_fp8_rowwise"]
        dequantize = kernel_imports["dequantize_fp8"]

        x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = 1000.0
        x[100, 200] = -500.0

        x_fp8, scales = quantize(x, block_size=128)
        x_reconstructed = dequantize(x_fp8, scales, block_m=1, block_n=128)

        # Should still produce valid output
        assert not torch.isnan(x_reconstructed).any()
        assert not torch.isinf(x_reconstructed).any()


@pytest.mark.cuda
class TestLightningIndexer:
    def test_output_shape(self, kernel_imports):
        """Lightning indexer produces correct output shape."""
        fused = kernel_imports["lightning_indexer_fused"]
        batch, seq_len, n_head, d_head = 4, 1024, 2, 64

        q = torch.randn(batch, seq_len, n_head, d_head, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seq_len, d_head, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(batch, seq_len, n_head, dtype=torch.float32, device="cuda")

        out = fused(q, k, w)
        assert out.shape == (batch, seq_len, seq_len)

    def test_reference_equivalence(self, kernel_imports):
        """Fused kernel matches reference implementation."""
        fused = kernel_imports["lightning_indexer_fused"]
        batch, seq_len, n_head, d_head = 4, 256, 2, 64

        q = torch.randn(batch, seq_len, n_head, d_head, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seq_len, d_head, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(batch, seq_len, n_head, dtype=torch.float32, device="cuda")

        # Fused
        out_fused = fused(q, k, w)

        # Reference
        scores_ref = F.relu(torch.einsum('bthd,bsd->bths', q.float(), k.float()))
        out_ref = torch.einsum('bths,bth->bts', scores_ref, w)

        rel_error = (out_ref - out_fused.float()).abs() / (out_ref.abs() + 1e-6)
        assert rel_error.mean().item() < 0.01  # < 1% mean error


@pytest.mark.cuda
class TestMoEGroupedGEMM:
    def test_output_shape(self, kernel_imports):
        """Grouped GEMM produces correct output shape."""
        gemm = kernel_imports["moe_grouped_gemm"]
        n_experts, d_model, hidden_dim = 8, 512, 128
        total_tokens = 1024

        weights = torch.randn(n_experts, hidden_dim, d_model, dtype=torch.bfloat16, device="cuda")
        sorted_x = torch.randn(total_tokens, d_model, dtype=torch.bfloat16, device="cuda")

        tokens_per_expert = total_tokens // n_experts
        expert_starts = torch.arange(0, total_tokens, tokens_per_expert, device="cuda")[:n_experts]
        expert_ends = torch.cat([expert_starts[1:], torch.tensor([total_tokens], device="cuda")])

        out = gemm(sorted_x, weights, expert_starts, expert_ends)
        assert out.shape == (total_tokens, hidden_dim)

    def test_reference_equivalence(self, kernel_imports):
        """Grouped GEMM matches sequential per-expert computation."""
        gemm = kernel_imports["moe_grouped_gemm"]
        n_experts, d_model, hidden_dim = 8, 512, 128
        total_tokens = 256

        weights = torch.randn(n_experts, hidden_dim, d_model, dtype=torch.bfloat16, device="cuda")
        sorted_x = torch.randn(total_tokens, d_model, dtype=torch.bfloat16, device="cuda")

        tokens_per_expert = total_tokens // n_experts
        expert_starts = torch.arange(0, total_tokens, tokens_per_expert, device="cuda")[:n_experts]
        expert_ends = torch.cat([expert_starts[1:], torch.tensor([total_tokens], device="cuda")])

        # Grouped
        out_grouped = gemm(sorted_x, weights, expert_starts, expert_ends)

        # Reference
        out_ref = torch.zeros(total_tokens, hidden_dim, dtype=torch.float32, device="cuda")
        for i in range(n_experts):
            start, end = expert_starts[i].item(), expert_ends[i].item()
            if end > start:
                out_ref[start:end] = F.linear(sorted_x[start:end].float(), weights[i].float())

        rel_error = (out_ref - out_grouped.float()).abs() / (out_ref.abs() + 1e-6)
        assert rel_error.mean().item() < 0.01


@pytest.mark.cuda
@pytest.mark.slow
class TestKernelBenchmarks:
    """Performance benchmarks (marked slow, run with pytest -m slow)."""

    def test_fp8_throughput(self, kernel_imports):
        """FP8 quantization achieves reasonable throughput."""
        import time
        quantize = kernel_imports["quantize_fp8_rowwise"]

        x = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(3):
            quantize(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            quantize(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1000

        throughput = (2048 * 2048 * 2) / (elapsed / 1000) / 1e9
        assert throughput > 10  # At least 10 GB/s

    def test_lightning_indexer_scaling(self, kernel_imports):
        """Lightning indexer scales reasonably with sequence length."""
        import time
        fused = kernel_imports["lightning_indexer_fused"]

        times = {}
        for seq_len in [512, 1024, 2048]:
            q = torch.randn(4, seq_len, 2, 64, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(4, seq_len, 64, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(4, seq_len, 2, dtype=torch.float32, device="cuda")

            for _ in range(3):
                fused(q, k, w)

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                fused(q, k, w)
            torch.cuda.synchronize()
            times[seq_len] = (time.perf_counter() - start) / 20 * 1000

        # 2x seq_len should be roughly 4x time (O(n^2))
        ratio = times[2048] / times[1024]
        assert 2.0 < ratio < 6.0  # Allow some variance
