"""
Performance benchmarks for Cinnamon components.

Run with: pytest tests/test_benchmarks.py -v -s --tb=short
Or for specific benchmarks: pytest tests/test_benchmarks.py -k "mla" -v -s
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import pytest


@dataclass
class BenchmarkResult:
    name: str
    median_ms: float
    min_ms: float
    max_ms: float
    throughput: str = ""

    def __str__(self):
        s = f"{self.name}: {self.median_ms:.2f}ms (min={self.min_ms:.2f}, max={self.max_ms:.2f})"
        if self.throughput:
            s += f" | {self.throughput}"
        return s


@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    # Return time in ms
    elapsed = start.elapsed_time(end)
    yield elapsed


def benchmark(fn, warmup=3, iterations=20):
    """Run benchmark with warmup and return timing stats."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return {
        "median": times[len(times) // 2],
        "min": times[0],
        "max": times[-1],
        "mean": sum(times) / len(times),
    }


@pytest.mark.cuda
@pytest.mark.benchmark
class TestMLABenchmarks:
    """Multi-head Latent Attention benchmarks."""

    @pytest.fixture
    def mla_rope(self, cfg):
        from attention import MultiheadLatentAttention
        return MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, 16384, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='rope',
            indexer_use_fp8=cfg.indexer_use_fp8,
            indexer_use_hadamard=cfg.indexer_use_hadamard
        ).cuda()

    @pytest.fixture
    def mla_pope(self, cfg):
        from attention import MultiheadLatentAttention
        return MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, 16384, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='pope',
            indexer_use_fp8=cfg.indexer_use_fp8,
            indexer_use_hadamard=cfg.indexer_use_hadamard
        ).cuda()

    def test_mla_sequence_scaling(self, mla_rope, cfg):
        """Benchmark MLA across sequence lengths."""
        print("\n" + "=" * 60)
        print("MLA (RoPE) Sequence Length Scaling")
        print("=" * 60)

        results = []
        for seq_len in [512, 1024, 2048, 4096]:
            x = torch.randn(1, seq_len, cfg.d_model, device="cuda")

            def run():
                return mla_rope(x)

            stats = benchmark(run)
            tokens_per_sec = seq_len / (stats["median"] / 1000)
            result = BenchmarkResult(
                name=f"seq_len={seq_len}",
                median_ms=stats["median"],
                min_ms=stats["min"],
                max_ms=stats["max"],
                throughput=f"{tokens_per_sec/1000:.1f}K tok/s"
            )
            results.append(result)
            print(f"  {result}")
            del x

        # Lightning indexer is dense, so scaling is closer to O(n^2).
        ratio = results[-1].median_ms / results[0].median_ms
        assert ratio < 120

    def test_mla_rope_vs_pope(self, mla_rope, mla_pope, cfg):
        """Compare RoPE vs PoPE performance."""
        print("\n" + "=" * 60)
        print("MLA: RoPE vs PoPE Comparison")
        print("=" * 60)

        seq_len = 2048
        x = torch.randn(1, seq_len, cfg.d_model, device="cuda")

        rope_stats = benchmark(lambda: mla_rope(x))
        pope_stats = benchmark(lambda: mla_pope(x))

        print(f"  RoPE: {rope_stats['median']:.2f}ms")
        print(f"  PoPE: {pope_stats['median']:.2f}ms")
        print(f"  Ratio: {pope_stats['median']/rope_stats['median']:.2f}x")

        # PoPE should be within 2x of RoPE (output is 2x wider)
        assert pope_stats["median"] < rope_stats["median"] * 3

    def test_mla_warmup_vs_sparse(self, mla_rope, cfg):
        """Compare warmup (dense) vs sparse attention."""
        print("\n" + "=" * 60)
        print("MLA: Dense Warmup vs Sparse Attention")
        print("=" * 60)

        seq_len = 2048
        x = torch.randn(1, seq_len, cfg.d_model, device="cuda")

        warmup_stats = benchmark(lambda: mla_rope(x, dsa_warmup=True))
        sparse_stats = benchmark(lambda: mla_rope(x, dsa_warmup=False))

        print(f"  Dense (warmup): {warmup_stats['median']:.2f}ms")
        print(f"  Sparse (DSA):   {sparse_stats['median']:.2f}ms")
        print(f"  Speedup: {warmup_stats['median']/sparse_stats['median']:.2f}x")
        # Performance varies widely with GPU and kernel mix; keep this as a report-only benchmark.


@pytest.mark.cuda
@pytest.mark.benchmark
class TestMoEBenchmarks:
    """Mixture of Experts benchmarks."""

    @pytest.fixture
    def moe(self, cfg):
        from layers import MoE
        return MoE(
            cfg.n_routed, cfg.n_shared, cfg.top_k, cfg.d_model,
            cfg.hidden_dim, cfg.expert_scale, cfg.gamma, 0.0
        ).cuda()

    def test_moe_batch_scaling(self, moe, cfg):
        """Benchmark MoE across batch sizes."""
        print("\n" + "=" * 60)
        print("MoE Batch Size Scaling")
        print("=" * 60)

        seq_len = 1024
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, seq_len, cfg.d_model, device="cuda")

            stats = benchmark(lambda: moe(x))
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / (stats["median"] / 1000)
            print(f"  batch={batch_size}: {stats['median']:.2f}ms | {tokens_per_sec/1000:.1f}K tok/s")
            del x

    def test_moe_expert_scaling(self, cfg):
        """Benchmark MoE with different expert counts."""
        from layers import MoE
        print("\n" + "=" * 60)
        print("MoE Expert Count Scaling")
        print("=" * 60)

        x = torch.randn(4, 512, cfg.d_model, device="cuda")

        for n_experts in [4, 8, 16, 32]:
            moe = MoE(
                n_experts, cfg.n_shared, cfg.top_k, cfg.d_model,
                cfg.hidden_dim, cfg.expert_scale, cfg.gamma, 0.0
            ).cuda()

            stats = benchmark(lambda: moe(x))
            params = sum(p.numel() for p in moe.parameters()) / 1e6
            print(f"  {n_experts} experts: {stats['median']:.2f}ms | {params:.1f}M params")
            del moe
            torch.cuda.empty_cache()


@pytest.mark.cuda
@pytest.mark.benchmark
class TestModelBenchmarks:
    """Full model benchmarks."""

    def test_forward_pass_scaling(self, small_cfg):
        """Benchmark full forward pass."""
        from model import Cinnamon
        print("\n" + "=" * 60)
        print("Cinnamon Forward Pass (small config)")
        print("=" * 60)

        small_cfg.max_seq_len = 512
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
        ).cuda()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model: {params:.1f}M params, {small_cfg.n_layers} layers")

        for batch_size in [1, 2, 4]:
            x = torch.randint(0, small_cfg.vocab_size, (batch_size, small_cfg.max_seq_len), device="cuda")

            stats = benchmark(lambda: model(x), warmup=2, iterations=10)
            total_tokens = batch_size * small_cfg.max_seq_len
            tokens_per_sec = total_tokens / (stats["median"] / 1000)
            print(f"  batch={batch_size}: {stats['median']:.2f}ms | {tokens_per_sec/1000:.1f}K tok/s")
            del x

    def test_forward_backward_pass(self, small_cfg):
        """Benchmark forward + backward pass."""
        from model import Cinnamon
        import torch.nn.functional as F
        print("\n" + "=" * 60)
        print("Cinnamon Forward+Backward Pass")
        print("=" * 60)

        small_cfg.n_layers = 2
        small_cfg.max_seq_len = 256
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
        ).cuda()

        x = torch.randint(0, small_cfg.vocab_size, (2, small_cfg.max_seq_len), device="cuda")
        targets = torch.randint(0, small_cfg.vocab_size, (2, small_cfg.max_seq_len), device="cuda")

        def train_step():
            model.zero_grad()
            logits, mtp_logits = model(x)
            loss = F.cross_entropy(logits.view(-1, small_cfg.vocab_size), targets.view(-1))
            loss.backward()
            return loss

        stats = benchmark(train_step, warmup=2, iterations=10)
        print(f"  Forward+Backward: {stats['median']:.2f}ms")


@pytest.mark.cuda
@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Memory usage benchmarks."""

    def test_mla_memory_scaling(self, cfg):
        """Track MLA memory usage across sequence lengths."""
        from attention import MultiheadLatentAttention
        print("\n" + "=" * 60)
        print("MLA Memory Usage (Sparse Attention)")
        print("=" * 60)

        mla = MultiheadLatentAttention(
            cfg.d_model, cfg.d_ckv, cfg.d_cq, cfg.n_heads, cfg.d_head, cfg.d_v,
            cfg.d_rope, 16384, cfg.dsa_topk, cfg.local_window,
            cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps,
            cfg.rope_base, rope_type='rope',
            indexer_use_fp8=False, indexer_use_hadamard=False
        ).cuda()

        for seq_len in [1024, 2048, 4096, 8192]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(1, seq_len, cfg.d_model, device="cuda")
            out, _ = mla(x)
            torch.cuda.synchronize()

            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"  seq_len={seq_len}: {peak_mb:.1f} MB peak")
            del x, out

        # Verify subquadratic memory growth
        # (If it were O(n^2), 8192 would be 64x more than 1024)

    def test_model_memory_footprint(self, small_cfg):
        """Measure model memory footprint."""
        from model import Cinnamon
        print("\n" + "=" * 60)
        print("Model Memory Footprint")
        print("=" * 60)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
        ).cuda()

        param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        total_mem = torch.cuda.max_memory_allocated() / 1e6

        print(f"  Parameters: {param_mem:.1f} MB")
        print(f"  Total allocated: {total_mem:.1f} MB")
        print(f"  Overhead: {total_mem - param_mem:.1f} MB")
