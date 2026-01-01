from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 8
    vocab_size: int = 50257  # gpt2 tokenizer
    hidden_dim: int = 1408
    n_heads: int = 8
    max_seq_len: int = 1024
    d_ckv: int = 256
    d_cq: int = 256
    d_head: int = 64
    d_rope: int = 32
    n_routed: int = 8
    n_shared: int = 1
    top_k: int = 2
    expert_scale: int = 4
    gamma: float = 0.001
    k_ts: int = 64  # reduced from 256 - 64/1024 = 6.25% sparsity is more efficient
    local_window: int = 64  # reduced to match k_ts
    n_indexer_heads: int = 2  # DeepSeek V3.2 uses small head count for indexer
    # Normalization and positional encoding
    rms_eps: float = 1e-6
    rope_base: float = 10000.0

@dataclass
class TrainConfig:
    lr: float = 3e-4
    max_tokens: int = 1000000000  # 1B tokens for quick baseline
    batch_size: int = 2
    accumulation_steps: int = 16  # Doubled to maintain effective batch size
    seq_len: int = 1024
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_steps: int = 100
    log_steps: int = 5
    checkpoint_steps: int = 500
    seed: int = 42  # TIL this is a reference from the hitchhiker's guide to the galaxy!
    peak_flops: float = 23.7e12
    mtp_lambda: float = 0.3
    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    # FP8 training (requires SM89+ for compute benefits, otherwise storage-only)
    use_fp8: bool = True  # DeepSeek V3 style block-wise FP8
    @property
    def max_steps(self):
        return self.max_tokens // (self.batch_size * self.seq_len * self.accumulation_steps)
    @property
    def warmup_steps(self):
        return self.max_steps // 10
