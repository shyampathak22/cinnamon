from dataclasses import dataclass

@dataclass
class ModelConfig:
    # ~202.6M params (scaled from 27M baseline for 10B token training)
    d_model: int = 512
    n_layers: int = 20
    vocab_size: int = 50257  # gpt2 tokenizer
    hidden_dim: int = 1536   # ~3x d_model (DeepSeek ratio)
    n_heads: int = 8
    max_seq_len: int = 1024  # model cache length (can be increased for eval)
    d_ckv: int = 256   # kv_lora_rank in MLA (scaled 2x)
    d_cq: int = 256    # q_lora_rank in MLA (scaled 2x)
    d_head: int = 64   # qk_nope_head_dim in MLA (scaled 2x)
    d_v: int = 64      # v_head_dim in MLA (scaled 2x)
    d_rope: int = 64   # qk_rope_head_dim in MLA (PoPE doubles this internally)
    n_routed: int = 8
    n_shared: int = 1
    top_k: int = 2
    expert_scale: int = 4
    gamma: float = 0.001  # DeepSeek V3 bias update speed
    mtp_depth: int = 1
    dsa_topk: int = 128   # tokens selected by DSA (~12.5% at 1024 context)
    local_window: int = 0  # 0 disables forced local selection (DeepSeek-V3.2)
    n_indexer_heads: int = 4  # scaled for larger model
    d_indexer_head: int = 64
    indexer_use_fp8: bool = False
    indexer_use_hadamard: bool = True
    # Normalization and positional encoding
    rms_eps: float = 1e-6
    rope_base: float = 10000.0
    original_seq_len: int = 1024  # training context length (YaRN reference)
    rope_factor: float = 1.0  # YaRN scaling factor (1.0 disables)
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    rope_type: str = 'pope'  # 'rope' (standard RoPE) or 'pope' (Polar PE)
    pope_delta_init: str = "zero"  # "zero" or "uniform"
    use_sparse_kernel: bool = True  # TRUE sparse attention kernel (O(L*K) vs O(L^2))

@dataclass
class TrainConfig:
    # Scaled for 25B tokens, ~200M params (DeepSeek V3 ratios)
    lr: float = 3e-4  # restored after adding RoPE projection norms
    max_tokens: int = 25_000_000_000  # 25B tokens
    batch_size: int = 2
    batch_size_sparse: int | None = None  # batch size after DSA warmup (sparse attn uses less mem)
    accumulation_steps: int = 32  # effective batch = 2*32*1024 = 65k tokens/step
    seq_len: int = 1024  # training context length (align with ModelConfig)
    seq_len_final: int | None = None  # optional post-warmup seq len
    grad_clip: float = 1.0  # DeepSeek V3: 1.0
    weight_decay: float = 0.1  # DeepSeek V3: 0.1
    eval_steps: int = 500
    log_steps: int = 10
    checkpoint_steps: int = 500  # aggressive for spot instances
    seed: int = 42
    peak_flops: float = 2250e12  # B300: ~2.25 PFLOPS FP8
    # MTP schedule: λ=0.3→0.1 at 67.6% (DeepSeek V3: 10T/14.8T)
    mtp_lambda: float = 0.3
    mtp_lambda_final: float = 0.1
    mtp_lambda_switch_ratio: float = 0.676  # 67.6% of training
    dsa_kl_weight: float = 1.0
    # DSA warmup: dense attention phase where indexer learns from model's attention patterns
    # Joint training: both model and indexer train together (not frozen model + indexer only)
    dsa_warmup_steps: int = 0
    # Balance loss: DeepSeek V3 uses α=0.0001 (NOT 0.01!)
    moe_balance_alpha: float = 1e-4
    # Gamma schedule: γ=0.001→0.0 at 96.6% (DeepSeek V3: 14.3T/14.8T)
    gamma_final: float = 0.0
    gamma_switch_ratio: float = 0.966  # 96.6% of training
    # DataLoader settings (tuned for 120 CPU B300 box)
    num_workers: int = 32  # ~1/4 of CPUs, leave room for other processes
    pin_memory: bool = True
    prefetch_factor: int = 8  # aggressive prefetch for large batches
    # FP8 training (requires SM89+ for compute benefits, otherwise storage-only)
    use_fp8: bool = True  # DeepSeek V3 style block-wise FP8
    @property
    def max_steps(self):
        return self.max_tokens // (self.batch_size * self.seq_len * self.accumulation_steps)
    @property
    def warmup_steps(self):
        return self.max_steps // 10
    @property
    def mtp_lambda_switch_tokens(self):
        return int(self.max_tokens * self.mtp_lambda_switch_ratio)
    @property
    def gamma_switch_tokens(self):
        return int(self.max_tokens * self.gamma_switch_ratio)
