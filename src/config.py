from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 8
    vocab_size: int = 50257  # gpt2 tokenizer
    hidden_dim: int = 1408
    n_heads: int = 8
    max_seq_len: int = 1024
    d_ckv: int = 256  # kv_lora_rank in MLA
    d_cq: int = 256   # q_lora_rank in MLA
    d_head: int = 64  # qk_nope_head_dim in MLA
    d_v: int = 64     # v_head_dim in MLA (default: same as d_head)
    d_rope: int = 32  # qk_rope_head_dim in MLA (base dim for PoPE)
    n_routed: int = 8
    n_shared: int = 1
    top_k: int = 2
    expert_scale: int = 4
    gamma: float = 0.001
    mtp_depth: int = 1
    dsa_topk: int = 64  # tokens selected by DSA
    local_window: int = 0  # 0 disables forced local selection (DeepSeek-V3.2)
    n_indexer_heads: int = 2  # DeepSeek-V3.2 uses small head count for indexer
    d_indexer_head: int = 64
    indexer_use_fp8: bool = True
    indexer_use_hadamard: bool = True
    # Normalization and positional encoding
    rms_eps: float = 1e-6
    rope_base: float = 10000.0
    original_seq_len: int = 4096
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    rope_type: str = 'rope'  # 'rope' (standard RoPE) or 'pope' (Polar PE)
    pope_delta_init: str = "zero"  # "zero" or "uniform"

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
    mtp_lambda_final: float = 0.1
    mtp_lambda_switch_tokens: int = 10_000_000_000_000  # 10T
    dsa_kl_weight: float = 1.0
    dsa_warmup_steps: int = 0
    dsa_warmup_lr: float = 1e-3
    moe_balance_alpha: float = 1e-2
    gamma_final: float = 0.0
    gamma_switch_tokens: int = 14_300_000_000_000  # 14.3T
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
