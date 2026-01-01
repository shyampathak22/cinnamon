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
    k_ts: int = 256
    local_window: int = 128

@dataclass
class TrainConfig:
    lr: float = 3e-4
    max_tokens: int = 1000000000  # 1B tokens for quick baseline
    batch_size: int = 4
    accumulation_steps: int = 8
    seq_len: int = 1024
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_steps: int = 100
    log_steps: int = 5
    checkpoint_steps: int = 500
    seed: int = 42 # TIL this is a reference from the hitchhiker's guide to the galaxy! pretty cool!
    peak_flops: float = 23.7e12
    mtp_lambda: float = 0.3
    @property
    def max_steps(self):
        return self.max_tokens // (self.batch_size * self.seq_len * self.accumulation_steps)
    @property
    def warmup_steps(self):
        return self.max_steps // 10
