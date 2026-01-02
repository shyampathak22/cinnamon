# Cinnamon

A from-scratch implementation of DeepSeek-V3 and DeepSeek-V3.2 architectures, with optional Polar Position Embedding (PoPE) support.

## Features

### Multi-head Latent Attention (MLA)
Low-rank compressed key-value and query representations with decoupled positional encoding:
- KV compression: `x → W_dkv → RMSNorm → W_uk/W_uv`
- Q compression: `x → W_dq → RMSNorm → W_uq`
- Separate RoPE/PoPE projections for position-aware attention

### DeepSeek Sparse Attention (DSA)
Lightning Indexer for efficient token selection at long contexts:
- Learned position-dependent importance weights
- Top-k token selection per query position
- Optional local window guarantee
- KL divergence auxiliary loss for indexer training
- Two-stage training: dense warmup → sparse

### Mixture of Experts (MoE)
Sigmoid-gated sparse expert routing:
- Bias-based load balancing with dynamic updates
- Gate normalization for stable training
- Shared + routed expert architecture
- Sequence-wise auxiliary balance loss

### Multi-Token Prediction (MTP)
Auxiliary prediction heads for improved sample efficiency:
- Full transformer block per MTP depth (not just linear projection)
- Shared embedding and output projection
- Lambda decay during training

### Position Encodings
- **RoPE**: Standard rotary position embedding with YaRN extension support
- **PoPE**: Polar Position Embedding with learnable phase offset δ ∈ [-2π, 0]

### FP8 Training
DeepSeek-style mixed precision:
- Row-wise quantization for activations (1×128 tiles)
- Block-wise quantization for weights (128×128 blocks)
- Custom Triton kernels for quantization and grouped GEMM

## Installation

```bash
git clone <repo-url> cinnamon
cd cinnamon
uv sync --dev
```

## Usage

### Training

```bash
# Train with RoPE (default)
uv run python src/train.py

# Train with PoPE
uv run python src/train.py --rope-type pope

# Key training flags
uv run python src/train.py \
    --lr 3e-4 \
    --batch-size 2 \
    --accumulation-steps 16 \
    --seq-len 1024 \
    --max-tokens 1000000000 \
    --rope-type rope \
    --dsa-warmup-steps 1000
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_attention.py -v

# Run benchmarks (with output)
uv run pytest -m benchmark -v -s

# Skip slow tests
uv run pytest -m "not slow"
```

## Architecture

```
Cinnamon
├── Embedding (shared with lm_head)
├── Transformer Blocks × n_layers
│   ├── RMSNorm
│   ├── Multi-head Latent Attention
│   │   ├── KV Compression (W_dkv → kv_norm → W_uk, W_uv)
│   │   ├── Q Compression (W_dq → q_norm → W_uq)
│   │   ├── Position Encoding (RoPE or PoPE)
│   │   ├── DSA Indexer (Lightning Indexer)
│   │   └── Sparse Attention
│   ├── RMSNorm
│   └── MoE
│       ├── Router (sigmoid gating)
│       ├── Shared Experts
│       └── Routed Experts (top-k selection)
├── RMSNorm
├── LM Head
└── MTP Modules × mtp_depth
    ├── Norm + Concat + Project
    └── Transformer Block
```

## Configuration

### Model Config (`src/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Model dimension |
| `n_layers` | 8 | Number of transformer layers |
| `n_heads` | 8 | Number of attention heads |
| `d_ckv` | 256 | KV compression dimension |
| `d_cq` | 256 | Q compression dimension |
| `d_head` | 64 | Attention head dimension |
| `d_v` | 64 | Value head dimension |
| `d_rope` | 32 | RoPE/PoPE dimension |
| `n_routed` | 8 | Number of routed experts |
| `n_shared` | 1 | Number of shared experts |
| `top_k` | 2 | Experts per token |
| `dsa_topk` | 64 | Tokens selected by DSA |
| `local_window` | 0 | Forced local attention window |
| `mtp_depth` | 1 | Multi-token prediction depth |
| `rope_type` | 'rope' | Position encoding ('rope' or 'pope') |

### Training Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Learning rate |
| `max_tokens` | 1B | Total training tokens |
| `batch_size` | 2 | Batch size per step |
| `accumulation_steps` | 16 | Gradient accumulation |
| `seq_len` | 1024 | Sequence length |
| `mtp_lambda` | 0.3 | MTP loss weight (decays to 0.1) |
| `dsa_kl_weight` | 1.0 | DSA KL divergence loss weight |
| `dsa_warmup_steps` | 0 | Dense attention warmup steps |
| `moe_balance_alpha` | 1e-2 | MoE balance loss weight |
| `use_fp8` | True | Enable FP8 training |

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Base architecture (MLA, MoE, MTP)
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) - DeepSeek Sparse Attention (DSA), Lightning Indexer
- [Decoupling the 'What' and 'Where' With Polar Coordinate Positional Embeddings](https://arxiv.org/abs/2509.10534) - PoPE

## Project Structure

```
cinnamon/
├── src/
│   ├── attention.py    # RoPE, PoPE, DSAIndexer, MLA
│   ├── layers.py       # Transformer, MoE, MTPModule
│   ├── model.py        # Cinnamon model
│   ├── config.py       # Model and training configs
│   ├── kernels.py      # FP8 quantization, Triton kernels
│   ├── norm.py         # RMSNorm
│   ├── train.py        # Training loop
│   └── preprocess.py   # Data preprocessing
├── tests/
│   ├── conftest.py     # Shared fixtures
│   ├── test_attention.py
│   ├── test_layers.py
│   ├── test_model.py
│   ├── test_kernels.py
│   ├── test_benchmarks.py
│   └── test_norm.py
└── pyproject.toml
```

## License

MIT
