# Cinnamon

A from-scratch implementation of DeepSeek-V3 and DeepSeek-V3.2 architectures, with optional Polar Position Embedding (PoPE) support.

NOTE: This is a WIP, I'm currently training/testing the following:

1. Evaluations of RoPE vs PoPE (currently running)
2. Currently debugging an mHC implementation (kernel issues) - Planning to run and ablate this weekend.
4. Debugging fp8 kernels for small model sizes (facing NaN loss issues due to training stability with small models)

So far, I've only been able to attempt training models with less than 30M params at a max of 250M tokens, not nearly enough to see emergent behaviors from the some of the architectural implementations. I would like to train larger models for much longer, of course, but I am limited by my constrained hardware (2x 5060 Ti GPUs). While they have ample memory (32GB between the cards) for the size and batches, the bottleneck becomes the speed of compute.

Further notes: 
- fp8 kernels will NOT work on all GPUs. Please refer to your CUDA compute capability to see if you can run the fp8 training. Otherwise, you can run the scripts with the --disable-fp8 flag, and the model will default to bf16.
- MTP Lambda and MoE Gamma values are set to switch at 67.6% and 96.6% of training, respectively.
- d_rope is used for both PoPE and RoPE dimension configs
- original_seq_len in configs is a YaRN reference. I have not tested the YaRN configurations yet, there could be bugs.
- Some config settings are overridden in the launch scripts for training to allow for a single launch of all 3 train modes.

Also, PLEASE RIP THIS APART! im actively learning, and I need peers to poke holes in my implementations! I welcome any and all criticisms!

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
# Train with RoPE
./scripts/launch_train_rope.sh

# Train with PoPE
./scripts/launch_train_pope.sh

# Train with RoPE + YaRN (for length extrapolation)
./scripts/launch_train_rope_yarn.sh

# Disable FP8 if your GPU doesn't support it
./scripts/launch_train_rope.sh --disable-fp8
```

### Evaluation

```bash
# Evaluate checkpoints on length extrapolation
uv run python src/eval.py artifacts/checkpoints/*/checkpoint_*.pt --lengths 512,1024,2048,4096
```

### Testing

```bash
uv run pytest
```

## Evaluation Results

### PoPE vs RoPE Length Extrapolation

Trained for 250M tokens with d_rope=64, seq_len 512→1024, evaluated on held-out validation data.

> **Note:** The DSA Lightning Indexer was mistakenly trained with RoPE even when rope_type=pope. This means the token selection for sparse attention used RoPE-based patterns, potentially limiting PoPE's effectiveness. These results are not fully representative of PoPE's capacity.

![RoPE vs PoPE Length Extrapolation](plots/rope_vs_pope_extrapolation.png)

| Sequence Length | RoPE (ppl) | PoPE (ppl) | Winner |
|-----------------|------------|------------|--------|
| 512 (train) | **125.51** | 126.36 | RoPE |
| 1024 (train) | **117.77** | 119.13 | RoPE |
| 2048 (2x) | 192.22 | **130.85** | PoPE |
| 4096 (4x) | 281.44 | **140.55** | PoPE |
| 8192 (8x) | 378.78 | **167.95** | PoPE |

**Key findings:**
- At training lengths, RoPE and PoPE perform similarly
- At extrapolation lengths, PoPE significantly outperforms RoPE
- **Degradation (1024 → 8192):** RoPE 3.22x vs PoPE 1.41x
- PoPE maintains much more stable perplexity as context length increases

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Base architecture (MLA, MoE, MTP)
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) - DeepSeek Sparse Attention (DSA), Lightning Indexer
- [Decoupling the 'What' and 'Where' With Polar Coordinate Positional Embeddings](https://arxiv.org/abs/2509.10534) - PoPE
