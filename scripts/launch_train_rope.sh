#!/usr/bin/env bash
# RoPE training script (~200M params, 25B tokens)
# Phase 1: DSA warmup at SEQ_LEN=512, indexer trains (model frozen)
# Phase 2: Full training at SEQ_LEN_FINAL=1024
set -euo pipefail

# Use system ptxas for Triton (required for Blackwell B300 sm_103a support)
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

PROJECT="${WANDB_PROJECT:-cinnamon}"
RUN_PREFIX="${RUN_PREFIX:-}"
BASE_NAME="rope-200M"
if [[ -n "$RUN_PREFIX" ]]; then
  RUN_NAME="${RUN_PREFIX}-${BASE_NAME}"
else
  RUN_NAME="${RUN_NAME:-$BASE_NAME}"
fi

# 25B tokens
MAX_TOKENS="${MAX_TOKENS:-25000000000}"

# Phase 1: DSA warmup at 512 seq len
# Phase 2: switches to 1024
SEQ_LEN="${SEQ_LEN:-512}"
SEQ_LEN_FINAL="${SEQ_LEN_FINAL:-1024}"

# DSA warmup: ~1% of training (conservative for from-scratch joint training)
# 250M tokens / (384 * 1 * 512 * 1 GPU) = ~1.3k steps at seq=512
# After seq_len switch to 1024: steps are 2x bigger, so ~650 steps equivalent
DSA_WARMUP_STEPS="${DSA_WARMUP_STEPS:-1500}"

# B300 (262GB VRAM): Memory calculation
#   Dense warmup (seq=512):  batch * 8 * 512^2 * 2 * 20 = batch * 84MB
#   Sparse training (seq=1024, top-128): batch * 8 * 1024 * 128 * 2 * 20 = batch * 42MB
# Sparse uses HALF the attention memory! So we can double batch after warmup.
BATCH_SIZE="${BATCH_SIZE:-384}"          # ~32GB attn during dense warmup
BATCH_SIZE_SPARSE="${BATCH_SIZE_SPARSE:-768}"  # ~32GB attn during sparse (same footprint!)
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-500}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"  # Aggressive for spot instance
# LR scaled with sqrt rule: 3e-4 * sqrt(384/24) = 3e-4 * 4 = 1.2e-3
LR="${LR:-1.2e-3}"

# GPU config - single B300 per model (262GB VRAM is plenty for 200M)
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

args=(
  --rope
  --run-name "$RUN_NAME"
  --wandb-project "$PROJECT"
  --max-tokens "$MAX_TOKENS"
  --seq-len "$SEQ_LEN"
  --max-seq-len "$SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --batch-size-sparse "$BATCH_SIZE_SPARSE"
  --accumulation-steps "$ACCUMULATION_STEPS"
  --eval-steps "$EVAL_STEPS"
  --checkpoint-steps "$CHECKPOINT_STEPS"
  --lr "$LR"
  --dsa-warmup-steps "$DSA_WARMUP_STEPS"
  --d-rope 128  # 2x PoPE's d_rope=64 so D_qk matches (64+128=192 for both)
)

if [[ -n "$SEQ_LEN_FINAL" ]]; then
  args+=(--seq-len-final "$SEQ_LEN_FINAL" --max-seq-len "$SEQ_LEN_FINAL")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  src/train.py "${args[@]}" "$@"
