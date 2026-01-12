#!/usr/bin/env bash
# PoPE training script (~200M params, 25B tokens)
# Phase 1: DSA warmup at SEQ_LEN=512, indexer trains (model frozen)
# Phase 2: Full training at SEQ_LEN_FINAL=1024
set -euo pipefail

PROJECT="${WANDB_PROJECT:-cinnamon}"
RUN_PREFIX="${RUN_PREFIX:-}"
BASE_NAME="pope-200M"
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
# 245M tokens / (24 * 1 * 512 * 2 GPUs) = ~10k steps
DSA_WARMUP_STEPS="${DSA_WARMUP_STEPS:-10000}"

BATCH_SIZE="${BATCH_SIZE:-24}"  # Divides by 12 to 2 when seq_len switches to 1024
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"  # Multiplies by 8 to 8 when seq_len switches
EVAL_STEPS="${EVAL_STEPS:-500}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"  # Aggressive for spot instance
LR="${LR:-3e-4}"

# GPU config - default to GPUs 2,3 for PoPE
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
MASTER_PORT="${MASTER_PORT:-29501}"

args=(
  --pope
  --run-name "$RUN_NAME"
  --wandb-project "$PROJECT"
  --max-tokens "$MAX_TOKENS"
  --seq-len "$SEQ_LEN"
  --max-seq-len "$SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --accumulation-steps "$ACCUMULATION_STEPS"
  --eval-steps "$EVAL_STEPS"
  --checkpoint-steps "$CHECKPOINT_STEPS"
  --lr "$LR"
  --dsa-warmup-steps "$DSA_WARMUP_STEPS"
)

if [[ -n "$SEQ_LEN_FINAL" ]]; then
  args+=(--seq-len-final "$SEQ_LEN_FINAL" --max-seq-len "$SEQ_LEN_FINAL")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  src/train.py "${args[@]}" "$@"
