#!/usr/bin/env bash
# PoPE training script (~202M params, 10B tokens)
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

# 10B tokens (DeepSeek V3 ratios for MTP/gamma switches in config.py)
MAX_TOKENS="${MAX_TOKENS:-10000000000}"

# Phase 1: DSA warmup at 512 seq len
# Phase 2: switches to 1024
SEQ_LEN="${SEQ_LEN:-512}"
SEQ_LEN_FINAL="${SEQ_LEN_FINAL:-1024}"

# DSA warmup: ~0.22% of training (DeepSeek V3.2 ratio: 2.1B/946B)
# 22M tokens / (2 * 32 * 512) = ~671 steps
DSA_WARMUP_STEPS="${DSA_WARMUP_STEPS:-700}"

BATCH_SIZE="${BATCH_SIZE:-24}"  # Divides by 12 to 2 when seq_len switches to 1024 (sparse kernel)
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"  # Multiplies by 8 to 8 when seq_len switches to 1024
EVAL_STEPS="${EVAL_STEPS:-200}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-10000}"
LR="${LR:-3e-4}"

# Number of GPUs
NUM_GPUS="${NUM_GPUS:-2}"

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

uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port=29501 src/train.py "${args[@]}" "$@"
