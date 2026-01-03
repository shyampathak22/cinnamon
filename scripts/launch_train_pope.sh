#!/usr/bin/env bash
set -euo pipefail

PROJECT="${WANDB_PROJECT:-cinnamon}"
RUN_PREFIX="${RUN_PREFIX:-}"
BASE_NAME="pope"
if [[ -n "$RUN_PREFIX" ]]; then
  RUN_NAME="${RUN_PREFIX}-${BASE_NAME}"
else
  RUN_NAME="${RUN_NAME:-$BASE_NAME}"
fi

MAX_TOKENS="${MAX_TOKENS:-100000000}"
SEQ_LEN="${SEQ_LEN:-1024}"
SEQ_LEN_FINAL="${SEQ_LEN_FINAL:-2048}"
DSA_WARMUP_STEPS="${DSA_WARMUP_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-100}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
LR="${LR:-3e-4}"

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

uv run torchrun --nproc_per_node=2 src/train.py "${args[@]}" "$@"
