#!/usr/bin/env bash
set -euo pipefail

PROJECT="${WANDB_PROJECT:-cinnamon}"
RUN_PREFIX="${RUN_PREFIX:-}"
BASE_NAME="rope-d_rope64"
if [[ -n "$RUN_PREFIX" ]]; then
  RUN_NAME="${RUN_PREFIX}-${BASE_NAME}"
else
  RUN_NAME="${RUN_NAME:-$BASE_NAME}"
fi

MAX_TOKENS="${MAX_TOKENS:-250000000}"
SEQ_LEN="${SEQ_LEN:-512}"
SEQ_LEN_FINAL="${SEQ_LEN_FINAL:-1024}"
DSA_WARMUP_STEPS="${DSA_WARMUP_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-100}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
LR="${LR:-6e-4}"

args=(
  --rope
  --run-name "$RUN_NAME"
  --wandb-project "$PROJECT"
  --max-tokens "$MAX_TOKENS"
  --seq-len "$SEQ_LEN"
  --max-seq-len "$SEQ_LEN"
  --rope-factor 1.0
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

uv run torchrun --nproc_per_node=1 --master_port=29500 src/train.py "${args[@]}" "$@"
