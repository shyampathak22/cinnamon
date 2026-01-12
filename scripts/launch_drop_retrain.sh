#!/usr/bin/env bash
# DroPE recalibration: Load pretrained checkpoint, drop positional embeddings, retrain 10%
# Usage: ./launch_drop_retrain.sh <rope|pope> <checkpoint_path>
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <rope|pope> <checkpoint_path>"
    echo "  rope: Load RoPE checkpoint, drop PE, recalibrate -> DropRoPE"
    echo "  pope: Load PoPE checkpoint, drop PE, recalibrate -> DropPoPE"
    exit 1
fi

SOURCE_TYPE="$1"
CHECKPOINT="$2"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

PROJECT="${WANDB_PROJECT:-cinnamon}"
RUN_PREFIX="${RUN_PREFIX:-}"

if [[ "$SOURCE_TYPE" == "rope" ]]; then
    BASE_NAME="drop-rope-200M"
    # RoPE used d_rope=128, need to match for loading weights
    D_ROPE=128
elif [[ "$SOURCE_TYPE" == "pope" ]]; then
    BASE_NAME="drop-pope-200M"
    # PoPE used d_rope=64
    D_ROPE=64
else
    echo "Error: First argument must be 'rope' or 'pope'"
    exit 1
fi

if [[ -n "$RUN_PREFIX" ]]; then
    RUN_NAME="${RUN_PREFIX}-${BASE_NAME}"
else
    RUN_NAME="${RUN_NAME:-$BASE_NAME}"
fi

# 10% of 25B = 2.5B tokens for recalibration
MAX_TOKENS="${MAX_TOKENS:-2500000000}"

# Same seq_len schedule as pretraining
SEQ_LEN="${SEQ_LEN:-512}"
SEQ_LEN_FINAL="${SEQ_LEN_FINAL:-1024}"

# Shorter DSA warmup for recalibration (10% of original)
DSA_WARMUP_STEPS="${DSA_WARMUP_STEPS:-1000}"

BATCH_SIZE="${BATCH_SIZE:-24}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-200}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-200}"
LR="${LR:-1e-4}"  # Lower LR for recalibration

NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "=== DroPE Recalibration ==="
echo "Source: $SOURCE_TYPE checkpoint"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $RUN_NAME"
echo "Tokens: ${MAX_TOKENS} (10% of pretrain)"
echo ""

args=(
    --none  # DroPE: no positional embeddings
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
    --d-rope "$D_ROPE"
    --resume "$CHECKPOINT"
    --load-weights-only  # Load weights but reset optimizer (since architecture changed)
)

if [[ -n "$SEQ_LEN_FINAL" ]]; then
    args+=(--seq-len-final "$SEQ_LEN_FINAL" --max-seq-len "$SEQ_LEN_FINAL")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    src/train.py "${args[@]}"
