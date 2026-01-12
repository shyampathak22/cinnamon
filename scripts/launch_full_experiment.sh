#!/usr/bin/env bash
# Full RoPE vs PoPE vs DroPE experiment pipeline
#
# Phase 1: Pretrain RoPE and PoPE in parallel (25B tokens each)
# Phase 2: Drop-retrain both models in parallel (2.5B tokens each)
# Phase 3: Evaluate all 4 models in parallel
#
# Total: ~23 hours on 4x B300s
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WANDB_PROJECT="${WANDB_PROJECT:-cinnamon}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/artifacts/checkpoints}"

echo "========================================"
echo "  Cinnamon: RoPE vs PoPE vs DroPE"
echo "========================================"
echo ""
echo "Phase 1: Pretrain (25B tokens) ~20 hrs"
echo "  - RoPE on GPUs 0,1"
echo "  - PoPE on GPUs 2,3"
echo ""
echo "Phase 2: Drop-retrain (2.5B tokens) ~2 hrs"
echo "  - DropRoPE on GPU 0"
echo "  - DropPoPE on GPU 1"
echo ""
echo "Phase 3: Evaluate ~1 hr"
echo "  - All 4 models in parallel"
echo ""
echo "========================================"
echo ""

# ============================================
# PHASE 1: PRETRAIN
# ============================================
echo "=== PHASE 1: PRETRAINING ==="
echo ""

CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 "$SCRIPT_DIR/launch_train_rope.sh" &
ROPE_PID=$!
echo "Started RoPE training (PID: $ROPE_PID)"

CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 "$SCRIPT_DIR/launch_train_pope.sh" &
POPE_PID=$!
echo "Started PoPE training (PID: $POPE_PID)"

echo ""
echo "Waiting for pretraining to complete..."
wait $ROPE_PID || { echo "RoPE training failed!"; exit 1; }
echo "RoPE training complete!"

wait $POPE_PID || { echo "PoPE training failed!"; exit 1; }
echo "PoPE training complete!"

# Find the latest checkpoints
ROPE_CKPT=$(ls -t "$CKPT_DIR"/rope-200M/checkpoint_*.pt 2>/dev/null | head -1)
POPE_CKPT=$(ls -t "$CKPT_DIR"/pope-200M/checkpoint_*.pt 2>/dev/null | head -1)

if [[ -z "$ROPE_CKPT" || -z "$POPE_CKPT" ]]; then
    echo "Error: Could not find pretrained checkpoints!"
    echo "  RoPE: $ROPE_CKPT"
    echo "  PoPE: $POPE_CKPT"
    exit 1
fi

echo ""
echo "Pretrained checkpoints:"
echo "  RoPE: $ROPE_CKPT"
echo "  PoPE: $POPE_CKPT"
echo ""

# ============================================
# PHASE 2: DROP-RETRAIN
# ============================================
echo "=== PHASE 2: DROP-RETRAIN (DroPE) ==="
echo ""

CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MASTER_PORT=29500 \
    "$SCRIPT_DIR/launch_drop_retrain.sh" rope "$ROPE_CKPT" &
DROP_ROPE_PID=$!
echo "Started DropRoPE training (PID: $DROP_ROPE_PID)"

CUDA_VISIBLE_DEVICES=1 NUM_GPUS=1 MASTER_PORT=29501 \
    "$SCRIPT_DIR/launch_drop_retrain.sh" pope "$POPE_CKPT" &
DROP_POPE_PID=$!
echo "Started DropPoPE training (PID: $DROP_POPE_PID)"

echo ""
echo "Waiting for drop-retrain to complete..."
wait $DROP_ROPE_PID || { echo "DropRoPE training failed!"; exit 1; }
echo "DropRoPE training complete!"

wait $DROP_POPE_PID || { echo "DropPoPE training failed!"; exit 1; }
echo "DropPoPE training complete!"

# Find drop checkpoints
DROP_ROPE_CKPT=$(ls -t "$CKPT_DIR"/drop-rope-200M/checkpoint_*.pt 2>/dev/null | head -1)
DROP_POPE_CKPT=$(ls -t "$CKPT_DIR"/drop-pope-200M/checkpoint_*.pt 2>/dev/null | head -1)

echo ""
echo "Drop-retrained checkpoints:"
echo "  DropRoPE: $DROP_ROPE_CKPT"
echo "  DropPoPE: $DROP_POPE_CKPT"
echo ""

# ============================================
# PHASE 3: EVALUATION
# ============================================
echo "=== PHASE 3: EVALUATION ==="
echo ""

export ROPE_CKPT POPE_CKPT DROP_ROPE_CKPT DROP_POPE_CKPT
export WANDB_PROJECT="${WANDB_PROJECT}-eval"

"$SCRIPT_DIR/launch_eval_all.sh"

# ============================================
# COMPARISON
# ============================================
echo ""
echo "=== GENERATING COMPARISON PLOTS ==="
uv run python "$PROJECT_DIR/src/eval_compare.py" \
    --results-dir "$PROJECT_DIR/artifacts/eval_results" \
    --wandb-project "${WANDB_PROJECT}-eval"

echo ""
echo "========================================"
echo "  EXPERIMENT COMPLETE!"
echo "========================================"
echo ""
echo "Results:"
echo "  Checkpoints: $CKPT_DIR"
echo "  Eval results: $PROJECT_DIR/artifacts/eval_results"
echo "  Wandb: https://wandb.ai/your-org/$WANDB_PROJECT"
echo ""
