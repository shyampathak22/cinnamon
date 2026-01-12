#!/usr/bin/env bash
# Run all 4 model evaluations in parallel (1 per GPU)
# Evaluates: RoPE, PoPE, DropRoPE, DropPoPE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Checkpoint directory
CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/artifacts/checkpoints}"

# Find latest checkpoints (or use provided paths)
ROPE_CKPT="${ROPE_CKPT:-$(ls -t "$CKPT_DIR"/rope-200M/checkpoint_*.pt 2>/dev/null | head -1 || echo '')}"
POPE_CKPT="${POPE_CKPT:-$(ls -t "$CKPT_DIR"/pope-200M/checkpoint_*.pt 2>/dev/null | head -1 || echo '')}"
DROP_ROPE_CKPT="${DROP_ROPE_CKPT:-$(ls -t "$CKPT_DIR"/drop-rope-200M/checkpoint_*.pt 2>/dev/null | head -1 || echo '')}"
DROP_POPE_CKPT="${DROP_POPE_CKPT:-$(ls -t "$CKPT_DIR"/drop-pope-200M/checkpoint_*.pt 2>/dev/null | head -1 || echo '')}"

WANDB_PROJECT="${WANDB_PROJECT:-cinnamon-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/eval_results}"

# Context lengths to evaluate (can be overridden)
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-512,1024,2048,4096,8192,16384}"

echo "=== Cinnamon Model Evaluation ==="
echo ""
echo "Checkpoints:"
echo "  RoPE:     ${ROPE_CKPT:-NOT FOUND}"
echo "  PoPE:     ${POPE_CKPT:-NOT FOUND}"
echo "  DropRoPE: ${DROP_ROPE_CKPT:-NOT FOUND}"
echo "  DropPoPE: ${DROP_POPE_CKPT:-NOT FOUND}"
echo ""
echo "Context lengths: $CONTEXT_LENGTHS"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Function to run eval for a single model
run_eval() {
    local gpu="$1"
    local ckpt="$2"
    local rope_type="$3"
    local d_rope="$4"
    local model_name="$5"

    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "[$model_name] Checkpoint not found, skipping"
        return 0
    fi

    echo "[$model_name] Starting evaluation on GPU $gpu..."

    # PG19 Perplexity
    CUDA_VISIBLE_DEVICES="$gpu" uv run python "$PROJECT_DIR/src/eval_pg19.py" \
        --checkpoint "$ckpt" \
        --rope-type "$rope_type" \
        --d-rope "$d_rope" \
        --model-name "$model_name" \
        --wandb-project "$WANDB_PROJECT" \
        --output-dir "$OUTPUT_DIR" \
        --context-lengths "$CONTEXT_LENGTHS" \
        --device cuda:0

    # NIAH
    CUDA_VISIBLE_DEVICES="$gpu" uv run python "$PROJECT_DIR/src/eval_niah.py" \
        --checkpoint "$ckpt" \
        --rope-type "$rope_type" \
        --d-rope "$d_rope" \
        --model-name "$model_name" \
        --wandb-project "$WANDB_PROJECT" \
        --output-dir "$OUTPUT_DIR" \
        --context-lengths "$CONTEXT_LENGTHS" \
        --device cuda:0

    echo "[$model_name] Evaluation complete!"
}

# Launch all 4 evaluations in parallel
run_eval 0 "$ROPE_CKPT" rope 128 "RoPE" &
PID_ROPE=$!

run_eval 1 "$POPE_CKPT" pope 64 "PoPE" &
PID_POPE=$!

run_eval 2 "$DROP_ROPE_CKPT" none 128 "DropRoPE" &
PID_DROP_ROPE=$!

run_eval 3 "$DROP_POPE_CKPT" none 64 "DropPoPE" &
PID_DROP_POPE=$!

echo ""
echo "Evaluation PIDs:"
echo "  RoPE:     $PID_ROPE"
echo "  PoPE:     $PID_POPE"
echo "  DropRoPE: $PID_DROP_ROPE"
echo "  DropPoPE: $PID_DROP_POPE"
echo ""
echo "Waiting for all evaluations to complete..."

# Wait for all and capture exit codes
wait $PID_ROPE || echo "RoPE eval failed"
wait $PID_POPE || echo "PoPE eval failed"
wait $PID_DROP_ROPE || echo "DropRoPE eval failed"
wait $PID_DROP_POPE || echo "DropPoPE eval failed"

echo ""
echo "=== All evaluations complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To generate comparison plots:"
echo "  uv run python src/eval_compare.py --results-dir $OUTPUT_DIR"
