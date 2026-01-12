#!/usr/bin/env bash
# Launch RoPE and PoPE training in parallel on 4x B300 GPUs
# RoPE: GPUs 0,1 (port 29500)
# PoPE: GPUs 2,3 (port 29501)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting RoPE vs PoPE comparison training..."
echo "  RoPE: GPUs 0,1 (master_port 29500)"
echo "  PoPE: GPUs 2,3 (master_port 29501)"
echo ""

# Launch both in parallel, wait for both to complete
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29500 "$SCRIPT_DIR/launch_train_rope.sh" "$@" &
ROPE_PID=$!

CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29501 "$SCRIPT_DIR/launch_train_pope.sh" "$@" &
POPE_PID=$!

echo "RoPE PID: $ROPE_PID"
echo "PoPE PID: $POPE_PID"

# Wait for both and capture exit codes
ROPE_EXIT=0
POPE_EXIT=0
wait $ROPE_PID || ROPE_EXIT=$?
wait $POPE_PID || POPE_EXIT=$?

echo ""
echo "Training completed:"
echo "  RoPE exit code: $ROPE_EXIT"
echo "  PoPE exit code: $POPE_EXIT"

# Exit with error if either failed
if [[ $ROPE_EXIT -ne 0 ]] || [[ $POPE_EXIT -ne 0 ]]; then
  exit 1
fi
