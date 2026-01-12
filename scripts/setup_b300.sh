#!/usr/bin/env bash
# Setup script for B300 GPU box (4x B300, 120 CPUs, 1.1TB RAM)
set -euo pipefail

echo "=== Cinnamon B300 Setup ==="
echo ""

# Check if we're on the right machine
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "Detected $GPU_COUNT GPUs"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# Sync dependencies
echo ""
echo "Installing dependencies..."
uv sync

# Add triton if not in pyproject.toml
echo ""
echo "Ensuring triton is installed..."
uv pip install triton

# Check CUDA
echo ""
echo "Checking CUDA..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Data preparation
echo ""
echo "=== Data Preparation ==="
DATA_DIR="./artifacts/tokenized_data"

if [[ -d "$DATA_DIR" ]] && ls "$DATA_DIR"/train_*.npy &>/dev/null; then
    SHARD_COUNT=$(ls "$DATA_DIR"/train_*.npy | wc -l)
    echo "Found $SHARD_COUNT existing train shards in $DATA_DIR"
    echo "To re-download, delete $DATA_DIR and re-run this script"
else
    echo "No tokenized data found. Starting download and tokenization..."
    echo "This will download ~280GB and tokenize to ~100GB"
    echo ""

    # Use 32 workers on B300 (has 120 CPUs)
    uv run python src/preprocess.py \
        --sample sample-100BT \
        --num_proc 64 \
        --max_tokens 30000000000 \
        --output_dir "$DATA_DIR"
fi

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training:"
echo "  # Run both RoPE and PoPE in parallel (recommended):"
echo "  ./scripts/launch_train_all.sh"
echo ""
echo "  # Or run individually:"
echo "  CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch_train_rope.sh"
echo "  CUDA_VISIBLE_DEVICES=2,3 ./scripts/launch_train_pope.sh"
echo ""
echo "  # Resume from checkpoint:"
echo "  ./scripts/launch_train_rope.sh --resume artifacts/checkpoints/rope-200M/checkpoint_1000.pt"
echo ""
echo "To monitor:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f wandb/latest-run/logs/debug.log"
