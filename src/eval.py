"""
Unified evaluation script for length extrapolation testing.

Infers config from checkpoint directory name:
  - rope/     -> rope_type=rope, rope_factor=1.0
  - rope-yarn/ -> rope_type=rope, rope_factor=8.0
  - pope/     -> rope_type=pope, rope_factor=1.0

Usage:
    python src/eval.py artifacts/checkpoints/rope/checkpoint_1500.pt
    python src/eval.py artifacts/checkpoints/*/checkpoint_1500.pt  # all three
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import tiktoken

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import ModelConfig
from model import Cinnamon

DATA_DIR = Path(__file__).parent.parent / "artifacts" / "tokenized_data"
_DROP_BUFFER_SUFFIXES = (".cos_cached", ".sin_cached", ".base_angles", ".freqs")

# Config mapping from directory name
CONFIGS = {
    "rope": {"rope_type": "rope", "rope_factor": 1.0},
    "rope-yarn": {"rope_type": "rope", "rope_factor": 8.0},
    "pope": {"rope_type": "pope", "rope_factor": 1.0},
}

TRAINING_SEQ_LEN = 2048  # All models trained at this final length
ORIGINAL_SEQ_LEN = 1024  # YaRN reference length


def infer_config(checkpoint_path: Path) -> dict:
    """Infer rope_type and rope_factor from checkpoint directory name."""
    dir_name = checkpoint_path.parent.name
    if dir_name not in CONFIGS:
        raise ValueError(f"Unknown checkpoint directory: {dir_name}. Expected one of {list(CONFIGS.keys())}")
    return CONFIGS[dir_name]


def load_model(checkpoint_path: Path, max_seq_len: int, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint with proper config."""
    config = infer_config(checkpoint_path)

    cfg = ModelConfig()
    cfg.max_seq_len = max_seq_len
    cfg.original_seq_len = ORIGINAL_SEQ_LEN
    cfg.rope_type = config["rope_type"]
    cfg.rope_factor = config["rope_factor"]

    model = Cinnamon(
        cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.hidden_dim, cfg.n_heads, cfg.max_seq_len,
        cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_v, cfg.d_rope, cfg.n_routed, cfg.n_shared,
        cfg.top_k, cfg.expert_scale, cfg.gamma, 0.0, cfg.dsa_topk, cfg.local_window,
        cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps, cfg.rope_base, cfg.rope_type,
        cfg.mtp_depth, cfg.pope_delta_init, cfg.original_seq_len, cfg.rope_factor, cfg.beta_fast,
        cfg.beta_slow, cfg.mscale, cfg.indexer_use_fp8, cfg.indexer_use_hadamard
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Normalize keys and strip buffers
    normalized = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "").replace("module.", "")
        if not new_k.endswith(_DROP_BUFFER_SUFFIXES):
            normalized[new_k] = v

    model.load_state_dict(normalized, strict=False)
    model.to(device)
    model.eval()

    return model, config


def load_data(max_tokens: int = 5_000_000) -> np.ndarray:
    """Load validation data."""
    shards = sorted(DATA_DIR.glob("val_*.npy"))
    if not shards:
        raise FileNotFoundError(f"No validation shards in {DATA_DIR}")

    chunks = []
    total = 0
    for shard in shards:
        arr = np.load(shard, mmap_mode='r')
        chunks.append(np.array(arr))
        total += len(arr)
        if total >= max_tokens:
            break

    return np.concatenate(chunks)[:max_tokens]


def compute_perplexity(model, data: np.ndarray, seq_len: int, batch_size: int,
                       device: torch.device, max_batches: int = 50) -> float:
    """Compute perplexity at given sequence length."""
    n_seqs = len(data) // (seq_len + 1)
    n_seqs = min(n_seqs, max_batches * batch_size)

    if n_seqs < batch_size:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, n_seqs - batch_size + 1, batch_size):
            batch = []
            for j in range(batch_size):
                start = (i + j) * (seq_len + 1)
                batch.append(data[start:start + seq_len + 1])

            batch = torch.tensor(np.stack(batch), dtype=torch.long, device=device)
            x, y = batch[:, :-1], batch[:, 1:]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")

            total_loss += loss.item()
            total_tokens += y.numel()

            if (i // batch_size) >= max_batches:
                break

    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float("inf")


def evaluate_checkpoint(checkpoint_path: Path, lengths: list[int], data: np.ndarray,
                        batch_size: int, max_batches: int, device: torch.device) -> dict:
    """Evaluate one checkpoint at multiple lengths."""
    max_len = max(lengths)
    model, config = load_model(checkpoint_path, max_len, device)

    name = checkpoint_path.parent.name
    results = {"name": name, "config": config, "results": []}

    print(f"\n{name.upper()} (rope_type={config['rope_type']}, rope_factor={config['rope_factor']})")
    print("-" * 50)

    for seq_len in lengths:
        ppl = compute_perplexity(model, data, seq_len, batch_size, device, max_batches)
        status = "EXTRAPOLATION" if seq_len > TRAINING_SEQ_LEN else ""
        print(f"  {seq_len:>6} tokens: ppl = {ppl:>8.2f}  {status}")
        results["results"].append({"seq_len": seq_len, "perplexity": ppl})

    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate length extrapolation")
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Checkpoint path(s)")
    parser.add_argument("--lengths", type=str, default="1024,2048,4096,8192",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=5_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=None, help="Save results to JSON")
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    device = torch.device(args.device)

    print(f"Loading {args.max_tokens:,} validation tokens...")
    data = load_data(args.max_tokens)
    print(f"Evaluating at lengths: {lengths}")
    print(f"Training length was: {TRAINING_SEQ_LEN}")

    all_results = []
    for ckpt_path in args.checkpoints:
        results = evaluate_checkpoint(ckpt_path, lengths, data, args.batch_size, args.max_batches, device)
        all_results.append(results)

    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        header = f"{'Length':>8} |"
        for r in all_results:
            header += f" {r['name']:>10} |"
        print(header)
        print("-" * 70)

        for i, seq_len in enumerate(lengths):
            row = f"{seq_len:>8} |"
            ppls = [r["results"][i]["perplexity"] for r in all_results]
            best = min(ppls)
            for ppl in ppls:
                marker = "*" if ppl == best else " "
                row += f" {ppl:>9.2f}{marker}|"
            status = " <- EXTRAPOLATION" if seq_len > TRAINING_SEQ_LEN else ""
            print(row + status)

        print("=" * 70)
        print(f"* = best | Training length: {TRAINING_SEQ_LEN}")

        # Degradation analysis
        print("\nDegradation (2048 -> 8192):")
        for r in all_results:
            ppl_2k = next(x["perplexity"] for x in r["results"] if x["seq_len"] == 2048)
            ppl_8k = next(x["perplexity"] for x in r["results"] if x["seq_len"] == 8192)
            ratio = ppl_8k / ppl_2k
            print(f"  {r['name']:>10}: {ppl_2k:.1f} -> {ppl_8k:.1f} ({ratio:.2f}x)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
