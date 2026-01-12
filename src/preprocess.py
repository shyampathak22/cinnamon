import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
import hashlib
from tqdm import tqdm

def get_split(text, val_ratio=0.01):
    # hash text with md5
    h = hashlib.md5(text.encode()).hexdigest()
    # convert hex to int and mod 100 to get 0-99
    bucket = int(h, 16) % 100
    # if bucket is in val range, return val
    return "val" if bucket < val_ratio * 100 else "train"

def save_shard(tokens, path):
    arr = np.array(tokens, dtype=np.uint32)
    np.save(path, arr)

# Global encoder for multiprocessing (created per worker)
enc = None

def init_worker():
    global enc
    enc = tiktoken.get_encoding("gpt2")

def tokenize(example):
    global enc
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(example["text"], disallowed_special=())
    split = get_split(example["text"])
    return {"tokens": tokens, "split": split}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default="./artifacts/tokenized_data/")
    parser.add_argument("--shard_size", type=int, default=100000000)  # 100M tokens per shard
    parser.add_argument("--num_proc", type=int, default=32)  # B300 has 120 CPUs
    parser.add_argument("--sample", type=str, default="sample-100BT",
                        choices=["sample-10BT", "sample-100BT", "sample-350BT"],
                        help="FineWeb-Edu sample size (default: sample-100BT for 25B+ tokens)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Stop after processing this many tokens (for quick tests)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: HuggingFaceFW/fineweb-edu/{args.sample}")
    print("  (sample-10BT: ~28GB, sample-100BT: ~280GB, sample-350BT: ~980GB)")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", args.sample, split="train")

    print(f"Tokenizing with {args.num_proc} workers...")
    dataset = dataset.map(
        tokenize,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    print("Writing shards...")
    train_tokens = []
    train_shard_idx = 0
    val_tokens = []
    val_shard_idx = 0
    total_train_tokens = 0
    total_val_tokens = 0

    pbar = tqdm(dataset, desc="Writing shards")
    for example in pbar:
        tokens = example["tokens"]
        split = example["split"]

        if split == "train":
            train_tokens.extend(tokens)
            while len(train_tokens) >= args.shard_size:
                save_shard(train_tokens[:args.shard_size], args.output_dir / f"train_{train_shard_idx:03d}.npy")
                total_train_tokens += args.shard_size
                train_tokens = train_tokens[args.shard_size:]
                train_shard_idx += 1
                pbar.set_postfix(train=f"{total_train_tokens/1e9:.2f}B", val=f"{total_val_tokens/1e9:.2f}B")
        else:
            val_tokens.extend(tokens)
            while len(val_tokens) >= args.shard_size:
                save_shard(val_tokens[:args.shard_size], args.output_dir / f"val_{val_shard_idx:03d}.npy")
                total_val_tokens += args.shard_size
                val_tokens = val_tokens[args.shard_size:]
                val_shard_idx += 1
                pbar.set_postfix(train=f"{total_train_tokens/1e9:.2f}B", val=f"{total_val_tokens/1e9:.2f}B")

        # Early exit if max_tokens reached
        if args.max_tokens and (total_train_tokens + total_val_tokens) >= args.max_tokens:
            print(f"\nReached {args.max_tokens/1e9:.1f}B tokens limit, stopping early")
            break
    pbar.close()

    # Save remaining tokens
    if train_tokens:
        save_shard(train_tokens, args.output_dir / f"train_{train_shard_idx:03d}.npy")
        total_train_tokens += len(train_tokens)
        print(f"Saved final train shard with {len(train_tokens):,} tokens")
    if val_tokens:
        save_shard(val_tokens, args.output_dir / f"val_{val_shard_idx:03d}.npy")
        total_val_tokens += len(val_tokens)
        print(f"Saved final val shard with {len(val_tokens):,} tokens")

    print(f"\nDone! Total: {(total_train_tokens + total_val_tokens)/1e9:.2f}B tokens")
    print(f"  Train: {total_train_tokens/1e9:.2f}B ({train_shard_idx + (1 if train_tokens else 0)} shards)")
    print(f"  Val:   {total_val_tokens/1e9:.2f}B ({val_shard_idx + (1 if val_tokens else 0)} shards)")

if __name__ == "__main__":
    main()
