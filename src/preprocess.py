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
    parser.add_argument("--shard_size", type=int, default=100000000)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (this will download ~28GB)...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")

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

    for example in tqdm(dataset, desc="Writing shards"):
        tokens = example["tokens"]
        split = example["split"]

        if split == "train":
            train_tokens.extend(tokens)
            while len(train_tokens) >= args.shard_size:
                save_shard(train_tokens[:args.shard_size], args.output_dir / f"train_{train_shard_idx:03d}.npy")
                train_tokens = train_tokens[args.shard_size:]
                train_shard_idx += 1
        else:
            val_tokens.extend(tokens)
            while len(val_tokens) >= args.shard_size:
                save_shard(val_tokens[:args.shard_size], args.output_dir / f"val_{val_shard_idx:03d}.npy")
                val_tokens = val_tokens[args.shard_size:]
                val_shard_idx += 1

    # Save remaining tokens
    if train_tokens:
        save_shard(train_tokens, args.output_dir / f"train_{train_shard_idx:03d}.npy")
        print(f"Saved final train shard with {len(train_tokens)} tokens")
    if val_tokens:
        save_shard(val_tokens, args.output_dir / f"val_{val_shard_idx:03d}.npy")
        print(f"Saved final val shard with {len(val_tokens)} tokens")

    print(f"Done! {train_shard_idx + 1} train shards, {val_shard_idx + 1} val shards")

if __name__ == "__main__":
    main()
