import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
import hashlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import queue
import threading

def get_split(text, val_ratio=0.01):
    h = hashlib.md5(text.encode()).hexdigest()
    bucket = int(h, 16) % 100
    return "val" if bucket < val_ratio * 100 else "train"

def save_shard(tokens, path):
    arr = np.array(tokens, dtype=np.uint32)
    np.save(path, arr)

def tokenize_batch(texts):
    """Tokenize a batch of texts - called in worker processes."""
    enc = tiktoken.get_encoding("gpt2")
    results = []
    for text in texts:
        tokens = enc.encode(text, disallowed_special=())
        split = get_split(text)
        results.append((tokens, split))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default="./artifacts/tokenized_data/")
    parser.add_argument("--shard_size", type=int, default=100_000_000)  # 100M tokens per shard
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--sample", type=str, default="sample-100BT",
                        choices=["sample-10BT", "sample-100BT", "sample-350BT"])
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Stop after this many tokens (default: process all)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for parallel tokenization")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming dataset: HuggingFaceFW/fineweb-edu/{args.sample}")
    print("  (streaming mode - no intermediate disk cache)")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        args.sample,
        split="train",
        streaming=True
    )

    train_tokens = []
    train_shard_idx = 0
    val_tokens = []
    val_shard_idx = 0
    total_train_tokens = 0
    total_val_tokens = 0

    batch_texts = []

    print(f"Processing with {args.num_proc} workers (batch_size={args.batch_size})...")

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        pbar = tqdm(dataset, desc="Tokenizing", unit=" docs")

        for example in pbar:
            batch_texts.append(example["text"])

            # Process batch when full
            if len(batch_texts) >= args.batch_size:
                # Submit batch to worker pool
                future = executor.submit(tokenize_batch, batch_texts)
                results = future.result()

                for tokens, split in results:
                    if split == "train":
                        train_tokens.extend(tokens)
                        while len(train_tokens) >= args.shard_size:
                            save_shard(train_tokens[:args.shard_size],
                                      args.output_dir / f"train_{train_shard_idx:03d}.npy")
                            total_train_tokens += args.shard_size
                            train_tokens = train_tokens[args.shard_size:]
                            train_shard_idx += 1
                    else:
                        val_tokens.extend(tokens)
                        while len(val_tokens) >= args.shard_size:
                            save_shard(val_tokens[:args.shard_size],
                                      args.output_dir / f"val_{val_shard_idx:03d}.npy")
                            total_val_tokens += args.shard_size
                            val_tokens = val_tokens[args.shard_size:]
                            val_shard_idx += 1

                batch_texts = []
                pbar.set_postfix(train=f"{total_train_tokens/1e9:.2f}B",
                               val=f"{total_val_tokens/1e9:.2f}B")

                # Early exit if max_tokens reached
                if args.max_tokens and (total_train_tokens + total_val_tokens) >= args.max_tokens:
                    print(f"\nReached {args.max_tokens/1e9:.1f}B tokens limit")
                    break

        # Process remaining batch
        if batch_texts:
            future = executor.submit(tokenize_batch, batch_texts)
            results = future.result()
            for tokens, split in results:
                if split == "train":
                    train_tokens.extend(tokens)
                else:
                    val_tokens.extend(tokens)

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
