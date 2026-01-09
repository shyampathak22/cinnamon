"""
Prepare INTELLECT-3-SFT data for 200M PoPE model.

Downloads and subsamples the dataset maintaining original ratios.
Stage 1: General reasoning (math, code, science, tool, chat, IF)
Stage 2: Agentic (Stage 1 + swe_swiss, toucan_tool)
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
import tiktoken
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Target sizes for 200M model (scaled down from INTELLECT-3's 100B+)
STAGE1_TOTAL = 90_000  # ~90K examples for general reasoning
STAGE2_ADDITIONAL = 30_000  # ~30K additional for agentic stage

# Original dataset sizes from HuggingFace
STAGE1_SOURCES = {
    "openreasoning_math": 2_040_000,
    "openreasoning_code": 1_900_000,
    "openreasoning_science": 1_600_000,
    "openreasoning_tool": 310_000,
    "am_chat": 952_000,
    "am_if": 54_700,
}

STAGE2_ONLY_SOURCES = {
    "swe_swiss": 10_300,
    "toucan_tool": 116_000,
}

def calculate_proportional_samples(sources: dict, total_target: int) -> dict:
    """Calculate number of samples per source maintaining original ratios."""
    total_original = sum(sources.values())
    samples = {}
    for name, count in sources.items():
        ratio = count / total_original
        samples[name] = max(1, int(total_target * ratio))

    # Adjust to hit exact target
    diff = total_target - sum(samples.values())
    if diff != 0:
        # Add/remove from largest source
        largest = max(samples, key=samples.get)
        samples[largest] += diff

    return samples


def format_and_filter_example(example: dict, max_tokens: int, source_name: str) -> dict | None:
    """Format example and return None if too long. Used for parallel processing."""
    # Lazy init tokenizer per process
    if not hasattr(format_and_filter_example, '_tokenizer'):
        format_and_filter_example._tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = format_and_filter_example._tokenizer

    messages = []

    # Process prompt messages
    for msg in example.get("prompt", []):
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })
        if msg.get("tool_calls"):
            messages[-1]["tool_calls"] = msg["tool_calls"]

    # Process completion messages
    for msg in example.get("completion", []):
        messages.append({
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
        })
        if msg.get("tool_calls"):
            messages[-1]["tool_calls"] = msg["tool_calls"]

    # Build text representation for tokenization
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"<|{role}|>\n{content}")

    full_text = "\n".join(text_parts) + "<|end|>"

    # Count tokens (disable special token check - data may contain them as literals)
    tokens = tokenizer.encode(full_text, disallowed_special=())
    num_tokens = len(tokens)

    # Filter by length
    if num_tokens > max_tokens:
        return None

    return {
        "messages": messages,
        "text": full_text,
        "num_tokens": num_tokens,
        "source": example.get("source", "unknown"),
        "subset": source_name,
    }


def process_batch(batch: list[dict], max_tokens: int, source_name: str) -> list[dict]:
    """Process a batch of examples, filtering by length."""
    results = []
    for example in batch:
        result = format_and_filter_example(example, max_tokens, source_name)
        if result is not None:
            results.append(result)
    return results


def sample_from_dataset(
    source_name: str,
    target_count: int,
    max_tokens: int,
    seed: int,
    oversample_factor: float = 20.0,  # Higher for long-form reasoning datasets
) -> list[dict]:
    """
    Efficiently sample from a dataset by:
    1. Taking a random subset (oversample_factor * target)
    2. Processing in parallel
    3. Taking first target_count that pass the filter
    """
    print(f"\nLoading {source_name}...")

    try:
        ds = load_dataset(
            "PrimeIntellect/INTELLECT-3-SFT",
            name=source_name,
            split="train",
        )
    except Exception as e:
        print(f"  ERROR loading {source_name}: {e}")
        return []

    dataset_size = len(ds)
    # Sample more than we need to account for filtering
    sample_size = min(int(target_count * oversample_factor), dataset_size)

    print(f"  Dataset size: {dataset_size:,}, sampling {sample_size:,} candidates...")

    # Use HF's efficient shuffle and select
    ds_sample = ds.shuffle(seed=seed).select(range(sample_size))

    # Process in parallel batches
    batch_size = 1000
    n_workers = min(mp.cpu_count(), 8)

    results = []
    processed = 0
    skipped = 0

    # Process in chunks
    for start_idx in range(0, sample_size, batch_size * n_workers):
        if len(results) >= target_count:
            break

        # Get batch of batches for parallel processing
        end_idx = min(start_idx + batch_size * n_workers, sample_size)
        chunk = [ds_sample[i] for i in range(start_idx, end_idx)]

        # Split into batches for workers
        batches = [chunk[i:i+batch_size] for i in range(0, len(chunk), batch_size)]

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_batch, batch, max_tokens, source_name)
                for batch in batches
            ]

            for future in as_completed(futures):
                batch_results = future.result()
                for item in batch_results:
                    if len(results) < target_count:
                        results.append(item)
                processed += batch_size

        # Progress update
        if processed % 10000 == 0 or len(results) >= target_count:
            print(f"  Processed {processed:,}, collected {len(results):,}/{target_count:,}")

    skipped = processed - len(results)
    print(f"  Collected {len(results):,} examples (skipped {skipped:,} > {max_tokens} tokens)")

    # If we didn't get enough, warn
    if len(results) < target_count:
        print(f"  WARNING: Only got {len(results):,}/{target_count:,} examples!")

    return results[:target_count]


def download_and_subsample(
    output_dir: Path,
    stage1_total: int = STAGE1_TOTAL,
    stage2_additional: int = STAGE2_ADDITIONAL,
    max_tokens: int = 8192,  # OpenReasoning has long CoT, need higher limit
    seed: int = 42,
):
    """Download dataset and create subsampled versions."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate proportional samples
    stage1_samples = calculate_proportional_samples(STAGE1_SOURCES, stage1_total)
    stage2_samples = calculate_proportional_samples(STAGE2_ONLY_SOURCES, stage2_additional)

    print("=" * 60)
    print("STAGE 1 SAMPLING PLAN (General Reasoning)")
    print("=" * 60)
    for name, count in stage1_samples.items():
        orig = STAGE1_SOURCES[name]
        print(f"  {name:25s}: {count:>6,} / {orig:>10,} ({count/orig*100:.2f}%)")
    print(f"  {'TOTAL':25s}: {sum(stage1_samples.values()):>6,}")

    print()
    print("=" * 60)
    print("STAGE 2 ADDITIONAL (Agentic)")
    print("=" * 60)
    for name, count in stage2_samples.items():
        orig = STAGE2_ONLY_SOURCES[name]
        print(f"  {name:25s}: {count:>6,} / {orig:>10,} ({count/orig*100:.2f}%)")
    print(f"  {'TOTAL':25s}: {sum(stage2_samples.values()):>6,}")

    print()
    print("=" * 60)
    print("DOWNLOADING AND PROCESSING...")
    print("=" * 60)

    stage1_data = []
    stage2_data = []

    # Process Stage 1 sources
    for source_name, target_count in stage1_samples.items():
        results = sample_from_dataset(source_name, target_count, max_tokens, seed)
        stage1_data.extend(results)

    # Process Stage 2 additional sources
    for source_name, target_count in stage2_samples.items():
        results = sample_from_dataset(source_name, target_count, max_tokens, seed)
        stage2_data.extend(results)

    # Shuffle final datasets
    random.shuffle(stage1_data)
    random.shuffle(stage2_data)

    # Save datasets
    stage1_path = output_dir / "sft_stage1.jsonl"
    stage2_path = output_dir / "sft_stage2.jsonl"

    print(f"\nSaving Stage 1 to {stage1_path}...")
    with open(stage1_path, "w") as f:
        for item in stage1_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saving Stage 2 to {stage2_path}...")
    with open(stage2_path, "w") as f:
        for item in stage2_data:
            f.write(json.dumps(item) + "\n")

    # Print statistics
    print()
    print("=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    if stage1_data:
        stage1_tokens = sum(d["num_tokens"] for d in stage1_data)
        print(f"Stage 1: {len(stage1_data):,} examples, {stage1_tokens:,} tokens")
        print(f"         Avg tokens/example: {stage1_tokens/len(stage1_data):.1f}")

    if stage2_data:
        stage2_tokens = sum(d["num_tokens"] for d in stage2_data)
        print(f"Stage 2: {len(stage2_data):,} examples, {stage2_tokens:,} tokens")
        print(f"         Avg tokens/example: {stage2_tokens/len(stage2_data):.1f}")

    print(f"Total:   {len(stage1_data) + len(stage2_data):,} examples")

    # Breakdown by source
    print()
    print("Stage 1 breakdown:")
    from collections import Counter
    s1_counts = Counter(d["subset"] for d in stage1_data)
    for name, count in sorted(s1_counts.items()):
        print(f"  {name:25s}: {count:>6,}")

    print()
    print("Stage 2 breakdown:")
    s2_counts = Counter(d["subset"] for d in stage2_data)
    for name, count in sorted(s2_counts.items()):
        print(f"  {name:25s}: {count:>6,}")

    return stage1_path, stage2_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SFT data")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sft_data"))
    parser.add_argument("--stage1-total", type=int, default=STAGE1_TOTAL)
    parser.add_argument("--stage2-additional", type=int, default=STAGE2_ADDITIONAL)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    download_and_subsample(
        output_dir=args.output_dir,
        stage1_total=args.stage1_total,
        stage2_additional=args.stage2_additional,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
