"""
Passkey Retrieval Evaluation for Length Extrapolation

This benchmark tests a model's ability to retrieve a hidden passkey from
increasingly long contexts. It creates beautiful heatmaps comparing different
position encoding schemes (RoPE vs PoPE).

Based on: "Landmark Attention: Random-Access Infinite Context Length for Transformers"
https://arxiv.org/abs/2305.16300
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import random
import re
import argparse
from tqdm import tqdm
import json
import tiktoken

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import Cinnamon
from config import ModelConfig

# Constants
GARBAGE_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
TASK_DESCRIPTION = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there.\n\n"
PASSKEY_TEMPLATE = "The pass key is {passkey}. Remember it. {passkey} is the pass key."
QUESTION = "\n\nWhat is the pass key? The pass key is"


@dataclass
class PasskeyResult:
    """Result of a single passkey retrieval test."""
    context_length: int
    depth_percent: float
    passkey: str
    model_output: str
    is_correct: bool
    exact_match: bool
    tokens_used: int


def generate_passkey_prompt(
    tokenizer,
    target_length: int,
    depth_percent: float,
    seed: int = None
) -> tuple[str, str, int]:
    """
    Generate a passkey retrieval prompt at a specific context length and depth.

    Args:
        tokenizer: Tokenizer to use for length calculation
        target_length: Target context length in tokens
        depth_percent: Where to place the passkey (0.0 = start, 1.0 = end)
        seed: Random seed for reproducibility

    Returns:
        (prompt, passkey, actual_token_count)
    """
    if seed is not None:
        random.seed(seed)

    # Generate a random 5-digit passkey
    passkey = str(random.randint(10000, 99999))
    passkey_line = PASSKEY_TEMPLATE.format(passkey=passkey)

    # Calculate how much garbage text we need
    # Start with a rough estimate, then refine
    task_tokens = len(tokenizer.encode(TASK_DESCRIPTION))
    passkey_tokens = len(tokenizer.encode(passkey_line))
    question_tokens = len(tokenizer.encode(QUESTION))
    overhead_tokens = task_tokens + passkey_tokens + question_tokens + 10  # buffer

    garbage_tokens_needed = target_length - overhead_tokens
    if garbage_tokens_needed < 0:
        garbage_tokens_needed = 100  # minimum

    # Generate garbage text
    garbage_unit_tokens = len(tokenizer.encode(GARBAGE_TEXT))
    n_repeats = (garbage_tokens_needed // garbage_unit_tokens) + 1
    full_garbage = GARBAGE_TEXT * n_repeats

    # Tokenize and truncate to exact length
    garbage_tokens = tokenizer.encode(full_garbage)[:garbage_tokens_needed]
    full_garbage = tokenizer.decode(garbage_tokens)

    # Split garbage at the depth point
    split_point = int(len(full_garbage) * depth_percent)
    # Find a clean split point (at a sentence boundary)
    if split_point > 0:
        period_pos = full_garbage.rfind('. ', 0, split_point)
        if period_pos > 0:
            split_point = period_pos + 2

    garbage_prefix = full_garbage[:split_point]
    garbage_suffix = full_garbage[split_point:]

    # Construct the full prompt
    prompt = TASK_DESCRIPTION + garbage_prefix + passkey_line + garbage_suffix + QUESTION

    # Get actual token count
    actual_tokens = len(tokenizer.encode(prompt))

    return prompt, passkey, actual_tokens


def extract_passkey_from_output(output: str) -> str:
    """
    Extract the passkey from model output.
    Handles various output formats robustly.
    """
    # Clean the output
    output = output.strip()

    # Try to find a 5-digit number
    matches = re.findall(r'\b(\d{5})\b', output)
    if matches:
        return matches[0]

    # Try to find any sequence of 5 digits
    matches = re.findall(r'(\d{5})', output)
    if matches:
        return matches[0]

    # Return first 5 characters if they're all digits
    if len(output) >= 5 and output[:5].isdigit():
        return output[:5]

    return output[:10] if output else ""


@torch.no_grad()
def evaluate_passkey(
    model: Cinnamon,
    tokenizer,
    context_length: int,
    depth_percent: float,
    device: torch.device,
    seed: int = None,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> PasskeyResult:
    """
    Evaluate passkey retrieval at a specific context length and depth.
    """
    model.eval()

    # Generate prompt
    prompt, passkey, actual_tokens = generate_passkey_prompt(
        tokenizer, context_length, depth_percent, seed
    )

    # Tokenize
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    # Generate output autoregressively
    generated = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        # Forward pass (only need logits, not MTP)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(current_ids)

        # Get next token (greedy or with temperature)
        next_logits = logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=1)

        # Stop if we've generated enough digits
        decoded = tokenizer.decode(generated)
        if len(re.findall(r'\d', decoded)) >= 5:
            break

    # Decode output
    model_output = tokenizer.decode(generated)
    extracted = extract_passkey_from_output(model_output)

    # Check correctness
    exact_match = extracted == passkey
    # Also check if the passkey appears anywhere in the output
    is_correct = passkey in model_output or exact_match

    return PasskeyResult(
        context_length=actual_tokens,
        depth_percent=depth_percent,
        passkey=passkey,
        model_output=model_output,
        is_correct=is_correct,
        exact_match=exact_match,
        tokens_used=actual_tokens
    )


def run_passkey_evaluation(
    model: Cinnamon,
    tokenizer,
    device: torch.device,
    context_lengths: list[int],
    depth_percents: list[float],
    n_samples: int = 3,
    base_seed: int = 42,
) -> dict:
    """
    Run full passkey evaluation across context lengths and depths.

    Returns:
        Dictionary with results matrix and metadata
    """
    results = {
        'context_lengths': context_lengths,
        'depth_percents': depth_percents,
        'n_samples': n_samples,
        'accuracy_matrix': np.zeros((len(depth_percents), len(context_lengths))),
        'exact_match_matrix': np.zeros((len(depth_percents), len(context_lengths))),
        'detailed_results': []
    }

    total_tests = len(context_lengths) * len(depth_percents) * n_samples
    pbar = tqdm(total=total_tests, desc="Passkey Evaluation")

    for i, depth in enumerate(depth_percents):
        for j, ctx_len in enumerate(context_lengths):
            correct_count = 0
            exact_count = 0

            for sample in range(n_samples):
                seed = base_seed + i * 1000 + j * 100 + sample

                try:
                    result = evaluate_passkey(
                        model, tokenizer, ctx_len, depth, device, seed
                    )

                    if result.is_correct:
                        correct_count += 1
                    if result.exact_match:
                        exact_count += 1

                    results['detailed_results'].append({
                        'context_length': ctx_len,
                        'depth_percent': depth,
                        'sample': sample,
                        'passkey': result.passkey,
                        'output': result.model_output,
                        'is_correct': result.is_correct,
                        'exact_match': result.exact_match
                    })
                except Exception as e:
                    print(f"\nError at ctx={ctx_len}, depth={depth}: {e}")

                pbar.update(1)

            results['accuracy_matrix'][i, j] = correct_count / n_samples
            results['exact_match_matrix'][i, j] = exact_count / n_samples

    pbar.close()
    return results


def create_heatmap(
    results: dict,
    title: str,
    output_path: Path,
    training_length: int = 1024,
    use_exact_match: bool = True,
    figsize: tuple = (12, 8)
):
    """
    Create a beautiful heatmap visualization of passkey retrieval results.
    """
    matrix = results['exact_match_matrix'] if use_exact_match else results['accuracy_matrix']
    context_lengths = results['context_lengths']
    depth_percents = results['depth_percents']

    # Create figure with custom styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: Red (fail) -> Yellow (partial) -> Green (success)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = mcolors.LinearSegmentedColormap.from_list('passkey', colors, N=256)

    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Retrieval Accuracy', fontsize=12, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(len(context_lengths)))
    ax.set_xticklabels([f'{l//1000}K' if l >= 1000 else str(l) for l in context_lengths],
                       fontsize=11)
    ax.set_yticks(range(len(depth_percents)))
    ax.set_yticklabels([f'{int(d*100)}%' for d in depth_percents], fontsize=11)

    # Labels
    ax.set_xlabel('Context Length (tokens)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Passkey Depth (% into context)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # Add value annotations
    for i in range(len(depth_percents)):
        for j in range(len(context_lengths)):
            value = matrix[i, j]
            text_color = 'white' if value < 0.5 else 'black'
            ax.text(j, i, f'{value:.0%}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color=text_color)

    # Mark training length with a vertical line
    if training_length in context_lengths:
        train_idx = context_lengths.index(training_length)
        ax.axvline(x=train_idx + 0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(train_idx + 0.5, -0.7, 'Training\nLength', ha='center', va='top',
               fontsize=9, color='gray', style='italic')

    # Add extrapolation region shading
    if training_length in context_lengths:
        train_idx = context_lengths.index(training_length)
        rect = Rectangle((train_idx + 0.5, -0.5),
                         len(context_lengths) - train_idx - 0.5,
                         len(depth_percents),
                         linewidth=0, edgecolor='none',
                         facecolor='blue', alpha=0.1)
        ax.add_patch(rect)
        ax.text(len(context_lengths) - 0.5, len(depth_percents) + 0.3,
               'Extrapolation Region', ha='right', va='bottom',
               fontsize=9, color='blue', alpha=0.7, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def create_comparison_heatmap(
    results_dict: dict[str, dict],
    output_path: Path,
    training_length: int = 1024,
    figsize: tuple = (16, 6)
):
    """
    Create side-by-side comparison heatmaps for multiple models.
    """
    n_models = len(results_dict)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    # Custom colormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = mcolors.LinearSegmentedColormap.from_list('passkey', colors, N=256)

    for idx, (name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        matrix = results['exact_match_matrix']
        context_lengths = results['context_lengths']
        depth_percents = results['depth_percents']

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(range(len(context_lengths)))
        ax.set_xticklabels([f'{l//1000}K' if l >= 1000 else str(l) for l in context_lengths],
                          fontsize=10)
        ax.set_yticks(range(len(depth_percents)))
        ax.set_yticklabels([f'{int(d*100)}%' for d in depth_percents], fontsize=10)

        ax.set_xlabel('Context Length', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Passkey Depth', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=13, fontweight='bold')

        # Add value annotations
        for i in range(len(depth_percents)):
            for j in range(len(context_lengths)):
                value = matrix[i, j]
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.0%}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color)

        # Mark training length
        if training_length in context_lengths:
            train_idx = context_lengths.index(training_length)
            ax.axvline(x=train_idx + 0.5, color='white', linestyle='--',
                      linewidth=2, alpha=0.8)

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Exact Match Accuracy', fontsize=11, fontweight='bold')

    # Main title
    fig.suptitle('Passkey Retrieval: Length Extrapolation Comparison',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved comparison heatmap to {output_path}")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, max_seq_len: int = None) -> tuple[Cinnamon, ModelConfig]:
    """Load model from checkpoint, inferring config from directory name."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Infer rope_type from directory name
    dir_name = checkpoint_path.parent.name.lower()
    if 'pope' in dir_name:
        rope_type = 'pope'
    else:
        rope_type = 'rope'

    print(f"Loading {checkpoint_path.name} with rope_type={rope_type}")

    # Use default ModelConfig (matches the checkpoint architecture)
    config = ModelConfig()
    config.rope_type = rope_type

    # Override max_seq_len for longer context evaluation
    if max_seq_len is not None:
        config.max_seq_len = max_seq_len

    # Build model
    model = Cinnamon(
        config.d_model, config.n_layers, config.vocab_size, config.hidden_dim,
        config.n_heads, config.max_seq_len, config.d_ckv, config.d_cq,
        config.d_head, config.d_v, config.d_rope, config.n_routed, config.n_shared,
        config.top_k, config.expert_scale, config.gamma, 0.01, config.dsa_topk,
        config.local_window, config.n_indexer_heads, config.d_indexer_head,
        config.rms_eps, config.rope_base, config.rope_type, config.mtp_depth,
        config.pope_delta_init, config.original_seq_len, config.rope_factor,
        config.beta_fast, config.beta_slow, config.mscale,
        indexer_use_fp8=False, indexer_use_hadamard=config.indexer_use_hadamard
    )

    # Get state dict and clean up keys from torch.compile + DDP
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Buffers to drop (will be regenerated at correct size for max_seq_len)
    DROP_BUFFER_SUFFIXES = ('.cos_cached', '.sin_cached', '.base_angles', '.freqs')

    # Remove _orig_mod.module. prefix if present (from torch.compile + DDP)
    # and drop position encoding buffers (they'll be regenerated at the right size)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('_orig_mod.module.'):
            new_key = new_key[len('_orig_mod.module.'):]
        elif new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
        elif new_key.startswith('module.'):
            new_key = new_key[len('module.'):]

        # Drop position encoding buffers (they're recreated at the correct size)
        if not new_key.endswith(DROP_BUFFER_SUFFIXES):
            cleaned_state_dict[new_key] = value

    # Load weights
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (position encoding buffers)
        missing = [k for k in missing if not any(x in k for x in ['cos_cached', 'sin_cached', 'base_angles', 'freqs', 'theta'])]
        if missing:
            print(f"  Warning: Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")

    return model, config


def main():
    parser = argparse.ArgumentParser(description='Passkey Retrieval Evaluation')
    parser.add_argument('checkpoints', nargs='+', type=Path,
                       help='Checkpoint paths to evaluate')
    parser.add_argument('--lengths', type=str, default='512,1024,2048,4096,8192',
                       help='Context lengths to test (comma-separated)')
    parser.add_argument('--depths', type=str, default='0.0,0.25,0.5,0.75,1.0',
                       help='Depth percentages to test (comma-separated)')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples per (length, depth) pair')
    parser.add_argument('--training-length', type=int, default=1024,
                       help='Training context length (for visualization)')
    parser.add_argument('--output-dir', type=Path, default=Path('plots'),
                       help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Parse lengths and depths
    context_lengths = [int(x) for x in args.lengths.split(',')]
    depth_percents = [float(x) for x in args.depths.split(',')]

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tiktoken.get_encoding('gpt2')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depth percents: {depth_percents}")
    print(f"Samples per test: {args.samples}")
    print()

    all_results = {}

    for ckpt_path in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_path}")
        print('='*60)

        # Load model with extended max_seq_len for long context eval
        max_ctx = max(context_lengths)
        model, config = load_model_from_checkpoint(ckpt_path, device, max_seq_len=max_ctx)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params/1e6:.1f}M params, rope_type={config.rope_type}")

        # Run evaluation
        results = run_passkey_evaluation(
            model, tokenizer, device,
            context_lengths, depth_percents,
            n_samples=args.samples,
            base_seed=args.seed
        )

        # Generate name from checkpoint path
        name = ckpt_path.parent.name
        if 'pope' in name.lower():
            display_name = 'PoPE'
        elif 'rope' in name.lower():
            display_name = 'RoPE'
        else:
            display_name = name

        all_results[display_name] = results

        # Save individual heatmap
        individual_path = args.output_dir / f'passkey_{name}.png'
        create_heatmap(
            results,
            f'Passkey Retrieval: {display_name}',
            individual_path,
            training_length=args.training_length
        )

        # Print summary
        print(f"\n{display_name} Results:")
        print(f"  Context Length | Avg Accuracy")
        print(f"  {'-'*30}")
        for j, ctx_len in enumerate(context_lengths):
            avg_acc = results['exact_match_matrix'][:, j].mean()
            marker = "  " if ctx_len <= args.training_length else "* "
            print(f"  {marker}{ctx_len:>6} tokens | {avg_acc:>6.1%}")

        # Save detailed results
        json_path = args.output_dir / f'passkey_{name}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'context_lengths': context_lengths,
                'depth_percents': depth_percents,
                'accuracy_matrix': results['accuracy_matrix'].tolist(),
                'exact_match_matrix': results['exact_match_matrix'].tolist(),
                'detailed_results': results['detailed_results']
            }, f, indent=2)

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Create comparison heatmap if multiple models
    if len(all_results) > 1:
        comparison_path = args.output_dir / 'passkey_comparison.png'
        create_comparison_heatmap(
            all_results, comparison_path,
            training_length=args.training_length
        )

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print('='*60)
        print(f"\n{'Context':<12}", end='')
        for name in all_results.keys():
            print(f"{name:<12}", end='')
        print()
        print('-' * (12 + 12 * len(all_results)))

        for j, ctx_len in enumerate(context_lengths):
            marker = "  " if ctx_len <= args.training_length else "* "
            print(f"{marker}{ctx_len:<10}", end='')
            for name, results in all_results.items():
                avg_acc = results['exact_match_matrix'][:, j].mean()
                print(f"{avg_acc:<12.1%}", end='')
            print()

        print(f"\n* = Extrapolation (beyond training length of {args.training_length})")


if __name__ == '__main__':
    main()
