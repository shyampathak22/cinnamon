"""
PG19 Perplexity Evaluation

Evaluates model perplexity on PG19 (Project Gutenberg books) at various context lengths.
Tests context length generalization - especially important for DroPE models.

Usage:
    python eval_pg19.py --checkpoint <path> --rope-type <rope|pope|none> [--d-rope 128]
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tiktoken
import wandb
import math
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from model import Cinnamon
from config import ModelConfig


def load_model(checkpoint_path, rope_type, d_rope, device):
    """Load model from checkpoint with specified rope_type."""
    model_config = ModelConfig()
    model_config.rope_type = rope_type
    model_config.d_rope = d_rope

    # For longer context eval, extend max_seq_len
    model_config.max_seq_len = 32768

    model = Cinnamon(
        model_config.d_model,
        model_config.n_layers,
        model_config.vocab_size,
        model_config.hidden_dim,
        model_config.n_heads,
        model_config.max_seq_len,
        model_config.d_ckv,
        model_config.d_cq,
        model_config.d_head,
        model_config.d_v,
        model_config.d_rope,
        model_config.n_routed,
        model_config.n_shared,
        model_config.top_k,
        model_config.expert_scale,
        model_config.gamma,
        0.0,  # balance_alpha (not needed for eval)
        model_config.dsa_topk,
        model_config.local_window,
        model_config.n_indexer_heads,
        model_config.d_indexer_head,
        model_config.rms_eps,
        model_config.rope_base,
        model_config.rope_type,
        model_config.mtp_depth,
        model_config.pope_delta_init,
        model_config.original_seq_len,
        model_config.rope_factor,
        model_config.beta_fast,
        model_config.beta_slow,
        model_config.mscale,
        model_config.indexer_use_fp8,
        model_config.indexer_use_hadamard,
        model_config.use_sparse_kernel,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Handle DDP prefix
    state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v
                  for k, v in state_dict.items()}

    # Filter out position embedding params if loading into DroPE model
    if rope_type == 'none':
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(x in k for x in ['w_qr', 'w_kr', 'qr_norm', 'kr_norm', 'pos_enc', 'delta'])}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, model_config


@torch.no_grad()
def evaluate_perplexity(model, tokens, context_length, stride=512, device='cuda'):
    """
    Evaluate perplexity with sliding window.

    Args:
        model: The language model
        tokens: Token IDs (1D tensor)
        context_length: Maximum context to use
        stride: How much to slide the window
        device: Device to run on

    Returns:
        Perplexity for this context length
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    seq_len = min(context_length, len(tokens) - 1)

    for start in range(0, len(tokens) - seq_len - 1, stride):
        end = start + seq_len + 1
        chunk = tokens[start:end].unsqueeze(0).to(device)

        x = chunk[:, :-1]
        y = chunk[:, 1:]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(x, dsa_warmup=False, compute_aux=False, skip_mtp=True)

        # Only count loss for tokens after the context window
        # This tests the model's ability to use long context
        loss = F.cross_entropy(
            logits[:, -stride:].reshape(-1, logits.size(-1)),
            y[:, -stride:].reshape(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += stride

        # Limit evaluation size for speed
        if total_tokens >= 50000:
            break

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser(description='PG19 Perplexity Evaluation')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--rope-type', type=str, required=True, choices=['rope', 'pope', 'none'])
    parser.add_argument('--d-rope', type=int, default=64, help='RoPE dimension')
    parser.add_argument('--model-name', type=str, default=None, help='Model name for logging')
    parser.add_argument('--wandb-project', type=str, default='cinnamon-eval')
    parser.add_argument('--output-dir', type=Path, default='./artifacts/eval_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--context-lengths', type=str, default='512,1024,2048,4096,8192,16384',
                        help='Comma-separated context lengths to evaluate')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model name
    if args.model_name is None:
        if args.rope_type == 'none':
            args.model_name = f"Drop{'RoPE' if args.d_rope == 128 else 'PoPE'}"
        else:
            args.model_name = args.rope_type.upper()

    # Parse context lengths
    context_lengths = [int(x) for x in args.context_lengths.split(',')]

    print(f"=== PG19 Perplexity Evaluation ===")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context lengths: {context_lengths}")
    print()

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=f"pg19-{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            'model_name': args.model_name,
            'rope_type': args.rope_type,
            'd_rope': args.d_rope,
            'checkpoint': str(args.checkpoint),
            'context_lengths': context_lengths,
            'eval_type': 'pg19_perplexity',
        }
    )

    # Load model
    print("Loading model...")
    model, model_config = load_model(args.checkpoint, args.rope_type, args.d_rope, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load PG19 dataset
    print("Loading PG19 dataset...")
    dataset = load_dataset("pg19", split="test", trust_remote_code=True)

    # Tokenize a subset of books
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = []
    for i, book in enumerate(dataset):
        if i >= 10:  # Use 10 books for evaluation
            break
        tokens = enc.encode(book['text'], disallowed_special=())
        all_tokens.extend(tokens)
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"Total tokens: {len(all_tokens):,}")

    # Evaluate at each context length
    results = {}
    for ctx_len in tqdm(context_lengths, desc="Evaluating context lengths"):
        if ctx_len > len(all_tokens) - 1:
            print(f"Skipping context length {ctx_len} (exceeds available tokens)")
            continue

        ppl, loss = evaluate_perplexity(model, all_tokens, ctx_len, device=args.device)
        results[ctx_len] = {'perplexity': ppl, 'loss': loss}

        print(f"  Context {ctx_len}: PPL = {ppl:.2f}, Loss = {loss:.4f}")

        # Log to wandb
        wandb.log({
            f'pg19/perplexity_{ctx_len}': ppl,
            f'pg19/loss_{ctx_len}': loss,
            'context_length': ctx_len,
        })

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ctx_lens = sorted(results.keys())
    ppls = [results[c]['perplexity'] for c in ctx_lens]
    losses = [results[c]['loss'] for c in ctx_lens]

    # Perplexity plot
    ax1.plot(ctx_lens, ppls, 'o-', linewidth=2, markersize=8, label=args.model_name)
    ax1.axvline(x=1024, color='gray', linestyle='--', alpha=0.5, label='Training context')
    ax1.set_xlabel('Context Length', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title(f'PG19 Perplexity vs Context Length\n{args.model_name}', fontsize=14)
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(ctx_lens, losses, 's-', linewidth=2, markersize=8, color='orange', label=args.model_name)
    ax2.axvline(x=1024, color='gray', linestyle='--', alpha=0.5, label='Training context')
    ax2.set_xlabel('Context Length', fontsize=12)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax2.set_title(f'PG19 Loss vs Context Length\n{args.model_name}', fontsize=14)
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = args.output_dir / f"pg19_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {fig_path}")

    # Log figure to wandb
    wandb.log({"pg19/perplexity_curve": wandb.Image(fig_path)})

    # Save results JSON
    results_path = args.output_dir / f"pg19_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'rope_type': args.rope_type,
            'd_rope': args.d_rope,
            'checkpoint': str(args.checkpoint),
            'results': {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print(f"Saved results to {results_path}")

    # Log summary table
    table = wandb.Table(columns=['Context Length', 'Perplexity', 'Loss'])
    for ctx_len in ctx_lens:
        table.add_data(ctx_len, results[ctx_len]['perplexity'], results[ctx_len]['loss'])
    wandb.log({'pg19/results_table': table})

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
