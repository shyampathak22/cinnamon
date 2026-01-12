"""
Needle in a Haystack (NIAH) / Passkey Retrieval Evaluation

Tests the model's ability to retrieve a specific piece of information (the "needle")
from various positions within a long context (the "haystack").

This is a key test for context length generalization and positional understanding.

Usage:
    python eval_niah.py --checkpoint <path> --rope-type <rope|pope|none> [--d-rope 128]
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tiktoken
import wandb
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from model import Cinnamon
from config import ModelConfig


# Needle template - a random passkey the model must retrieve
NEEDLE_TEMPLATE = "The secret passkey is: {passkey}. Remember this passkey."
QUERY_TEMPLATE = "What is the secret passkey mentioned in the text above?"

# Haystack filler text (generic text to pad the context)
HAYSTACK_FILLER = """
The city was bustling with activity as people went about their daily routines.
Cars honked in the distance while pedestrians hurried along the sidewalks.
Street vendors called out their wares, adding to the cacophony of urban life.
In the parks, children played while their parents watched from nearby benches.
The sun cast long shadows as the afternoon wore on, painting the buildings golden.
Office workers streamed out of tall glass towers, eager to head home after a long day.
Restaurants began to fill with hungry patrons seeking their evening meals.
The rhythm of the city continued unabated, a never-ending symphony of human activity.
"""


def load_model(checkpoint_path, rope_type, d_rope, device):
    """Load model from checkpoint with specified rope_type."""
    model_config = ModelConfig()
    model_config.rope_type = rope_type
    model_config.d_rope = d_rope
    model_config.max_seq_len = 32768  # Extended for long context eval

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
        0.0,  # balance_alpha
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

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v
                  for k, v in state_dict.items()}

    if rope_type == 'none':
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(x in k for x in ['w_qr', 'w_kr', 'qr_norm', 'kr_norm', 'pos_enc', 'delta'])}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, model_config


def generate_passkey():
    """Generate a random 5-digit passkey."""
    return str(np.random.randint(10000, 99999))


def create_haystack(enc, target_length, needle_position_ratio, passkey):
    """
    Create a haystack with a needle inserted at a specific position.

    Args:
        enc: Tokenizer
        target_length: Target total length in tokens
        needle_position_ratio: Where to insert needle (0.0 = start, 1.0 = end)
        passkey: The passkey to hide

    Returns:
        tokens: Token IDs
        needle_start: Token position where needle starts
    """
    needle = NEEDLE_TEMPLATE.format(passkey=passkey)
    needle_tokens = enc.encode(needle, disallowed_special=())

    # Calculate how much filler we need
    filler_tokens = enc.encode(HAYSTACK_FILLER, disallowed_special=())
    filler_needed = target_length - len(needle_tokens) - 50  # Leave room for query

    # Repeat filler to reach target length
    full_filler = []
    while len(full_filler) < filler_needed:
        full_filler.extend(filler_tokens)
    full_filler = full_filler[:filler_needed]

    # Insert needle at specified position
    needle_pos = int(len(full_filler) * needle_position_ratio)
    haystack = full_filler[:needle_pos] + needle_tokens + full_filler[needle_pos:]

    return haystack, needle_pos


@torch.no_grad()
def evaluate_retrieval(model, enc, haystack_tokens, passkey, device):
    """
    Evaluate if the model can retrieve the passkey.

    We use a simple approach: compute the probability the model assigns
    to the correct passkey digits when prompted.

    Returns:
        score: Average probability of correct next digit (0-1)
        correct: Whether the model would generate the correct passkey
    """
    # Add query to the end
    query = "\n\n" + QUERY_TEMPLATE + " The passkey is: "
    query_tokens = enc.encode(query, disallowed_special=())

    full_sequence = haystack_tokens + query_tokens
    input_ids = torch.tensor(full_sequence, dtype=torch.long, device=device).unsqueeze(0)

    # Get model predictions
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits, _ = model(input_ids, dsa_warmup=False, compute_aux=False, skip_mtp=True)

    # Get probabilities for the passkey digits
    passkey_tokens = enc.encode(passkey, disallowed_special=())

    probs = []
    correct_digits = 0

    for i, target_token in enumerate(passkey_tokens):
        # Position in logits where we predict this token
        pos = len(full_sequence) - 1 + i
        if pos >= logits.size(1):
            break

        token_probs = F.softmax(logits[0, pos], dim=-1)
        prob = token_probs[target_token].item()
        probs.append(prob)

        # Check if this would be the top prediction
        if logits[0, pos].argmax().item() == target_token:
            correct_digits += 1

    avg_prob = np.mean(probs) if probs else 0.0
    fully_correct = correct_digits == len(passkey_tokens)

    return avg_prob, fully_correct, correct_digits / len(passkey_tokens)


def main():
    parser = argparse.ArgumentParser(description='Needle in a Haystack Evaluation')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rope-type', type=str, required=True, choices=['rope', 'pope', 'none'])
    parser.add_argument('--d-rope', type=int, default=64)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--wandb-project', type=str, default='cinnamon-eval')
    parser.add_argument('--output-dir', type=Path, default='./artifacts/eval_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--context-lengths', type=str, default='512,1024,2048,4096,8192',
                        help='Context lengths to test')
    parser.add_argument('--depth-steps', type=int, default=10,
                        help='Number of depth positions to test (0-100%)')
    parser.add_argument('--num-trials', type=int, default=3,
                        help='Number of trials per (length, depth) pair')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name is None:
        if args.rope_type == 'none':
            args.model_name = f"Drop{'RoPE' if args.d_rope == 128 else 'PoPE'}"
        else:
            args.model_name = args.rope_type.upper()

    context_lengths = [int(x) for x in args.context_lengths.split(',')]
    depth_ratios = np.linspace(0.0, 1.0, args.depth_steps)

    print(f"=== Needle in a Haystack Evaluation ===")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depth steps: {args.depth_steps}")
    print(f"Trials per cell: {args.num_trials}")
    print()

    run = wandb.init(
        project=args.wandb_project,
        name=f"niah-{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            'model_name': args.model_name,
            'rope_type': args.rope_type,
            'd_rope': args.d_rope,
            'checkpoint': str(args.checkpoint),
            'context_lengths': context_lengths,
            'depth_steps': args.depth_steps,
            'num_trials': args.num_trials,
            'eval_type': 'niah',
        }
    )

    print("Loading model...")
    model, model_config = load_model(args.checkpoint, args.rope_type, args.d_rope, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    enc = tiktoken.get_encoding("gpt2")

    # Results matrix: [context_length, depth]
    results = np.zeros((len(context_lengths), len(depth_ratios)))
    accuracy = np.zeros((len(context_lengths), len(depth_ratios)))

    for i, ctx_len in enumerate(tqdm(context_lengths, desc="Context lengths")):
        for j, depth in enumerate(tqdm(depth_ratios, desc=f"  Depths @ {ctx_len}", leave=False)):
            trial_probs = []
            trial_correct = []

            for trial in range(args.num_trials):
                passkey = generate_passkey()
                haystack, needle_pos = create_haystack(enc, ctx_len, depth, passkey)

                if len(haystack) > 30000:  # Safety limit
                    trial_probs.append(0.0)
                    trial_correct.append(0.0)
                    continue

                prob, correct, digit_acc = evaluate_retrieval(model, enc, haystack, passkey, args.device)
                trial_probs.append(prob)
                trial_correct.append(float(correct))

            results[i, j] = np.mean(trial_probs)
            accuracy[i, j] = np.mean(trial_correct)

            wandb.log({
                'niah/avg_prob': results[i, j],
                'niah/accuracy': accuracy[i, j],
                'niah/context_length': ctx_len,
                'niah/depth_ratio': depth,
            })

    # Create visualization - NIAH heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Probability heatmap
    sns.heatmap(
        results,
        ax=axes[0],
        xticklabels=[f"{d:.0%}" for d in depth_ratios],
        yticklabels=context_lengths,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        cbar_kws={'label': 'Avg Probability'}
    )
    axes[0].set_xlabel('Needle Depth (% into context)', fontsize=12)
    axes[0].set_ylabel('Context Length (tokens)', fontsize=12)
    axes[0].set_title(f'NIAH: Passkey Retrieval Probability\n{args.model_name}', fontsize=14)

    # Accuracy heatmap
    sns.heatmap(
        accuracy,
        ax=axes[1],
        xticklabels=[f"{d:.0%}" for d in depth_ratios],
        yticklabels=context_lengths,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=True, fmt='.0%',
        cbar_kws={'label': 'Exact Match Accuracy'}
    )
    axes[1].set_xlabel('Needle Depth (% into context)', fontsize=12)
    axes[1].set_ylabel('Context Length (tokens)', fontsize=12)
    axes[1].set_title(f'NIAH: Exact Passkey Match\n{args.model_name}', fontsize=14)

    plt.tight_layout()

    fig_path = args.output_dir / f"niah_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {fig_path}")

    wandb.log({"niah/heatmap": wandb.Image(fig_path)})

    # Save results
    results_path = args.output_dir / f"niah_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'rope_type': args.rope_type,
            'd_rope': args.d_rope,
            'checkpoint': str(args.checkpoint),
            'context_lengths': context_lengths,
            'depth_ratios': depth_ratios.tolist(),
            'probability_matrix': results.tolist(),
            'accuracy_matrix': accuracy.tolist(),
        }, f, indent=2)
    print(f"Saved results to {results_path}")

    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Average probability: {results.mean():.3f}")
    print(f"Average accuracy: {accuracy.mean():.1%}")
    print(f"Accuracy @ training length (1024): {accuracy[context_lengths.index(1024) if 1024 in context_lengths else 0].mean():.1%}")

    wandb.log({
        'niah/summary_avg_prob': results.mean(),
        'niah/summary_avg_accuracy': accuracy.mean(),
    })

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
