"""
Compare evaluation results across all 4 models and generate combined visualizations.

Usage:
    python eval_compare.py --results-dir ./artifacts/eval_results
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
from datetime import datetime
from glob import glob


def load_pg19_results(results_dir):
    """Load all PG19 results from directory."""
    results = {}
    for path in glob(str(results_dir / "pg19_*.json")):
        with open(path) as f:
            data = json.load(f)
            model_name = data['model_name']
            if model_name not in results:
                results[model_name] = data
    return results


def load_niah_results(results_dir):
    """Load all NIAH results from directory."""
    results = {}
    for path in glob(str(results_dir / "niah_*.json")):
        with open(path) as f:
            data = json.load(f)
            model_name = data['model_name']
            if model_name not in results:
                results[model_name] = data
    return results


def plot_pg19_comparison(pg19_results, output_dir):
    """Create combined PG19 perplexity comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'RoPE': '#2ecc71', 'PoPE': '#3498db', 'DropRoPE': '#e74c3c', 'DropPoPE': '#9b59b6'}
    markers = {'RoPE': 'o', 'PoPE': 's', 'DropRoPE': '^', 'DropPoPE': 'D'}

    # Plot perplexity
    for model_name, data in sorted(pg19_results.items()):
        ctx_lens = sorted([int(k) for k in data['results'].keys()])
        ppls = [data['results'][str(c)]['perplexity'] for c in ctx_lens]

        axes[0].plot(ctx_lens, ppls, f'{markers.get(model_name, "o")}-',
                     color=colors.get(model_name, 'gray'),
                     linewidth=2, markersize=8, label=model_name)

    axes[0].axvline(x=1024, color='gray', linestyle='--', alpha=0.5, label='Training ctx')
    axes[0].set_xlabel('Context Length', fontsize=12)
    axes[0].set_ylabel('Perplexity', fontsize=12)
    axes[0].set_title('PG19 Perplexity vs Context Length', fontsize=14)
    axes[0].set_xscale('log', base=2)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot relative degradation from training context
    for model_name, data in sorted(pg19_results.items()):
        ctx_lens = sorted([int(k) for k in data['results'].keys()])
        ppls = [data['results'][str(c)]['perplexity'] for c in ctx_lens]

        # Normalize to performance at 1024
        baseline_ppl = data['results'].get('1024', data['results'][str(ctx_lens[0])])['perplexity']
        relative = [p / baseline_ppl for p in ppls]

        axes[1].plot(ctx_lens, relative, f'{markers.get(model_name, "o")}-',
                     color=colors.get(model_name, 'gray'),
                     linewidth=2, markersize=8, label=model_name)

    axes[1].axvline(x=1024, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Context Length', fontsize=12)
    axes[1].set_ylabel('Relative Perplexity (vs 1024)', fontsize=12)
    axes[1].set_title('Context Length Generalization', fontsize=14)
    axes[1].set_xscale('log', base=2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / f"pg19_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved PG19 comparison to {fig_path}")

    return fig_path


def plot_niah_comparison(niah_results, output_dir):
    """Create combined NIAH comparison plot - 2x2 grid of heatmaps."""
    models = sorted(niah_results.keys())
    n_models = len(models)

    if n_models == 0:
        print("No NIAH results found")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        if idx >= 4:
            break

        data = niah_results[model_name]
        accuracy = np.array(data['accuracy_matrix'])
        ctx_lens = data['context_lengths']
        depths = data['depth_ratios']

        sns.heatmap(
            accuracy,
            ax=axes[idx],
            xticklabels=[f"{d:.0%}" for d in depths],
            yticklabels=ctx_lens,
            cmap='RdYlGn',
            vmin=0, vmax=1,
            annot=True, fmt='.0%',
            cbar_kws={'label': 'Accuracy'}
        )
        axes[idx].set_xlabel('Needle Depth', fontsize=11)
        axes[idx].set_ylabel('Context Length', fontsize=11)
        axes[idx].set_title(f'{model_name}', fontsize=13, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(models), 4):
        axes[idx].axis('off')

    plt.suptitle('Needle in a Haystack: Passkey Retrieval Accuracy', fontsize=16, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / f"niah_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved NIAH comparison to {fig_path}")

    return fig_path


def plot_summary_bar(pg19_results, niah_results, output_dir):
    """Create summary bar chart comparing all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = sorted(set(pg19_results.keys()) | set(niah_results.keys()))
    colors = {'RoPE': '#2ecc71', 'PoPE': '#3498db', 'DropRoPE': '#e74c3c', 'DropPoPE': '#9b59b6'}

    # Bar 1: PPL at training context (1024)
    ppls_1024 = []
    for m in models:
        if m in pg19_results:
            ppl = pg19_results[m]['results'].get('1024', {}).get('perplexity', 0)
            ppls_1024.append(ppl)
        else:
            ppls_1024.append(0)

    bars1 = axes[0].bar(models, ppls_1024, color=[colors.get(m, 'gray') for m in models])
    axes[0].set_ylabel('Perplexity', fontsize=12)
    axes[0].set_title('PG19 PPL @ 1024 (Training Length)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, ppls_1024):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # Bar 2: PPL degradation at 8192 (8x training length)
    degradation = []
    for m in models:
        if m in pg19_results:
            ppl_1024 = pg19_results[m]['results'].get('1024', {}).get('perplexity', 1)
            ppl_8192 = pg19_results[m]['results'].get('8192', {}).get('perplexity', ppl_1024)
            degradation.append(ppl_8192 / ppl_1024)
        else:
            degradation.append(0)

    bars2 = axes[1].bar(models, degradation, color=[colors.get(m, 'gray') for m in models])
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Relative Perplexity', fontsize=12)
    axes[1].set_title('PPL Degradation @ 8192 (8x Training)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, degradation):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}x', ha='center', va='bottom', fontsize=10)

    # Bar 3: NIAH average accuracy
    niah_acc = []
    for m in models:
        if m in niah_results:
            acc = np.mean(niah_results[m]['accuracy_matrix'])
            niah_acc.append(acc)
        else:
            niah_acc.append(0)

    bars3 = axes[2].bar(models, niah_acc, color=[colors.get(m, 'gray') for m in models])
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_ylim(0, 1)
    axes[2].set_title('NIAH Average Accuracy', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, niah_acc):
        if val > 0:
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.0%}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Model Comparison Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / f"summary_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary comparison to {fig_path}")

    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument('--results-dir', type=Path, default='./artifacts/eval_results')
    parser.add_argument('--wandb-project', type=str, default='cinnamon-eval')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Evaluation Comparison ===")
    print(f"Results directory: {args.results_dir}")
    print()

    # Load results
    pg19_results = load_pg19_results(args.results_dir)
    niah_results = load_niah_results(args.results_dir)

    print(f"Found PG19 results for: {list(pg19_results.keys())}")
    print(f"Found NIAH results for: {list(niah_results.keys())}")
    print()

    if not pg19_results and not niah_results:
        print("No results found!")
        return

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=f"comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            'models': list(set(pg19_results.keys()) | set(niah_results.keys())),
        }
    )

    # Generate plots
    if pg19_results:
        pg19_fig = plot_pg19_comparison(pg19_results, args.output_dir)
        if pg19_fig:
            wandb.log({"comparison/pg19": wandb.Image(pg19_fig)})

    if niah_results:
        niah_fig = plot_niah_comparison(niah_results, args.output_dir)
        if niah_fig:
            wandb.log({"comparison/niah": wandb.Image(niah_fig)})

    if pg19_results or niah_results:
        summary_fig = plot_summary_bar(pg19_results, niah_results, args.output_dir)
        if summary_fig:
            wandb.log({"comparison/summary": wandb.Image(summary_fig)})

    # Log summary table
    models = sorted(set(pg19_results.keys()) | set(niah_results.keys()))
    table = wandb.Table(columns=['Model', 'PPL@1024', 'PPL@8192', 'PPL Degradation', 'NIAH Accuracy'])

    for m in models:
        ppl_1024 = pg19_results.get(m, {}).get('results', {}).get('1024', {}).get('perplexity', 0)
        ppl_8192 = pg19_results.get(m, {}).get('results', {}).get('8192', {}).get('perplexity', 0)
        degradation = ppl_8192 / ppl_1024 if ppl_1024 > 0 else 0
        niah_acc = np.mean(niah_results.get(m, {}).get('accuracy_matrix', [[0]])) if m in niah_results else 0

        table.add_data(m, f"{ppl_1024:.2f}", f"{ppl_8192:.2f}", f"{degradation:.2f}x", f"{niah_acc:.1%}")

    wandb.log({"comparison/summary_table": table})

    wandb.finish()
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
