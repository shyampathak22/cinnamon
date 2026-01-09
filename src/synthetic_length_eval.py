"""
Synthetic Length Extrapolation Evaluation

Rigorous synthetic tasks that isolate positional encoding capability.
Key design principles:
1. Use UNIQUE tokens to prevent induction head shortcuts
2. Multiple task variants with increasing difficulty
3. Statistical rigor with multiple seeds
4. Per-position accuracy analysis

Tasks:
1. Copying (unique tokens): Forces position tracking, not content matching
2. Selective Copy: Copy only tokens at specified positions
3. Associative Recall: Key-value retrieval at varying distances
4. First-token Retrieval: Always retrieve the first token (tests absolute position)

Usage:
    python synthetic_length_eval.py --task copy --train-len 64 --test-lens 64,128,256,512
    python synthetic_length_eval.py --task recall --train-len 32 --test-lens 32,64,128,256
    python synthetic_length_eval.py --task first --train-len 64 --test-lens 64,128,256,512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal
import argparse
from tqdm import tqdm
import json
import math

# ============================================================================
# Synthetic Data Generation
# ============================================================================

class CopyDataset(Dataset):
    """
    Copying task with UNIQUE tokens to prevent induction head shortcuts.

    Input:  [A, B, C, D, SEP, PAD, PAD, PAD, PAD]
    Target: [PAD, PAD, PAD, PAD, PAD, A, B, C, D]

    CRITICAL: Each sequence uses unique tokens (permutation of vocab).
    This forces the model to use positional information, not content matching.

    Why unique tokens matter:
    - With repeated tokens, induction heads can match content: "I saw X before, next was Y"
    - With unique tokens, the ONLY way to solve copying is tracking position
    """
    def __init__(self, seq_len: int, vocab_size: int = 512, n_samples: int = 10000,
                 sep_token: int = 0, pad_token: int = 1, seed: int = 42):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = n_samples
        self.sep_token = sep_token
        self.pad_token = pad_token

        # Pre-generate all samples with UNIQUE tokens per sequence
        rng = np.random.default_rng(seed)
        self.sequences = []
        available_tokens = list(range(2, vocab_size))  # Exclude SEP=0, PAD=1

        for _ in range(n_samples):
            # Sample unique tokens for this sequence (no repeats!)
            if seq_len <= len(available_tokens):
                seq = rng.choice(available_tokens, size=seq_len, replace=False)
            else:
                # If seq_len > vocab_size-2, we must allow some repeats
                seq = rng.choice(available_tokens, size=seq_len, replace=True)
            self.sequences.append(seq)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        total_len = 2 * self.seq_len + 1

        input_ids = np.full(total_len, self.pad_token, dtype=np.int64)
        input_ids[:self.seq_len] = seq
        input_ids[self.seq_len] = self.sep_token

        target = np.full(total_len, -100, dtype=np.int64)
        target[self.seq_len:self.seq_len + self.seq_len] = seq

        return torch.tensor(input_ids), torch.tensor(target)


class FirstTokenDataset(Dataset):
    """
    First-token retrieval: After seeing a sequence, recall the FIRST token.

    Input:  [A, B, C, D, E, F, ..., SEP, PAD]
    Target: [ignore..., A]

    This tests ABSOLUTE position tracking at position 0.
    The model must remember what was at position 0 after processing the full sequence.
    Harder at longer lengths because position 0 is further in the past.
    """
    def __init__(self, seq_len: int, vocab_size: int = 512, n_samples: int = 10000,
                 sep_token: int = 0, pad_token: int = 1, seed: int = 42):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = n_samples
        self.sep_token = sep_token
        self.pad_token = pad_token

        rng = np.random.default_rng(seed)
        available_tokens = list(range(2, vocab_size))
        self.sequences = []
        for _ in range(n_samples):
            seq = rng.choice(available_tokens, size=seq_len, replace=False if seq_len <= len(available_tokens) else True)
            self.sequences.append(seq)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        total_len = self.seq_len + 2  # seq + SEP + output position

        input_ids = np.full(total_len, self.pad_token, dtype=np.int64)
        input_ids[:self.seq_len] = seq
        input_ids[self.seq_len] = self.sep_token

        target = np.full(total_len, -100, dtype=np.int64)
        target[self.seq_len + 1] = seq[0]  # Retrieve first token

        return torch.tensor(input_ids), torch.tensor(target)


class NthTokenDataset(Dataset):
    """
    Nth-token retrieval: Retrieve the token at a specified position.

    Input:  [A, B, C, D, ..., SEP, position_indicator, PAD]
    Target: [ignore..., token_at_position]

    The position indicator is the query position (0, 1, 2, ...).
    Tests whether model can retrieve from ARBITRARY positions.
    """
    def __init__(self, seq_len: int, vocab_size: int = 512, n_samples: int = 10000,
                 sep_token: int = 0, pad_token: int = 1, seed: int = 42):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = n_samples
        self.sep_token = sep_token
        self.pad_token = pad_token

        rng = np.random.default_rng(seed)
        available_tokens = list(range(2, vocab_size))
        self.data = []
        for _ in range(n_samples):
            seq = rng.choice(available_tokens, size=seq_len, replace=False if seq_len <= len(available_tokens) else True)
            query_pos = rng.integers(0, seq_len)  # Random position to retrieve
            self.data.append((seq, query_pos))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq, query_pos = self.data[idx]
        total_len = self.seq_len + 3  # seq + SEP + pos_indicator + output

        input_ids = np.full(total_len, self.pad_token, dtype=np.int64)
        input_ids[:self.seq_len] = seq
        input_ids[self.seq_len] = self.sep_token
        # Position indicator: use a special token range (last 256 tokens for positions)
        input_ids[self.seq_len + 1] = self.vocab_size - 256 + query_pos  # position token

        target = np.full(total_len, -100, dtype=np.int64)
        target[self.seq_len + 2] = seq[query_pos]

        return torch.tensor(input_ids), torch.tensor(target)


class AssociativeRecallDataset(Dataset):
    """
    Associative Recall: Given key-value pairs, retrieve value for queried key.

    Format: [k1, v1, k2, v2, ..., kn, vn, SEP, query_key, PAD]
    Target: [ignore..., query_value]

    Tests: Can the model retrieve from arbitrary positions in context?
    """
    def __init__(self, n_pairs: int, vocab_size: int = 256, n_samples: int = 10000,
                 sep_token: int = 0, pad_token: int = 1, seed: int = 42):
        self.n_pairs = n_pairs
        self.vocab_size = vocab_size
        self.n_samples = n_samples
        self.sep_token = sep_token
        self.pad_token = pad_token

        rng = np.random.default_rng(seed)

        self.data = []
        for _ in range(n_samples):
            # Generate unique keys and random values
            keys = rng.choice(range(2, vocab_size), size=n_pairs, replace=False)
            values = rng.integers(2, vocab_size, size=n_pairs)
            # Random query (one of the keys)
            query_idx = rng.integers(0, n_pairs)
            self.data.append((keys, values, query_idx))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        keys, values, query_idx = self.data[idx]

        # Input: [k1, v1, k2, v2, ..., SEP, query_key, PAD]
        # Length: 2*n_pairs + 1 + 1 + 1 = 2*n_pairs + 3
        total_len = 2 * self.n_pairs + 3

        input_ids = np.full(total_len, self.pad_token, dtype=np.int64)
        # Fill key-value pairs
        for i, (k, v) in enumerate(zip(keys, values)):
            input_ids[2*i] = k
            input_ids[2*i + 1] = v
        input_ids[2*self.n_pairs] = self.sep_token
        input_ids[2*self.n_pairs + 1] = keys[query_idx]

        # Target: only predict the value after the query
        target = np.full(total_len, -100, dtype=np.int64)
        target[2*self.n_pairs + 2] = values[query_idx]

        return torch.tensor(input_ids), torch.tensor(target)


# ============================================================================
# Simple Transformer for Synthetic Tasks
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Standard RoPE for comparison."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, x, positions=None):
        seq_len = x.shape[1]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)

        # Apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated


class PoPE(nn.Module):
    """Polar Positional Encoding with learnable phase offset."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.register_buffer('theta', base ** (-(torch.arange(dim, dtype=torch.float32) / dim)))

        pos = torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer('base_angles', torch.outer(pos, self.theta))

        # Learnable phase offset
        self.delta = nn.Parameter(torch.zeros(dim))

    def forward(self, x, apply_delta=True, positions=None):
        seq_len = x.shape[1]

        if seq_len > self.base_angles.shape[0]:
            pos = torch.arange(seq_len, device=self.theta.device, dtype=self.theta.dtype)
            phases = torch.outer(pos, self.theta)
        else:
            phases = self.base_angles[:seq_len]

        phases = phases.view(1, seq_len, 1, -1)

        if apply_delta:
            delta = self.delta.clamp(-2 * math.pi, 0.0)
            phases = phases + delta

        # Magnitude via softplus
        mu = F.softplus(x)
        cos_out = mu * phases.cos()
        sin_out = mu * phases.sin()
        return torch.cat([cos_out, sin_out], dim=-1)


class SyntheticTransformer(nn.Module):
    """
    Minimal transformer for synthetic length generalization tasks.
    Supports RoPE, PoPE, or no positional encoding (NoPE) for comparison.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_rope: int = 32,
        max_seq_len: int = 512,
        pos_encoding: Literal['rope', 'pope', 'nope'] = 'pope',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_rope = d_rope
        self.pos_encoding_type = pos_encoding

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_rope, max_seq_len, pos_encoding, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_rope, max_seq_len, pos_encoding, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_rope, max_seq_len, pos_encoding, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_rope, max_seq_len, pos_encoding, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_rope = d_rope
        self.pos_encoding_type = pos_encoding
        self.scale = self.d_head ** -0.5

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == 'rope':
            self.pos_enc = RotaryEmbedding(d_rope, max_seq_len)
            self.rope_dim = d_rope
        elif pos_encoding == 'pope':
            self.pos_enc = PoPE(d_rope, max_seq_len)
            self.rope_dim = d_rope * 2  # PoPE doubles dimension
            # Normalize before PoPE for stability
            self.qr_norm = RMSNorm(d_rope)
            self.kr_norm = RMSNorm(d_rope)
        else:
            self.pos_enc = None
            self.rope_dim = 0

    def forward(self, x):
        B, L, _ = x.shape

        q = self.w_q(x).view(B, L, self.n_heads, self.d_head)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_head)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_head)

        if self.pos_enc is not None:
            if self.pos_encoding_type == 'rope':
                q_rope = q[..., :self.d_rope]
                k_rope = k[..., :self.d_rope]
                q_nope = q[..., self.d_rope:]
                k_nope = k[..., self.d_rope:]

                q_rope = self.pos_enc(q_rope)
                k_rope = self.pos_enc(k_rope)

                q = torch.cat([q_rope, q_nope], dim=-1)
                k = torch.cat([k_rope, k_nope], dim=-1)
            else:  # pope
                q_rope = q[..., :self.d_rope]
                k_rope = k[..., :self.d_rope]
                q_nope = q[..., self.d_rope:]
                k_nope = k[..., self.d_rope:]

                # Normalize before PoPE
                q_rope = self.qr_norm(q_rope.reshape(-1, self.d_rope)).view(B, L, self.n_heads, self.d_rope)
                k_rope = self.kr_norm(k_rope.reshape(-1, self.d_rope)).view(B, L, self.n_heads, self.d_rope)

                q_rope = self.pos_enc(q_rope, apply_delta=False)
                k_rope = self.pos_enc(k_rope, apply_delta=True)

                q = torch.cat([q_nope, q_rope], dim=-1)
                k = torch.cat([k_nope, k_rope], dim=-1)

        # Transpose for attention: [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.w_o(out)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        # Compute loss only on non-ignored positions
        mask = targets != -100
        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()

        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(input_ids)

        mask = targets != -100
        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        total_loss += loss.item() * mask.sum().item()

        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens


def get_dataset_class(task: str):
    """Return the appropriate dataset class for a task."""
    return {
        'copy': CopyDataset,
        'recall': AssociativeRecallDataset,
        'first': FirstTokenDataset,
        'nth': NthTokenDataset,
    }[task]


def run_single_seed(
    task: str,
    pe: str,
    train_len: int,
    test_lens: list[int],
    n_epochs: int,
    batch_size: int,
    d_model: int,
    n_layers: int,
    seed: int,
    device: str,
    vocab_size: int = 512,
):
    """Run a single training run with one seed. Returns dict of test accuracies."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    DatasetClass = get_dataset_class(task)

    # Create training data
    train_dataset = DatasetClass(seq_len=train_len, vocab_size=vocab_size, n_samples=10000, seed=seed)
    val_dataset = DatasetClass(seq_len=train_len, vocab_size=vocab_size, n_samples=1000, seed=seed+1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    max_seq = max(test_lens) * 3
    model = SyntheticTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_rope=d_model // 4,
        max_seq_len=max_seq,
        pos_encoding=pe
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Training loop
    best_val_acc = 0
    best_state = None
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test length extrapolation
    test_results = {}
    for test_len in test_lens:
        test_dataset = DatasetClass(seq_len=test_len, vocab_size=vocab_size, n_samples=500, seed=seed+1000)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        test_loss, test_acc = evaluate(model, test_loader, device)
        test_results[test_len] = test_acc

    return test_results, best_val_acc


def run_length_extrapolation_eval(
    task: Literal['copy', 'recall', 'first', 'nth'],
    pos_encodings: list[str],
    train_len: int,
    test_lens: list[int],
    n_epochs: int = 50,
    batch_size: int = 64,
    d_model: int = 128,
    n_layers: int = 4,
    n_seeds: int = 3,
    base_seed: int = 42,
    device: str = 'cuda'
):
    """
    Run full length extrapolation evaluation with statistical rigor.

    Runs multiple seeds and reports mean ± std.
    """
    results = {
        'task': task,
        'train_len': train_len,
        'test_lens': test_lens,
        'pos_encodings': pos_encodings,
        'n_seeds': n_seeds,
        'accuracies': {pe: {tl: [] for tl in test_lens} for pe in pos_encodings},
        'mean_accuracies': {pe: {} for pe in pos_encodings},
        'std_accuracies': {pe: {} for pe in pos_encodings},
    }

    for pe in pos_encodings:
        print(f"\n{'='*60}")
        print(f"Training {pe.upper()} on {task} task (train_len={train_len})")
        print(f"Running {n_seeds} seeds for statistical significance")
        print('='*60)

        all_val_accs = []

        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx * 1000
            print(f"\n--- Seed {seed_idx+1}/{n_seeds} (seed={seed}) ---")

            test_results, val_acc = run_single_seed(
                task=task,
                pe=pe,
                train_len=train_len,
                test_lens=test_lens,
                n_epochs=n_epochs,
                batch_size=batch_size,
                d_model=d_model,
                n_layers=n_layers,
                seed=seed,
                device=device
            )

            all_val_accs.append(val_acc)

            for test_len, acc in test_results.items():
                results['accuracies'][pe][test_len].append(acc)

            # Print this seed's results
            print(f"Val acc: {val_acc:.1%}")
            for test_len in test_lens:
                extrap = "*" if test_len > train_len else " "
                print(f"  {extrap}len={test_len:4d}: {test_results[test_len]:.1%}")

        # Compute statistics
        print(f"\n{pe.upper()} Summary (mean ± std over {n_seeds} seeds):")
        print(f"  Validation: {np.mean(all_val_accs):.1%} ± {np.std(all_val_accs):.1%}")
        for test_len in test_lens:
            accs = results['accuracies'][pe][test_len]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results['mean_accuracies'][pe][test_len] = mean_acc
            results['std_accuracies'][pe][test_len] = std_acc
            extrap = "*" if test_len > train_len else " "
            print(f"  {extrap}len={test_len:4d}: {mean_acc:.1%} ± {std_acc:.1%}")

    return results


def run_length_extrapolation_eval_simple(
    task: Literal['copy', 'recall', 'first', 'nth'],
    pos_encodings: list[str],
    train_len: int,
    test_lens: list[int],
    n_epochs: int = 50,
    batch_size: int = 64,
    d_model: int = 128,
    n_layers: int = 4,
    seed: int = 42,
    device: str = 'cuda'
):
    """Simple version with single seed (for quick testing)."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    DatasetClass = get_dataset_class(task)

    results = {
        'task': task,
        'train_len': train_len,
        'test_lens': test_lens,
        'pos_encodings': pos_encodings,
        'accuracies': {pe: {} for pe in pos_encodings}
    }

    for pe in pos_encodings:
        print(f"\n{'='*60}")
        print(f"Training {pe.upper()} on {task} task (train_len={train_len})")
        print('='*60)

        # Create training data
        train_dataset = DatasetClass(seq_len=train_len, n_samples=10000, seed=seed)
        val_dataset = DatasetClass(seq_len=train_len, n_samples=1000, seed=seed+1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        max_seq = max(test_lens) * 3  # Buffer for longest test
        model = SyntheticTransformer(
            vocab_size=512,
            d_model=d_model,
            n_heads=4,
            n_layers=n_layers,
            d_rope=d_model // 4,
            max_seq_len=max_seq,
            pos_encoding=pe
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params/1e6:.2f}M params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        # Training loop
        best_val_acc = 0
        pbar = tqdm(range(n_epochs), desc=f"{pe}")
        for epoch in pbar:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            pbar.set_postfix({
                'train_acc': f'{train_acc:.1%}',
                'val_acc': f'{val_acc:.1%}',
                'best': f'{best_val_acc:.1%}'
            })

        print(f"\nFinal training accuracy: {train_acc:.1%}")
        print(f"Final validation accuracy: {val_acc:.1%}")

        # Test length extrapolation
        print(f"\nLength extrapolation test:")
        for test_len in test_lens:
            test_dataset = DatasetClass(seq_len=test_len, n_samples=500, seed=seed+100)

            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            test_loss, test_acc = evaluate(model, test_loader, device)

            extrap = "  " if test_len <= train_len else "* "
            print(f"  {extrap}len={test_len:4d}: acc={test_acc:.1%}")
            results['accuracies'][pe][test_len] = test_acc

    return results


def plot_results(results: dict, output_path: Path):
    """Create a nice plot of length extrapolation results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'rope': '#e74c3c', 'pope': '#2ecc71', 'nope': '#3498db'}
    markers = {'rope': 'o', 'pope': 's', 'nope': '^'}

    for pe in results['pos_encodings']:
        accs = results['accuracies'][pe]
        lens = sorted(accs.keys())
        vals = [accs[l] for l in lens]

        ax.plot(lens, vals,
                color=colors.get(pe, 'gray'),
                marker=markers.get(pe, 'o'),
                linewidth=2,
                markersize=8,
                label=pe.upper())

    # Mark training length
    train_len = results['train_len']
    ax.axvline(x=train_len, color='gray', linestyle='--', alpha=0.7, label=f'Train len ({train_len})')

    ax.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f"Length Extrapolation: {results['task'].upper()} Task", fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_xscale('log', base=2)

    # Add extrapolation region shading
    ax.axvspan(train_len, max(results['test_lens']), alpha=0.1, color='blue', label='_nolegend_')
    ax.text(train_len * 1.5, 0.95, 'Extrapolation\nRegion', fontsize=9, color='blue', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Synthetic Length Extrapolation Eval')
    parser.add_argument('--task', type=str, choices=['copy', 'recall'], default='copy',
                       help='Task to evaluate')
    parser.add_argument('--train-len', type=int, default=32,
                       help='Training sequence length')
    parser.add_argument('--test-lens', type=str, default='16,32,64,128,256,512',
                       help='Test lengths (comma-separated)')
    parser.add_argument('--pos-encodings', type=str, default='rope,pope',
                       help='Position encodings to compare (comma-separated)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('plots'),
                       help='Output directory')
    args = parser.parse_args()

    test_lens = [int(x) for x in args.test_lens.split(',')]
    pos_encodings = [x.strip() for x in args.pos_encodings.split(',')]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_length_extrapolation_eval(
        task=args.task,
        pos_encodings=pos_encodings,
        train_len=args.train_len,
        test_lens=test_lens,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        seed=args.seed,
        device=device
    )

    # Save results
    json_path = args.output_dir / f'synthetic_{args.task}_len{args.train_len}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot
    plot_path = args.output_dir / f'synthetic_{args.task}_len{args.train_len}.png'
    plot_results(results, plot_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"\nTask: {args.task}, Train length: {args.train_len}")
    print(f"\n{'Length':<10}", end='')
    for pe in pos_encodings:
        print(f"{pe.upper():<12}", end='')
    print()
    print('-' * (10 + 12 * len(pos_encodings)))

    for test_len in test_lens:
        marker = "  " if test_len <= args.train_len else "* "
        print(f"{marker}{test_len:<8}", end='')
        for pe in pos_encodings:
            acc = results['accuracies'][pe].get(test_len, 0)
            print(f"{acc:<12.1%}", end='')
        print()

    print(f"\n* = Extrapolation (beyond training length)")


if __name__ == '__main__':
    main()
