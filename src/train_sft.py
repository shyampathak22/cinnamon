"""
SFT (Supervised Fine-Tuning) training for Cinnamon.

Key differences from pretraining:
- Variable-length sequences from JSONL
- Loss computed only on completion tokens (not prompts)
- Gradient checkpointing for long sequences
- Lower learning rate
- MTP modules skipped entirely during forward pass (saves memory)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import logging
import json

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
import wandb
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import tiktoken
import math
import time
import os
from model import Cinnamon
from config import ModelConfig
from dataclasses import dataclass, asdict

SFT_DATA_DIR = Path(__file__).parent.parent / "artifacts" / "sft_data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "artifacts" / "checkpoints"
_DROP_BUFFER_SUFFIXES = (".cos_cached", ".sin_cached", ".base_angles")


@dataclass
class SFTConfig:
    """SFT-specific training config."""
    lr: float = 2e-5  # Lower than pretraining
    min_lr: float = 2e-6  # 10% of peak
    batch_size: int = 1  # Small due to long sequences
    accumulation_steps: int = 16  # Effective batch = 16
    max_seq_len: int = 6144  # Match data filtering
    epochs: int = 3  # Multiple passes over data
    warmup_ratio: float = 0.03  # 3% warmup
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_steps: int = 200
    log_steps: int = 1
    checkpoint_steps: int = 500
    seed: int = 42
    num_workers: int = 12
    pin_memory: bool = True
    use_fp8: bool = True
    gradient_checkpointing: bool = True
    pack_sequences: bool = False  # Pack multiple examples per sequence


class SFTDataset(Dataset):
    """Load SFT data from JSONL with variable-length sequences."""

    def __init__(self, data_path: Path, max_seq_len: int, tokenizer_name: str = "gpt2", pack_sequences: bool = False):
        self.max_seq_len = max_seq_len
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.pad_token_id = self.tokenizer.eot_token  # Use EOT as pad
        self.pack_sequences = pack_sequences

        # Check for cached tokenized version
        cache_suffix = ".packed.cache.pt" if pack_sequences else ".cache.pt"
        cache_path = data_path.with_suffix(cache_suffix)
        if cache_path.exists() and cache_path.stat().st_mtime > data_path.stat().st_mtime:
            print(f"Loading cached tokenized data from {cache_path}...")
            cached = torch.load(cache_path, weights_only=True)
            self.tokenized_examples = cached["examples"]
            print(f"Loaded {len(self.tokenized_examples):,} cached examples")
            return

        # Load and tokenize all examples upfront
        print(f"Tokenizing {data_path} (will cache for next run)...")
        raw_examples = []
        with open(data_path) as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                input_ids, loss_mask = self._tokenize_messages(example["messages"])

                # Truncate if needed
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    loss_mask = loss_mask[:max_seq_len]

                raw_examples.append({
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                })

                if (i + 1) % 10000 == 0:
                    print(f"  Tokenized {i + 1:,} examples...")

        if pack_sequences:
            print(f"Packing {len(raw_examples):,} examples into {max_seq_len}-token sequences...")
            self.tokenized_examples = self._pack_examples(raw_examples, max_seq_len)
            print(f"  Packed into {len(self.tokenized_examples):,} sequences (eliminates padding!)")
        else:
            self.tokenized_examples = [
                {"input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
                 "loss_mask": torch.tensor(ex["loss_mask"], dtype=torch.float)}
                for ex in raw_examples
            ]

        # Cache for next run
        print(f"Caching tokenized data to {cache_path}...")
        torch.save({"examples": self.tokenized_examples}, cache_path)
        print(f"Loaded {len(self.tokenized_examples):,} examples from {data_path}")

    def _pack_examples(self, examples: list, max_seq_len: int) -> list:
        """Pack multiple examples into fixed-length sequences to eliminate padding."""
        # Sort by length for better packing
        examples = sorted(examples, key=lambda x: len(x["input_ids"]))

        packed = []
        current_ids = []
        current_mask = []

        for ex in examples:
            ids = ex["input_ids"]
            mask = ex["loss_mask"]

            # If adding this example would exceed max_seq_len, finalize current pack
            if current_ids and len(current_ids) + len(ids) > max_seq_len:
                # Pad to max_seq_len
                pad_len = max_seq_len - len(current_ids)
                current_ids.extend([self.pad_token_id] * pad_len)
                current_mask.extend([0.0] * pad_len)

                packed.append({
                    "input_ids": torch.tensor(current_ids, dtype=torch.long),
                    "loss_mask": torch.tensor(current_mask, dtype=torch.float),
                })
                current_ids = []
                current_mask = []

            # Add example to current pack
            current_ids.extend(ids)
            current_mask.extend(mask)

        # Don't forget the last pack
        if current_ids:
            pad_len = max_seq_len - len(current_ids)
            current_ids.extend([self.pad_token_id] * pad_len)
            current_mask.extend([0.0] * pad_len)
            packed.append({
                "input_ids": torch.tensor(current_ids, dtype=torch.long),
                "loss_mask": torch.tensor(current_mask, dtype=torch.float),
            })

        return packed

    def __len__(self):
        return len(self.tokenized_examples)

    def _tokenize_messages(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """
        Tokenize messages and return (input_ids, loss_mask).
        loss_mask = 1 for tokens we compute loss on (assistant responses), 0 otherwise.
        """
        input_ids = []
        loss_mask = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Format: <|role|>\ncontent
            role_tokens = self.tokenizer.encode(f"<|{role}|>\n", disallowed_special=())
            content_tokens = self.tokenizer.encode(content, disallowed_special=())

            # Role tokens never have loss
            input_ids.extend(role_tokens)
            loss_mask.extend([0] * len(role_tokens))

            # Content tokens: loss only for assistant
            input_ids.extend(content_tokens)
            if role == "assistant":
                loss_mask.extend([1] * len(content_tokens))
            else:
                loss_mask.extend([0] * len(content_tokens))

        # Add end token
        end_tokens = self.tokenizer.encode("<|end|>", disallowed_special=())
        input_ids.extend(end_tokens)
        loss_mask.extend([1] * len(end_tokens))  # Include end token in loss

        return input_ids, loss_mask

    def __getitem__(self, idx):
        item = self.tokenized_examples[idx]
        return {
            "input_ids": item["input_ids"],
            "loss_mask": item["loss_mask"],
            "length": len(item["input_ids"]),
        }


def collate_fn(batch, pad_token_id: int):
    """Collate variable-length sequences with padding."""
    max_len = max(item["length"] for item in batch)

    input_ids = []
    loss_masks = []

    for item in batch:
        seq_len = item["length"]
        pad_len = max_len - seq_len

        # Pad input_ids
        ids = item["input_ids"]
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=pad_token_id)
        input_ids.append(ids)

        # Pad loss_mask (pad positions have 0 loss)
        mask = item["loss_mask"]
        if pad_len > 0:
            mask = F.pad(mask, (0, pad_len), value=0.0)
        loss_masks.append(mask)

    return {
        "input_ids": torch.stack(input_ids),
        "loss_mask": torch.stack(loss_masks),
    }


def _strip_position_buffers(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.endswith(_DROP_BUFFER_SUFFIXES)}


def setup_ddp():
    """Setup distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return rank, local_rank, world_size
    else:
        # Single GPU
        return 0, 0, 1


class SFTTrainer:
    """Trainer for supervised fine-tuning."""

    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        model_config: ModelConfig,
        train_data_path: Path,
        val_data_path: Path | None,
        rank: int,
        local_rank: int,
        world_size: int,
        wandb_project: str = "cinnamon-sft",
        run_name: str | None = None,
    ):
        self.config = config
        self.model_config = model_config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        # Set seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        # Build datasets
        self.train_dataset = SFTDataset(train_data_path, config.max_seq_len, pack_sequences=config.pack_sequences)
        self.val_dataset = None
        if val_data_path and val_data_path.exists():
            self.val_dataset = SFTDataset(val_data_path, config.max_seq_len)

        pad_token_id = self.train_dataset.pad_token_id

        # Build dataloaders with length-based bucketing
        # Sort by length and create batches of similar lengths to minimize padding
        lengths = [ex["input_ids"].shape[0] for ex in self.train_dataset.tokenized_examples]
        sorted_indices = np.argsort(lengths)

        # Create buckets (groups of batch_size * accumulation_steps)
        bucket_size = config.batch_size * config.accumulation_steps * 4  # 4 accum cycles per bucket
        buckets = [sorted_indices[i:i + bucket_size] for i in range(0, len(sorted_indices), bucket_size)]

        # Shuffle buckets, then flatten
        np.random.shuffle(buckets)
        shuffled_indices = np.concatenate(buckets)

        from torch.utils.data import Sampler
        class BucketSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices
            def __iter__(self):
                return iter(self.indices.tolist())
            def __len__(self):
                return len(self.indices)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            sampler=BucketSampler(shuffled_indices),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.num_workers > 0,  # Keep workers alive between batches
            prefetch_factor=8 if config.num_workers > 0 else None,  # Prefetch more batches
            collate_fn=lambda b: collate_fn(b, pad_token_id),
            drop_last=True,
        )

        self.val_loader = None
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.num_workers > 0,
                prefetch_factor=8 if config.num_workers > 0 else None,
                collate_fn=lambda b: collate_fn(b, pad_token_id),
            )

        # Model
        self.model = model

        # Optimizer - no special indexer handling for SFT
        decay_params = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.lr, fused=True)

        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // config.accumulation_steps
        self.total_steps = steps_per_epoch * config.epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self._get_lr_lambda()
        )

        # State
        self.step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self.num_params = sum(p.numel() for p in model.parameters())

        # Logging
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._wandb_run = None
        if rank == 0:
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    **asdict(config),
                    "num_params": self.num_params,
                    "total_steps": self.total_steps,
                    "warmup_steps": self.warmup_steps,
                    "train_examples": len(self.train_dataset),
                },
            )

        base_name = run_name or (self._wandb_run.name if self._wandb_run else time.strftime("sft-%Y%m%d-%H%M%S"))
        self.checkpoint_dir = CHECKPOINT_DIR / base_name
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_lr_lambda(self):
        """Cosine decay with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            min_ratio = self.config.min_lr / self.config.lr
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
        return lr_lambda

    def train_step(self, batch):
        """Single training step (MTP disabled to save memory)."""
        input_ids = batch["input_ids"].to(self.local_rank, non_blocking=True)
        loss_mask = batch["loss_mask"].to(self.local_rank, non_blocking=True)

        # Shift for autoregressive: predict next token
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        mask = loss_mask[:, 1:]  # Align mask with targets

        with autocast("cuda", dtype=torch.bfloat16):
            # Get main logits only (skip_mtp=True to save memory)
            main_logits, _ = self.model(x, dsa_warmup=False, compute_aux=False, skip_mtp=True)

            # Main loss (masked)
            logits_flat = main_logits.reshape(-1, main_logits.size(-1))
            targets_flat = y.reshape(-1)
            mask_flat = mask.reshape(-1)
            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            main_loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            loss = main_loss / self.config.accumulation_steps

        loss.backward()

        # Count tokens with loss
        tokens_with_loss = mask_flat.sum().item()

        return loss * self.config.accumulation_steps, main_loss.item(), tokens_with_loss

    @torch.no_grad()
    def val_step(self, batch):
        """Validation step."""
        input_ids = batch["input_ids"].to(self.local_rank, non_blocking=True)
        loss_mask = batch["loss_mask"].to(self.local_rank, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        mask = loss_mask[:, 1:]

        with autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(x, dsa_warmup=False, compute_aux=False, skip_mtp=True)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = y.reshape(-1)
            mask_flat = mask.reshape(-1)

            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)

        return loss

    def validate(self, max_batches: int = 100):
        """Run validation."""
        if self.val_loader is None:
            return None

        self.model.eval()
        losses = []

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            loss = self.val_step(batch)
            losses.append(loss.item())

        self.model.train()

        if losses:
            return sum(losses) / len(losses)
        return None

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        if self.rank != 0:
            return

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod

        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "tokens_seen": self.tokens_seen,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def train(self):
        """Main training loop."""
        self.model.train()

        if self.rank == 0:
            pbar = tqdm(total=self.total_steps, desc="SFT Training")

        ema_loss = None
        microstep = 0
        accum_time = 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            if self.rank == 0:
                print(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")

            for batch in self.train_loader:
                start_t = time.perf_counter()

                # Gradient sync only on last accumulation step
                is_last_accum = (microstep + 1) % self.config.accumulation_steps == 0
                if self.world_size > 1:
                    sync_context = contextlib.nullcontext() if is_last_accum else self.model.no_sync()
                else:
                    sync_context = contextlib.nullcontext()

                with sync_context:
                    loss, main_loss, tokens_with_loss = self.train_step(batch)

                end_t = time.perf_counter()
                accum_time += (end_t - start_t)
                microstep += 1
                self.tokens_seen += tokens_with_loss

                if microstep % self.config.accumulation_steps == 0:
                    # Gradient clipping and optimizer step
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.grad_clip
                    ).item()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.step += 1

                    # EMA loss
                    loss_val = loss.item() if hasattr(loss, 'item') else loss
                    if ema_loss is None:
                        ema_loss = loss_val
                    else:
                        ema_loss = 0.9 * ema_loss + 0.1 * loss_val

                    # Logging
                    if self.rank == 0:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"loss: {loss_val:.4f} | lr: {self.scheduler.get_last_lr()[0]:.2e}"
                        )

                        if self.step % self.config.log_steps == 0 and self._wandb_run:
                            wandb.log({
                                "train/loss": loss_val,
                                "train/ema_loss": ema_loss,
                                "train/grad_norm": grad_norm,
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "train/step": self.step,
                                "train/epoch": self.epoch,
                                "train/tokens_seen": self.tokens_seen,
                                "train/step_time_ms": accum_time * 1000,
                            })

                    accum_time = 0

                    # Validation
                    if self.step % self.config.eval_steps == 0:
                        val_loss = self.validate()
                        if val_loss is not None and self.rank == 0:
                            print(f"\nVal loss: {val_loss:.4f}, perplexity: {math.exp(val_loss):.2f}")
                            if self._wandb_run:
                                wandb.log({
                                    "eval/loss": val_loss,
                                    "eval/perplexity": math.exp(val_loss),
                                    "eval/step": self.step,
                                })

                    # Checkpointing
                    if self.step % self.config.checkpoint_steps == 0:
                        self.save_checkpoint(f"step_{self.step}")

                    if self.step >= self.total_steps:
                        break

            if self.step >= self.total_steps:
                break

            # End of epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        # Final checkpoint
        self.save_checkpoint("final")

        if self.rank == 0:
            pbar.close()
            print(f"\nTraining complete! Final loss: {ema_loss:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SFT training for Cinnamon")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained checkpoint")
    parser.add_argument("--train-data", type=Path, default=SFT_DATA_DIR / "sft_stage1.jsonl")
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="cinnamon-sft")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=1)

    # Optimization flags
    parser.add_argument("--disable-fp8", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--pack-sequences", action="store_true", help="Pack multiple examples into each sequence")

    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_ddp()

    # Config
    sft_config = SFTConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        max_seq_len=args.max_seq_len,
        eval_steps=args.eval_steps,
        checkpoint_steps=args.checkpoint_steps,
        log_steps=args.log_steps,
        use_fp8=not args.disable_fp8,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        pack_sequences=args.pack_sequences,
    )

    model_config = ModelConfig()
    model_config.max_seq_len = args.max_seq_len  # Extend for SFT

    # Build model
    if rank == 0:
        print(f"Building model with max_seq_len={args.max_seq_len}...")

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
        0.0,  # balance_alpha - not used in SFT
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

    # Load pretrained weights
    if rank == 0:
        print(f"Loading pretrained weights from {args.checkpoint}...")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _strip_position_buffers(state_dict)

    # Handle DDP prefix if present
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.endswith(_DROP_BUFFER_SUFFIXES)]
    unexpected = [k for k in incompatible.unexpected_keys if not k.endswith(_DROP_BUFFER_SUFFIXES)]
    if missing or unexpected:
        print(f"Warning: Missing keys: {missing}, Unexpected keys: {unexpected}")

    # Enable gradient checkpointing
    if sft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("Gradient checkpointing enabled")

    # FP8 conversion
    if sft_config.use_fp8:
        from kernels import convert_to_fp8
        model = convert_to_fp8(model)
        if rank == 0:
            print("FP8 training enabled")

    # Move to GPU
    model.to(local_rank)

    # DDP wrapper
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Compile
    if not args.disable_compile:
        model = torch.compile(model)
        if rank == 0:
            print("torch.compile enabled")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        config=sft_config,
        model_config=model_config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )

    if rank == 0:
        print(f"\nSFT Training Configuration:")
        print(f"  Train examples: {len(trainer.train_dataset):,}")
        print(f"  Total steps: {trainer.total_steps:,}")
        print(f"  Warmup steps: {trainer.warmup_steps:,}")
        print(f"  Batch size: {sft_config.batch_size} x {sft_config.accumulation_steps} = {sft_config.batch_size * sft_config.accumulation_steps}")
        print(f"  Learning rate: {sft_config.lr}")
        print(f"  Max seq len: {sft_config.max_seq_len}")
        print(f"  Sparse attention kernel: {model_config.use_sparse_kernel}")
        print()

    # Train!
    trainer.train()


if __name__ == "__main__":
    main()
