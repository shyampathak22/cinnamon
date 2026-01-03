import torch
import torch.nn
import torch.nn.functional as F
import logging

# Silence spammy torch.compile warnings
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
import wandb
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import math
import time
import os
import re
from model import Cinnamon
from config import ModelConfig, TrainConfig
from layers import MoE
from dataclasses import asdict

DATA_DIR = Path(__file__).parent.parent / "artifacts" / "tokenized_data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "artifacts" / "checkpoints"
_DROP_BUFFER_SUFFIXES = (".cos_cached", ".sin_cached", ".base_angles")


def _strip_position_buffers(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.endswith(_DROP_BUFFER_SUFFIXES)}


def _sanitize_run_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return safe or "run"

class ShardedDataset(IterableDataset):
    
    def __init__(self, data_dir, split, seq_len, rank, world_size, seed=42):
        super().__init__()
        # find all shards for a given split
        self.shards = sorted(list(Path(data_dir).rglob(f'{split}_*.npy')))
        self.seed = seed
        self.epoch = 0
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    
    def __iter__(self):
        # shuffle shards using seed
        rng = random.Random(self.seed + self.epoch)
        shards = self.shards.copy()
        rng.shuffle(shards)
        shards = [shard for i, shard in enumerate(shards) if i % self.world_size == self.rank]
        self.epoch += 1
        for shard in shards:
            # mem-map the shards
            arr = np.load(shard, mmap_mode='r')
            # iterate and yield tensors
            for i in range(0, len(arr) - self.seq_len, self.seq_len+1):
                chunk = arr[i:i + self.seq_len +1]
                yield torch.tensor(chunk, dtype=torch.long)

def setup_ddp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size

class Trainer():
    def __init__(self, model, config, model_config, rank, local_rank, world_size, wandb_project="cinnamon", run_name=None):
        self.seed = config.seed
        self.set_seed(self.seed)
        self.model = model
        self.config = config
        self.model_config = model_config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.train_loader = self._build_loader("train", config.seq_len)
        self.val_loader = self._build_loader("val", config.seq_len)
        indexer_decay = []
        indexer_no_decay = []
        main_decay = []
        main_no_decay = []
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                if "indexer" in name:
                    indexer_decay.append(p)
                else:
                    main_decay.append(p)
            else:
                if "indexer" in name:
                    indexer_no_decay.append(p)
                else:
                    main_no_decay.append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": main_decay, "weight_decay": config.weight_decay, "lr": config.lr, "group": "main"},
            {"params": main_no_decay, "weight_decay": 0.0, "lr": config.lr, "group": "main"},
            {"params": indexer_decay, "weight_decay": config.weight_decay, "lr": config.lr, "group": "indexer"},
            {"params": indexer_no_decay, "weight_decay": 0.0, "lr": config.lr, "group": "indexer"},
        ])
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.get_lr_lambda(config.warmup_steps, config.max_steps))
        self.step = 0
        self.tokens_seen = 0
        self._dsa_warmup_active = False
        self._gamma_active = model_config.gamma
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.flops_per_step = 6 * self.num_params * self.config.batch_size * self.config.seq_len * self.config.accumulation_steps
        self.tokens_per_step = self.config.batch_size * self.config.seq_len * self.config.accumulation_steps
        self._seq_len_switched = False
        # Note: GradScaler removed - not needed for BF16 (only for FP16)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._wandb_run = None
        if self.rank == 0:
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={**asdict(config), **asdict(model_config), "num_params": self.num_params, "max_steps": config.max_steps, "warmup_steps": config.warmup_steps},
            )
        base_name = run_name
        if not base_name and self._wandb_run is not None:
            base_name = self._wandb_run.name or self._wandb_run.id
        if not base_name:
            base_name = time.strftime("run-%Y%m%d-%H%M%S")
        safe_name = _sanitize_run_name(base_name)
        self.checkpoint_dir = CHECKPOINT_DIR / safe_name
        if self.rank == 0:
            if self.checkpoint_dir.exists():
                suffix = time.strftime("%Y%m%d-%H%M%S")
                self.checkpoint_dir = CHECKPOINT_DIR / f"{safe_name}-{suffix}"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def set_seed(self, seed):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

    def _build_loader(self, split, seq_len):
        dataset = ShardedDataset(DATA_DIR, split, seq_len, self.rank, self.world_size, self.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

    def _set_seq_len(self, seq_len):
        self.config.seq_len = seq_len
        self.train_loader = self._build_loader("train", seq_len)
        self.val_loader = self._build_loader("val", seq_len)
        self.tokens_per_step = self.config.batch_size * self.config.seq_len * self.config.accumulation_steps
        self.flops_per_step = 6 * self.num_params * self.config.batch_size * self.config.seq_len * self.config.accumulation_steps
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self.get_lr_lambda(self.config.warmup_steps, self.config.max_steps),
            last_epoch=self.step - 1,
        )

    def _maybe_switch_seq_len(self, dsa_warmup):
        target = self.config.seq_len_final
        if dsa_warmup or self._seq_len_switched or not target:
            return False
        if target == self.config.seq_len:
            self._seq_len_switched = True
            return False
        if self.rank == 0:
            print(f"Switching seq_len from {self.config.seq_len} to {target} after DSA warmup.")
        self._seq_len_switched = True
        self._set_seq_len(target)
        return True

    def load_state(self, checkpoint, load_optimizer=True):
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = int(checkpoint.get("step", 0))
        self.tokens_seen = int(checkpoint.get("tokens_seen", 0))

    def get_lr_lambda(self, warmup_steps, max_steps, min_lr_ratio=0.1):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return lr_lambda

    def _set_indexer_lr(self, lr):
        for group in self.optimizer.param_groups:
            if group.get("group") == "indexer":
                group["lr"] = lr

    def _set_dsa_warmup_state(self, warmup):
        if warmup == self._dsa_warmup_active:
            return
        self._dsa_warmup_active = warmup
        for name, p in self.model.named_parameters():
            if "indexer" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(not warmup)
        self._set_indexer_lr(self.config.dsa_warmup_lr if warmup else self.config.lr)
        self.optimizer.zero_grad(set_to_none=True)

    def _set_moe_gamma(self, gamma):
        if gamma == self._gamma_active:
            return
        self._gamma_active = gamma
        model = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        for module in model.modules():
            if isinstance(module, MoE):
                module.gamma = gamma

    def get_mtp_lambda(self):
        if self.tokens_seen >= self.config.mtp_lambda_switch_tokens:
            return self.config.mtp_lambda_final
        return self.config.mtp_lambda
    
    def train_step(self, batch):
        # Transfer once, then slice (avoids 3 separate H2D transfers)
        batch = batch.to(self.local_rank, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        dsa_warmup = self.step < self.config.dsa_warmup_steps

        with autocast('cuda', dtype=torch.bfloat16):
            main_logits, mtp_logits, dsa_kl, moe_balance = self.model(
                x, dsa_warmup=dsa_warmup, compute_aux=True
            )
            main_loss = F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)), y.reshape(-1))
            mtp_losses = []
            for depth, logits in enumerate(mtp_logits, start=1):
                target = batch[:, depth + 1:]
                mtp_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1)))
            mtp_loss = sum(mtp_losses) / len(mtp_losses) if mtp_losses else torch.tensor(0.0, device=x.device)
            mtp_lambda = self.get_mtp_lambda()

            aux_loss = 0.0
            aux_loss_value = 0.0
            if dsa_kl is not None:
                aux_loss = aux_loss + self.config.dsa_kl_weight * dsa_kl
                aux_loss_value += (self.config.dsa_kl_weight * dsa_kl).item()
            if moe_balance is not None:
                aux_loss = aux_loss + moe_balance
                aux_loss_value += moe_balance.item()

            loss = (main_loss + mtp_lambda * mtp_loss + aux_loss) / self.config.accumulation_steps

        # BF16 doesn't need GradScaler - direct backward
        loss.backward()
        return loss * self.config.accumulation_steps, mtp_loss, main_loss, dsa_kl, moe_balance, aux_loss_value, mtp_lambda

    @torch.no_grad()
    def val_step(self, batch):
        # Transfer once, then slice
        batch = batch.to(self.local_rank, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]

        with autocast('cuda', dtype=torch.bfloat16):
            main_logits, mtp_logits = self.model(x, dsa_warmup=False, compute_aux=False)
            main_loss = F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)), y.reshape(-1))
            mtp_losses = []
            for depth, logits in enumerate(mtp_logits, start=1):
                target = batch[:, depth + 1:]
                mtp_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1)))
            mtp_loss = sum(mtp_losses) / len(mtp_losses) if mtp_losses else torch.tensor(0.0, device=x.device)
            loss = (main_loss + self.config.mtp_lambda * mtp_loss)
        return loss, main_loss, mtp_loss
    
    def val(self, num_batches=100):
        losses = []
        main_losses = []
        mtp_losses = []
        val_step = 0
        if self.rank == 0:
            pbar = tqdm(total=num_batches, desc=f"Evaluating")
        for i, batch in enumerate(self.val_loader):
            if i >= num_batches:
                break
            loss, main_loss, mtp_loss = self.val_step(batch)
            losses.append(loss.item())
            main_losses.append(main_loss.item())
            mtp_losses.append(mtp_loss.item())
            val_step += 1
            if self.rank == 0:
                pbar.update(1)
        if self.rank == 0:
            pbar.close()
        if len(losses) == 0:
            return float('inf'), float('inf'), float('inf')
        eval_loss = sum(losses) / len(losses)
        eval_main_loss = sum(main_losses) / len(main_losses)
        eval_mtp_loss = sum(mtp_losses) / len(mtp_losses)
        return eval_loss, eval_main_loss, eval_mtp_loss
        
    def train(self):
        self.model.train()
        if self.rank == 0:
            pbar = tqdm(total=self.config.max_steps, desc=f"Training", leave=True)
        ema_loss = None
        ema_mfu = None
        ema_tok_per_sec = None
        microstep = 0
        accum_time = 0
        while self.step < self.config.max_steps:
            for batch in self.train_loader:
                dsa_warmup = self.step < self.config.dsa_warmup_steps
                self._set_dsa_warmup_state(dsa_warmup)
                if self._maybe_switch_seq_len(dsa_warmup):
                    if self.rank == 0:
                        pbar.total = self.config.max_steps
                        pbar.refresh()
                    break
                start_t = time.perf_counter()
                loss, mtp_loss, main_loss, dsa_kl, moe_balance, aux_loss_value, mtp_lambda = self.train_step(batch)
                end_t = time.perf_counter()
                accum_time += (end_t - start_t)
                microstep += 1
                self.tokens_seen += batch.size(0) * self.config.seq_len
                if self.tokens_seen >= self.config.gamma_switch_tokens:
                    self._set_moe_gamma(self.config.gamma_final)
                else:
                    self._set_moe_gamma(self.model_config.gamma)
                if microstep % self.config.accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip).item()
                    self.optimizer.step()
                    if not dsa_warmup:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    if self.rank == 0:
                        pbar.update(1)
                    elapsed_t = accum_time
                    accum_time = 0
                    tok_per_sec = self.tokens_per_step / elapsed_t * self.world_size
                    mfu = (self.flops_per_step / elapsed_t) / self.config.peak_flops
                    if ema_loss is None:
                        ema_loss = loss.item()
                    else:
                        ema_loss = 0.9 * ema_loss + (0.1) * loss.item()
                    if ema_mfu is None:
                        ema_mfu = mfu
                    else:
                        ema_mfu = 0.9 * ema_mfu + (0.1) * mfu
                    if ema_tok_per_sec is None:
                        ema_tok_per_sec = tok_per_sec
                    else:
                        ema_tok_per_sec = 0.9 * ema_tok_per_sec + (0.1) * tok_per_sec
                    if self.rank == 0:
                        pbar.set_postfix_str(f"loss: {loss.item():.4f} | ema loss: {ema_loss:.4f} | tok/s: {ema_tok_per_sec:.2f} | mfu: {ema_mfu * 100:.2f}% | tokens_seen: {self.tokens_seen/1e9:.2f}B")
                        if self._wandb_run is not None:
                            lr_main = None
                            lr_indexer = None
                            for group in self.optimizer.param_groups:
                                if group.get("group") == "main" and lr_main is None:
                                    lr_main = group.get("lr")
                                if group.get("group") == "indexer" and lr_indexer is None:
                                    lr_indexer = group.get("lr")
                            wandb.log({
                                "train/loss": loss.item(),
                                "train/aux_loss": aux_loss_value,
                                "train/mtp_lambda": mtp_lambda,
                                "train/grad_norm": grad_norm,
                                "train/step": self.step,
                                "train/tokens_seen": self.tokens_seen,
                                "train/tok_per_sec": tok_per_sec,
                                "train/step_time_ms": elapsed_t * 1000.0,
                                "train/mtp_loss": mtp_loss.item(),
                                "train/main_loss": main_loss.item(),
                                "train/dsa_kl": dsa_kl.item() if dsa_kl is not None else 0.0,
                                "train/moe_balance": moe_balance.item() if moe_balance is not None else 0.0,
                                "train/dsa_warmup": int(dsa_warmup),
                                "train/moe_gamma": self._gamma_active,
                                "train/seq_len": self.config.seq_len,
                                "train/lr_main": lr_main,
                                "train/lr_indexer": lr_indexer,
                            })
                    if self.step % self.config.eval_steps == 0 and self.step > 0:
                        self.model.eval()
                        eval_loss, eval_main_loss, eval_mtp_loss = self.val()
                        if self.rank == 0:
                            print(f"Eval Perplexity: {math.exp(eval_main_loss):.4f}")
                        if self.rank == 0 and self._wandb_run is not None:
                            wandb.log({
                                "eval/loss": eval_loss,
                                "eval/main_loss": eval_main_loss,
                                "eval/mtp_loss": eval_mtp_loss,
                                "eval/perplexity": math.exp(eval_loss),
                                "eval/main_perplexity": math.exp(eval_main_loss),
                                "eval/mtp_perplexity": math.exp(eval_mtp_loss),
                                "eval/tokens_seen": self.tokens_seen,
                            })
                        self.model.train()
                    if self.step % self.config.checkpoint_steps == 0 and self.step > 0:
                        if self.rank == 0:
                            torch.save({
                                "step": self.step,
                                'model_state_dict': self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "loss": loss.item(),
                                "tokens_seen": self.tokens_seen,
                                "config": self.config,
                                "ema_loss": ema_loss
                            }, self.checkpoint_dir / f"checkpoint_{self.step}.pt")
                            print(f"saved checkpoint to {self.checkpoint_dir / f'checkpoint_{self.step}.pt'}")
                    if self.step >= self.config.max_steps:
                        break
        if self.rank == 0:
                pbar.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Cinnamon model')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rope', action='store_true', help='Use standard RoPE (default)')
    group.add_argument('--pope', action='store_true', help='Use Polar Position Embedding')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--wandb-project', type=str, default='cinnamon')
    parser.add_argument('--resume', type=Path, default=None, help="Path to checkpoint to resume")
    parser.add_argument('--load-weights-only', action='store_true', help="Load model weights only (reset optimizer/scheduler)")
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--max-seq-len', type=int, default=None)
    parser.add_argument('--original-seq-len', type=int, default=None)
    parser.add_argument('--rope-factor', type=float, default=None)
    parser.add_argument('--beta-fast', type=int, default=None)
    parser.add_argument('--beta-slow', type=int, default=None)
    parser.add_argument('--mscale', type=float, default=None)
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--accumulation-steps', type=int, default=None)
    parser.add_argument('--seq-len-final', type=int, default=None)
    parser.add_argument('--eval-steps', type=int, default=None)
    parser.add_argument('--checkpoint-steps', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--dsa-warmup-steps', type=int, default=None)
    parser.add_argument('--dsa-warmup-lr', type=float, default=None)
    parser.add_argument('--disable-fp8', action='store_true')
    args = parser.parse_args()

    if args.load_weights_only and args.resume is None:
        parser.error("--load-weights-only requires --resume")

    # Determine rope_type from CLI args
    if args.pope:
        rope_type = 'pope'
    else:
        rope_type = 'rope'  # Default

    rank, local_rank, world_size = setup_ddp()

    model_config = ModelConfig()
    model_config.rope_type = rope_type  # Override from CLI
    train_config = TrainConfig()

    if args.seq_len is not None:
        train_config.seq_len = args.seq_len
    if args.seq_len_final is not None:
        train_config.seq_len_final = args.seq_len_final
    if args.max_tokens is not None:
        train_config.max_tokens = args.max_tokens
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.accumulation_steps is not None:
        train_config.accumulation_steps = args.accumulation_steps
    if args.eval_steps is not None:
        train_config.eval_steps = args.eval_steps
    if args.checkpoint_steps is not None:
        train_config.checkpoint_steps = args.checkpoint_steps
    if args.lr is not None:
        train_config.lr = args.lr
    if args.weight_decay is not None:
        train_config.weight_decay = args.weight_decay
    if args.dsa_warmup_steps is not None:
        train_config.dsa_warmup_steps = args.dsa_warmup_steps
    if args.dsa_warmup_lr is not None:
        train_config.dsa_warmup_lr = args.dsa_warmup_lr
    if args.disable_fp8:
        train_config.use_fp8 = False

    if args.rope_factor is not None:
        model_config.rope_factor = args.rope_factor
    if args.beta_fast is not None:
        model_config.beta_fast = args.beta_fast
    if args.beta_slow is not None:
        model_config.beta_slow = args.beta_slow
    if args.mscale is not None:
        model_config.mscale = args.mscale

    if args.original_seq_len is not None:
        model_config.original_seq_len = args.original_seq_len
    else:
        if model_config.original_seq_len != train_config.seq_len and rank == 0:
            print(
                f"Aligning original_seq_len ({model_config.original_seq_len}) "
                f"to training seq_len ({train_config.seq_len}) for YaRN."
            )
        model_config.original_seq_len = train_config.seq_len

    if args.max_seq_len is not None:
        model_config.max_seq_len = args.max_seq_len
    if model_config.max_seq_len < train_config.seq_len:
        if rank == 0:
            print(
                f"Raising max_seq_len ({model_config.max_seq_len}) "
                f"to training seq_len ({train_config.seq_len})."
            )
        model_config.max_seq_len = train_config.seq_len
    if train_config.seq_len_final is not None and model_config.max_seq_len < train_config.seq_len_final:
        if rank == 0:
            print(
                f"Raising max_seq_len ({model_config.max_seq_len}) "
                f"to final seq_len ({train_config.seq_len_final})."
            )
        model_config.max_seq_len = train_config.seq_len_final

    if rank == 0:
        print(f"Initializing Cinnamon model with {rope_type.upper()} positional encoding...")
    model = Cinnamon(model_config.d_model,
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
                     train_config.moe_balance_alpha,
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
                     model_config.indexer_use_hadamard)
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu")
        state_dict = resume_state["model_state_dict"] if "model_state_dict" in resume_state else resume_state
        state_dict = _strip_position_buffers(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.endswith(_DROP_BUFFER_SUFFIXES)]
        unexpected = [k for k in incompatible.unexpected_keys if not k.endswith(_DROP_BUFFER_SUFFIXES)]
        if missing or unexpected:
            raise RuntimeError(f"Unexpected state dict keys. Missing: {missing}, Unexpected: {unexpected}")
        if rank == 0:
            print(f"Loaded checkpoint weights from {args.resume}")
    if train_config.use_fp8:
        from kernels import convert_to_fp8
        model = convert_to_fp8(model)
        if rank == 0:
            print("FP8 training enabled (SM89+ for compute benefits, otherwise storage-only)")
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model = torch.compile(model)  # Default mode works best with DDP + MoE
    if rank == 0:
        print("Initializing trainer...")
    trainer = Trainer(
        model,
        train_config,
        model_config,
        rank,
        local_rank,
        world_size,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )
    if resume_state is not None and not args.load_weights_only:
        trainer.load_state(resume_state, load_optimizer=True)
        if rank == 0:
            print(f"Resumed optimizer/scheduler from {args.resume}")
    if resume_state is not None and args.load_weights_only and rank == 0:
        print("Loaded weights only; optimizer/scheduler state reset.")
    if rank == 0:
        print(f"Model and trainer initialized!\nParameters: {trainer.num_params}")
    trainer.train()
