import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import wandb

from config import ModelConfig
from model import Cinnamon


_DROP_BUFFER_SUFFIXES = (".cos_cached", ".sin_cached", ".base_angles")


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype_arg: str | None, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _load_model_config(path: Path | None) -> ModelConfig:
    cfg = ModelConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _parse_lengths(arg: str | None) -> list[int] | None:
    if not arg:
        return None
    lengths = []
    seen = set()
    for segment in arg.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if ":" in segment:
            parts = [p.strip() for p in segment.split(":") if p.strip()]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid length range: {segment}")
            start = int(parts[0])
            end = int(parts[1])
            mult = int(parts[2]) if len(parts) == 3 else 2
            if start <= 0 or end <= 0 or mult <= 1:
                raise ValueError(f"Invalid length range: {segment}")
            if start > end:
                raise ValueError(f"Start > end in length range: {segment}")
            value = start
            while value <= end:
                if value not in seen:
                    lengths.append(value)
                    seen.add(value)
                value *= mult
        else:
            length = int(segment)
            if length <= 0:
                raise ValueError(f"Invalid sequence length: {length}")
            if length not in seen:
                lengths.append(length)
                seen.add(length)
    return lengths


def _strip_position_buffers(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.endswith(_DROP_BUFFER_SUFFIXES)}


def _normalize_context(example: dict) -> str:
    if "ctx" in example and example["ctx"]:
        return example["ctx"].strip()
    ctx_a = example.get("ctx_a", "").strip()
    ctx_b = example.get("ctx_b", "").strip()
    if ctx_b:
        if ctx_a and not ctx_a.endswith((" ", "\n")):
            ctx_a = ctx_a + " "
        return (ctx_a + ctx_b).strip()
    return ctx_a


def _normalize_ending(context: str, ending: str) -> str:
    if context and ending and not context.endswith((" ", "\n")) and not ending.startswith((" ", "\n")):
        return " " + ending
    return ending


def _build_tokens(enc, context: str, ending: str, max_seq_len: int):
    ending_text = _normalize_ending(context, ending)
    ctx_tokens = enc.encode(context, disallowed_special=())
    ending_tokens = enc.encode(ending_text, disallowed_special=())
    total_len = len(ctx_tokens) + len(ending_tokens)
    if total_len > max_seq_len:
        overflow = total_len - max_seq_len
        if overflow >= len(ctx_tokens):
            ctx_tokens = ctx_tokens[-1:]
        else:
            ctx_tokens = ctx_tokens[overflow:]
    return ctx_tokens, ending_tokens


def _score_candidate(model, tokens, ctx_len, device, dtype):
    if ctx_len <= 0 or len(tokens) < 2:
        return float("-inf")
    input_ids = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
    target_ids = torch.tensor(tokens[1:], device=device).unsqueeze(0)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=dtype):
                logits, _ = model(input_ids)
        else:
            logits, _ = model(input_ids)
        logprobs = F.log_softmax(logits, dim=-1)
        tgt_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        start = ctx_len - 1
        end = start + (len(tokens) - ctx_len)
        return tgt_logprobs[:, start:end].sum().item()


def evaluate_split(model, enc, split, max_seq_len, max_examples, device, dtype, save_preds):
    dataset = load_dataset("hellaswag", split=split)
    if max_examples and max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    correct = 0
    labeled = 0
    margins = []
    preds = []

    for idx, example in enumerate(tqdm(dataset, desc=f"HellaSwag {split}")):
        context = _normalize_context(example)
        endings = example["endings"]
        scores = []
        for ending in endings:
            ctx_tokens, ending_tokens = _build_tokens(enc, context, ending, max_seq_len)
            tokens = ctx_tokens + ending_tokens
            score = _score_candidate(model, tokens, len(ctx_tokens), device, dtype)
            scores.append(score)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) > 1:
            margins.append(sorted_scores[0] - sorted_scores[1])

        label = example.get("label", -1)
        if label is not None and label != -1:
            labeled += 1
            if pred == int(label):
                correct += 1

        if save_preds is not None:
            example_id = example.get("ind", example.get("source_id", idx))
            preds.append(
                {
                    "id": example_id,
                    "prediction": pred,
                    "scores": scores,
                    "label": int(label) if label is not None else -1,
                }
            )

    accuracy = correct / labeled if labeled > 0 else None
    mean_margin = sum(margins) / len(margins) if margins else None

    result = {
        "split": split,
        "max_seq_len": max_seq_len,
        "examples": len(dataset),
        "labeled_examples": labeled,
        "accuracy": accuracy,
        "mean_margin": mean_margin,
    }
    return result, preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cinnamon on HellaSwag.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Comma-separated eval lengths or start:end[:mult] (e.g., 1024,2048 or 1024:8192:2)",
    )
    parser.add_argument("--original-seq-len", type=int, default=None, help="Training context length for YaRN scaling")
    parser.add_argument("--rope-factor", type=float, default=None, help="YaRN rope factor (1.0 disables YaRN)")
    parser.add_argument("--beta-fast", type=int, default=None)
    parser.add_argument("--beta-slow", type=int, default=None)
    parser.add_argument("--mscale", type=float, default=None)
    parser.add_argument("--disable-mtp", action="store_true")
    parser.add_argument("--save-preds", type=Path, default=None)
    parser.add_argument("--save-metrics", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default="cinnamon")
    parser.add_argument("--run-name", type=str, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rope", action="store_true")
    group.add_argument("--pope", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    cfg = _load_model_config(args.model_config)
    if args.rope:
        cfg.rope_type = "rope"
    if args.pope:
        cfg.rope_type = "pope"
    if args.original_seq_len is not None:
        cfg.original_seq_len = args.original_seq_len
    if args.rope_factor is not None:
        cfg.rope_factor = args.rope_factor
    if args.beta_fast is not None:
        cfg.beta_fast = args.beta_fast
    if args.beta_slow is not None:
        cfg.beta_slow = args.beta_slow
    if args.mscale is not None:
        cfg.mscale = args.mscale
    eval_lengths = _parse_lengths(args.lengths)
    if eval_lengths is None:
        eval_lengths = [args.max_seq_len if args.max_seq_len is not None else cfg.max_seq_len]
    model_max_seq_len = max(eval_lengths)
    if args.max_seq_len is not None:
        model_max_seq_len = max(model_max_seq_len, args.max_seq_len)
    cfg.max_seq_len = model_max_seq_len

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            **asdict(cfg),
            "split": args.split,
            "lengths": eval_lengths,
            "max_examples": args.max_examples,
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "dtype": str(dtype),
            "disable_mtp": args.disable_mtp,
        },
    )

    model = Cinnamon(
        cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.hidden_dim, cfg.n_heads, cfg.max_seq_len,
        cfg.d_ckv, cfg.d_cq, cfg.d_head, cfg.d_v, cfg.d_rope, cfg.n_routed, cfg.n_shared,
        cfg.top_k, cfg.expert_scale, cfg.gamma, 0.0, cfg.dsa_topk, cfg.local_window,
        cfg.n_indexer_heads, cfg.d_indexer_head, cfg.rms_eps, cfg.rope_base, cfg.rope_type,
        cfg.mtp_depth, cfg.pope_delta_init, cfg.original_seq_len, cfg.rope_factor, cfg.beta_fast,
        cfg.beta_slow, cfg.mscale, cfg.indexer_use_fp8, cfg.indexer_use_hadamard
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = _strip_position_buffers(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.endswith(_DROP_BUFFER_SUFFIXES)]
    unexpected = [k for k in incompatible.unexpected_keys if not k.endswith(_DROP_BUFFER_SUFFIXES)]
    if missing or unexpected:
        raise RuntimeError(f"Unexpected state dict keys. Missing: {missing}, Unexpected: {unexpected}")
    if args.disable_mtp:
        model.mtp_modules = nn.ModuleList([])
        model.mtp_depth = 0
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    all_metrics = []
    all_preds = []
    for length in eval_lengths:
        for split in splits:
            metrics, preds = evaluate_split(
                model,
                enc,
                split,
                length,
                args.max_examples,
                device,
                dtype,
                args.save_preds,
            )
            all_metrics.append(metrics)
            if preds:
                for pred in preds:
                    pred["split"] = split
                    pred["max_seq_len"] = length
                all_preds.extend(preds)
            if metrics["accuracy"] is None:
                print(f"len={length} {split}: {metrics['examples']} examples (no labels)")
            else:
                print(
                    f"len={length} {split}: acc={metrics['accuracy']:.4f} "
                    f"({metrics['labeled_examples']}/{metrics['examples']}) "
                    f"mean_margin={metrics['mean_margin']:.4f}"
                )
            if wandb_run is not None:
                log_data = {
                    f"eval/hellaswag/{split}/accuracy": metrics["accuracy"],
                    f"eval/hellaswag/{split}/mean_margin": metrics["mean_margin"],
                    f"eval/hellaswag/{split}/examples": metrics["examples"],
                    f"eval/hellaswag/{split}/labeled_examples": metrics["labeled_examples"],
                    "eval/hellaswag/max_seq_len": length,
                }
                wandb.log({k: v for k, v in log_data.items() if v is not None})

    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        args.save_preds.write_text("\n".join(json.dumps(p) for p in all_preds))
    if args.save_metrics is not None:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(json.dumps({"metrics": all_metrics, "model_config": asdict(cfg)}, indent=2))


if __name__ == "__main__":
    main()
