"""
Main training script.
Usage:
  CPU (local):   python train.py
  GPU (cloud):   python train.py --device cuda
"""
import os
import time
import math
import argparse
import torch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(2)
from torch.utils.data import DataLoader
from pathlib import Path

from config import ModelConfig, TrainConfig, SMALL
from model import LLM
from data.dataset import PackedDataset, StreamingDataset
from training.optimizer import build_optimizer
from training.scheduler import get_lr
from training.evaluator import estimate_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--resume",    type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming dataset (no pre-tokenized shards needed)")
    return parser.parse_args()


def save_checkpoint(model, optimizer, step, loss, cfg: TrainConfig):
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:07d}.pt")
    torch.save({
        "step":            step,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss":            loss,
        "model_config":    model.config,
    }, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Checkpoint] Resumed from step {ckpt['step']} (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


def train():
    args   = parse_args()
    tcfg   = TrainConfig(device=args.device)
    mcfg   = SMALL
    device = args.device

    torch.manual_seed(42)

    # ── Model ──────────────────────────────────────────────────────────
    model = LLM(mcfg).to(device)

    # torch.compile gives 20-40% speedup on CPU (PyTorch 2.0+)
    if tcfg.compile and hasattr(torch, "compile"):
        print("[Train] Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        learning_rate = tcfg.learning_rate,
        weight_decay  = tcfg.weight_decay,
        beta1         = tcfg.beta1,
        beta2         = tcfg.beta2,
        device        = device,
    )

    # ── Data ───────────────────────────────────────────────────────────
    if args.streaming:
        # No pre-tokenized shards needed — tokenizes on the fly
        print("[Data] Using streaming dataset (slower but no prep needed)")
        train_dataset = StreamingDataset(
            tokenizer_path = tcfg.tokenizer_path,
            max_seq_len    = mcfg.max_seq_len,
        )
        val_dataset = StreamingDataset(
            tokenizer_path = tcfg.tokenizer_path,
            max_seq_len    = mcfg.max_seq_len,
            seed           = 99,
        )
    else:
        train_dataset = PackedDataset(
            data_dir    = tcfg.data_dir,
            max_seq_len = mcfg.max_seq_len,
            split       = "train",
        )
        val_dataset = PackedDataset(
            data_dir    = tcfg.data_dir,
            max_seq_len = mcfg.max_seq_len,
            split       = "val",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = tcfg.batch_size,
        shuffle     = not args.streaming,
        num_workers = 0,          # 0 for CPU / streaming compatibility
        pin_memory  = (device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = tcfg.batch_size,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Resume ─────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)

    # ── Training loop ──────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    step          = start_step
    accum_step    = 0
    running_loss  = 0.0
    t0            = time.time()
    train_iter    = iter(train_loader)

    print(f"\n[Train] Starting from step {step} | device={device} | "
          f"batch={tcfg.batch_size} | grad_accum={tcfg.grad_accumulation_steps}")
    print(f"[Train] Effective batch size = "
          f"{tcfg.batch_size * tcfg.grad_accumulation_steps} tokens × {mcfg.max_seq_len} seq_len\n")

    while step < tcfg.max_iters:

        # ── Get batch ──────────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        # ── Update LR ──────────────────────────────────────────────────
        lr = get_lr(
            step          = step,
            warmup_steps  = tcfg.warmup_iters,
            max_steps     = tcfg.lr_decay_iters,
            max_lr        = tcfg.learning_rate,
            min_lr        = tcfg.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward + backward ─────────────────────────────────────────
        _, loss = model(x, y)

        # Scale loss for gradient accumulation
        loss = loss / tcfg.grad_accumulation_steps
        loss.backward()
        running_loss += loss.item() * tcfg.grad_accumulation_steps
        accum_step   += 1

        # ── Optimizer step (every grad_accumulation_steps) ─────────────
        if accum_step == tcfg.grad_accumulation_steps:

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            accum_step = 0
            step      += 1

            # ── Logging ────────────────────────────────────────────────
            if step % tcfg.log_interval == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    tcfg.batch_size *
                    tcfg.grad_accumulation_steps *
                    mcfg.max_seq_len
                ) / dt
                print(
                    f"step {step:6d} | "
                    f"loss {running_loss / (tcfg.log_interval * tcfg.grad_accumulation_steps):.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tokens_per_sec:,.0f} tok/s | "
                    f"{dt:.1f}s"
                )
                running_loss = 0.0
                t0 = time.time()

            # ── Evaluation ─────────────────────────────────────────────
            if step % tcfg.eval_interval == 0:
                metrics = estimate_loss(model, val_loader, tcfg.eval_iters, device)
                print(f"\n{'─'*55}")
                print(f"  EVAL step {step:6d} | "
                      f"val_loss {metrics['val_loss']} | "
                      f"perplexity {metrics['perplexity']}")
                print(f"{'─'*55}\n")

            # ── Checkpoint ─────────────────────────────────────────────
            if step % tcfg.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, step, running_loss, tcfg)

    # Final checkpoint
    save_checkpoint(model, optimizer, step, running_loss, tcfg)
    print("\n[Train] Training complete.")


if __name__ == "__main__":
    train()