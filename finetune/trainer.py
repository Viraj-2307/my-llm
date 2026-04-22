"""
LoRA fine-tuning training loop.
"""
import torch
import torch.nn as nn
import time
import os
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from finetune.lora import save_lora_weights


def finetune(
    model: nn.Module,
    dataset,
    output_dir: str = "finetune/checkpoints",
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    num_epochs: int = 3,
    grad_accumulation: int = 4,
    grad_clip: float = 1.0,
    warmup_steps: int = 50,
    val_split: float = 0.05,
    log_interval: int = 10,
    device: str = "cpu",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train / val split
    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    # Only optimize LoRA params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=0.01
    )

    total_steps  = (len(train_loader) // grad_accumulation) * num_epochs
    print(f"[Finetune] {len(train_ds)} train | {len(val_ds)} val samples")
    print(f"[Finetune] {total_steps} total optimizer steps")
    print(f"[Finetune] Trainable params: {sum(p.numel() for p in trainable):,}")

    step = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        accum_step   = 0
        t0           = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # LR warmup
            if step < warmup_steps:
                lr = learning_rate * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            _, loss = model(x, y)
            loss.backward()
            running_loss += loss.item()
            accum_step   += 1

            if accum_step == grad_accumulation:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accum_step = 0
                step      += 1

                if step % log_interval == 0:
                    dt       = time.time() - t0
                    avg_loss = running_loss / (log_interval * grad_accumulation)
                    print(
                        f"epoch {epoch+1} | step {step:4d} | "
                        f"loss {avg_loss:.4f} | "
                        f"{dt:.1f}s"
                    )
                    running_loss = 0.0
                    t0 = time.time()

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses)
        print(f"\nepoch {epoch+1} complete | val_loss: {val_loss:.4f}\n")

        # Save LoRA weights after each epoch
        save_lora_weights(
            model,
            os.path.join(output_dir, f"lora_epoch_{epoch+1}.pt")
        )

    print("[Finetune] Training complete.")
    return model