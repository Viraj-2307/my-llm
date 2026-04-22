"""
Compute perplexity on held-out test data.
Perplexity = exp(average cross-entropy loss)
Lower is better. GPT-2 small = ~29, your target = ~200-400
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LLM
from data.dataset import PackedDataset


@torch.no_grad()
def compute_perplexity(
    model: LLM,
    data_dir: str,
    max_seq_len: int = 1024,
    batch_size: int = 4,
    max_batches: int = 50,
    device: str = "cpu",
) -> dict:

    model.eval()

    dataset = PackedDataset(
        data_dir    = data_dir,
        max_seq_len = max_seq_len,
        split       = "val",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss   = 0.0
    total_tokens = 0
    num_batches  = 0

    print(f"Computing perplexity over {min(max_batches, len(loader))} batches...")

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)

        # Count non-padding tokens
        n_tokens = (y != -1).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens
        num_batches  += 1

        if (i + 1) % 10 == 0:
            running_ppl = torch.tensor(
                total_loss / total_tokens
            ).exp().item()
            print(f"  Batch {i+1}/{max_batches} | running perplexity: {running_ppl:.2f}")

    avg_loss   = total_loss / total_tokens
    perplexity = torch.tensor(avg_loss).exp().item()

    model.train()

    return {
        "avg_loss":    round(avg_loss, 4),
        "perplexity":  round(perplexity, 2),
        "num_batches": num_batches,
        "num_tokens":  total_tokens,
    }