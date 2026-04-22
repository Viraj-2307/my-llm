import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def estimate_loss(
    model,
    val_loader: DataLoader,
    eval_iters: int,
    device: str,
) -> dict:
    """
    Estimate validation loss over eval_iters batches.
    Switches model to eval mode, then back to train.
    """
    model.eval()
    losses = []

    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()

    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    perplexity = torch.tensor(avg_loss).exp().item()

    return {
        "val_loss":    round(avg_loss, 4),
        "perplexity":  round(perplexity, 2),
    }