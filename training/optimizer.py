import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    device: str,
) -> torch.optim.AdamW:
    """
    AdamW with weight decay applied only to weight matrices,
    NOT to biases, norms, or embeddings.
    This is standard practice from the GPT-2/LLaMA papers.
    """
    decay_params     = []
    no_decay_params  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Apply weight decay only to 2D+ tensors (linear weights)
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)  # biases, norms, 1D params

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use fused AdamW on CUDA, standard on CPU
    use_fused = (device == "cuda") and \
                ("fused" in torch.optim.AdamW.__init__.__doc__ or True)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        fused=use_fused if device == "cuda" else False,
    )

    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"[Optimizer] Decay params: {n_decay:,} | No-decay params: {n_no_decay:,}")

    return optimizer