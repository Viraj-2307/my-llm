"""
LoRA (Low-Rank Adaptation) implementation.
Wraps existing nn.Linear layers with trainable low-rank adapters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    Replaces nn.Linear with LoRA version.
    Original weights W are frozen.
    Adds trainable matrices A (in→rank) and B (rank→out).
    Output = W·x + (B·A·x) * (alpha/rank)
    """
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank

        in_features  = linear.in_features
        out_features = linear.out_features

        # Freeze original weights
        self.weight = linear.weight
        self.weight.requires_grad = False
        self.bias = linear.bias

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)   # B init to zero
        )                                     # so initial output = W·x

        self.lora_dropout = nn.Dropout(dropout)

        # Initialize A with kaiming uniform (standard for linear layers)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA path: x → dropout → A → B → scale
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}"


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: list[str] = ["wq", "wk", "wv", "wo"],
) -> nn.Module:
    """
    Walk the model and replace target Linear layers with LoRALinear.
    Only injects into attention layers by default.
    """
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]

                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank,
                                        alpha=alpha, dropout=dropout)
                setattr(parent, attr, lora_layer)
                replaced += 1

    print(f"[LoRA] Injected into {replaced} layers "
          f"(rank={rank}, alpha={alpha})")
    return model


def freeze_base_model(model: nn.Module) -> nn.Module:
    """Freeze all parameters except LoRA matrices."""
    frozen = 0
    trainable = 0

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    print(f"[LoRA] Frozen:    {frozen:,} params")
    print(f"[LoRA] Trainable: {trainable:,} params "
          f"({trainable/(frozen+trainable)*100:.1f}%)")
    return model


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights — much smaller than full model."""
    lora_state = {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)} LoRA tensors → {path}")


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu"):
    """Load LoRA weights back into an injected model."""
    lora_state = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f"[LoRA] Loaded from {path}")
    print(f"[LoRA] Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    return model