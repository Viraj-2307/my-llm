import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Faster than LayerNorm — no mean subtraction, no bias.
    Used in LLaMA, Mistral, and most modern LLMs.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scale

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x * rsqrt(mean(x^2) + eps)
        return x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight