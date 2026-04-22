import torch
from typing import Tuple

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin rotation matrices for RoPE.
    Called once at model init — cached for all forward passes.

    Returns:
        cos, sin: both shape (max_seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Inverse frequencies: theta_i = 1 / (base ^ (2i / head_dim))
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )  # shape: (head_dim // 2,)

    # Position indices
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (seq_len, head_dim // 2)
    freqs = torch.outer(positions, inv_freq)

    cos = freqs.cos()  # (max_seq_len, head_dim // 2)
    sin = freqs.sin()  # (max_seq_len, head_dim // 2)

    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensor.

    Args:
        x:   (batch, seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)

    Returns:
        Rotated tensor, same shape as x
    """
    B, T, H, D = x.shape
    half = D // 2

    # Split into two halves
    x1 = x[..., :half]   # (B, T, H, D/2)
    x2 = x[..., half:]   # (B, T, H, D/2)

    # Reshape cos/sin for broadcasting: (1, T, 1, D/2)
    cos = cos[:T].unsqueeze(0).unsqueeze(2)
    sin = sin[:T].unsqueeze(0).unsqueeze(2)

    # Rotation: [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)

    return rotated