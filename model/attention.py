import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    n_heads query heads share n_kv_heads key/value heads.
    When n_kv_heads == n_heads → standard MHA.
    When n_kv_heads == 1       → Multi-Query Attention (MQA).

    GQA reduces KV cache size by (n_heads / n_kv_heads)x —
    critical for long-context CPU inference.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # repeat factor
        self.head_dim   = dim // n_heads
        self.dropout    = dropout

        # Projections
        self.wq = nn.Linear(dim, n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim,    bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match query heads.
        (B, T, n_kv_heads, head_dim) → (B, T, n_heads, head_dim)
        """
        if self.n_rep == 1:
            return x
        B, T, n_kv, D = x.shape
        return x.unsqueeze(3).expand(B, T, n_kv, self.n_rep, D)\
                .reshape(B, T, n_kv * self.n_rep, D)

    def forward(
        self,
        x: torch.Tensor,              # (B, T, dim)
        cos: torch.Tensor,            # (max_seq_len, head_dim // 2)
        sin: torch.Tensor,
        mask: torch.Tensor = None,    # (T, T) causal mask
    ) -> torch.Tensor:

        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads
        k = self._repeat_kv(k)   # (B, T, n_heads, head_dim)
        v = self._repeat_kv(v)

        # Reshape for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        # Use PyTorch 2.0 flash attention if available (CPU-compatible)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(mask is None),
            )
        else:
            # Manual fallback
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn + mask
            else:
                causal = torch.triu(
                    torch.full((T, T), float("-inf"), device=x.device), diagonal=1
                )
                attn = attn + causal
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            attn_out = torch.matmul(attn, v)

        # Reshape and project output
        out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)