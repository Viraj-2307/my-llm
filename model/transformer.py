import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .rmsnorm import RMSNorm
from .attention import GroupedQueryAttention
from .rope import precompute_rope_freqs


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    FFN(x) = (Swish(W_gate · x) ⊙ (W_up · x)) · W_down

    Two input projections (gate + up), one output projection.
    ~20% better than standard ReLU FFN at same parameter count.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # SiLU = Swish
        up   = self.w_up(x)
        return self.w_down(gate * up)   # element-wise gate


class TransformerBlock(nn.Module):
    """
    Single transformer block:
    x → RMSNorm → GQA Attention → residual
      → RMSNorm → SwiGLU FFN   → residual
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.dim, eps=config.rms_norm_eps)

        self.attn = GroupedQueryAttention(
            dim         = config.dim,
            n_heads     = config.n_heads,
            n_kv_heads  = config.n_kv_heads,
            max_seq_len = config.max_seq_len,
            dropout     = config.dropout,
        )

        self.ffn = SwiGLUFFN(
            dim        = config.dim,
            hidden_dim = config.ffn_hidden_dim,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Attention with pre-norm + residual
        x = x + self.dropout(
            self.attn(self.norm1(x), cos, sin, mask)
        )
        # FFN with pre-norm + residual
        x = x + self.dropout(
            self.ffn(self.norm2(x))
        )
        return x


class LLM(nn.Module):
    """
    Decoder-only transformer LLM.
    Architecture: Embedding → N × TransformerBlock → RMSNorm → LM Head
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding table
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm + output projection
        self.norm_out = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head  = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights
        # Reduces params and improves training stability
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies once
        cos, sin = precompute_rope_freqs(
            head_dim    = config.head_dim,
            max_seq_len = config.max_seq_len,
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"LLM initialized: {self.num_params():,} parameters")

    def _init_weights(self, module: nn.Module):
        """LLaMA-style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,           # (B, T) token ids
        targets: torch.Tensor = None # (B, T) for training loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, T = idx.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} > max_seq_len {self.config.max_seq_len}"

        # Token embeddings
        x = self.token_emb(idx)   # (B, T, dim)

        # Forward through all transformer blocks
        for block in self.blocks:
            x = block(x, self.cos, self.sin)

        # Final norm
        x = self.norm_out(x)

        if targets is not None:
            # Training: compute loss over all positions
            logits = self.lm_head(x)               # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
                ignore_index=-1,
                reduction="mean",
            )
        else:
            # Inference: only compute logits for last token (efficient)
            logits = self.lm_head(x[:, [-1], :])   # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,         # (B, T) prompt token ids
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k + top-p (nucleus) sampling.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len \
                           else idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)          # (B, 1, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k > 0:
                topk_vals = torch.topk(logits, top_k).values
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens above the cumulative probability threshold
                remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx