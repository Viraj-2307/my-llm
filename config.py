# config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    # --- Dimensions ---
    vocab_size: int = 32000        # BPE vocabulary
    dim: int = 512                 # embedding dimension
    n_layers: int = 8              # transformer blocks
    n_heads: int = 8               # query heads
    n_kv_heads: int = 4            # key/value heads (GQA — fewer than q heads)
    ffn_dim_multiplier: float = 2.66  # SwiGLU hidden = int(dim * multiplier)

    # --- Context ---
    max_seq_len: int = 1024        # max context length

    # --- Regularization ---
    dropout: float = 0.0           # 0 for inference, 0.1 for training

    # --- Normalization ---
    rms_norm_eps: float = 1e-5

    # --- Derived (auto-computed) ---
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        # Round to nearest multiple of 256 for hardware efficiency
        raw = int(self.dim * self.ffn_dim_multiplier)
        return (raw + 255) // 256 * 256


@dataclass
class TrainConfig:
    # --- Data ---
    dataset_name: str = "HuggingFaceFW/fineweb"
    tokenizer_path: str = "tokenizer/tokenizer.model"
    data_dir: str = "data/shards"

    # --- Training ---
    batch_size: int = 4
    grad_accumulation_steps: int = 4       # ← reduced from 8
    max_iters: int = 2_000                 # ← reduced from 100_000
    eval_interval: int = 200               # ← reduced from 500
    eval_iters: int = 20                   # ← reduced from 100
    log_interval: int = 10

    # --- Optimizer ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- LR Schedule ---
    warmup_iters: int = 100                # ← reduced from 2000
    lr_decay_iters: int = 2_000            # ← match max_iters
    min_lr: float = 3e-5

    # --- System ---
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500         # ← reduced from 2000
    resume_from: Optional[str] = None


@dataclass
class InferenceConfig:
    model_path: str = "checkpoints/final"
    gguf_path: str = "export/model-q4_k_m.gguf"
    n_threads: int = 8             # CPU threads for llama.cpp
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40


# --- Preset configs ---

SMALL = ModelConfig(
    dim=512, n_layers=6, n_heads=8, n_kv_heads=4,
)  # ~120M params

MEDIUM = ModelConfig(
    dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
)  # ~350M params