import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine LR schedule with linear warmup.

    Phase 1 (0 → warmup_steps):     Linear ramp from 0 to max_lr
    Phase 2 (warmup → max_steps):   Cosine decay from max_lr to min_lr
    Phase 3 (> max_steps):          Hold at min_lr
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # After decay
    if step > max_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
    return min_lr + coeff * (max_lr - min_lr)