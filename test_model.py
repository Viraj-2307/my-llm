# test_model.py
import torch
from config import ModelConfig, SMALL
from model import LLM

config = SMALL  # 120M params

model = LLM(config)

# Forward pass
B, T = 2, 64
x = torch.randint(0, config.vocab_size, (B, T))
y = torch.randint(0, config.vocab_size, (B, T))

logits, loss = model(x, y)
print(f"Logits shape: {logits.shape}")   # (2, 64, 32000)
print(f"Loss:         {loss.item():.4f}")  # ~10.4 (ln(32000))

# Generation test
prompt = torch.randint(0, config.vocab_size, (1, 10))
output = model.generate(prompt, max_new_tokens=20)
print(f"Generated:    {output.shape}")   # (1, 30)