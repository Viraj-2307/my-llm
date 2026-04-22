"""
Qualitative generation evaluation.
Tests the model on a fixed set of prompts and saves outputs.
"""
import torch
import sentencepiece as spm
import json
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LLM


EVAL_PROMPTS = [
    # Factual
    "The capital of France is",
    "Water is made of",
    "The speed of light is",

    # Completion
    "Once upon a time there was a",
    "The scientist discovered a new",
    "In the year 2050, humans will",

    # Reasoning
    "The best way to learn programming is",
    "Artificial intelligence will change the world by",

    # Domain
    "Machine learning is a field of",
    "The transformer architecture was introduced in",
]


@torch.no_grad()
def run_generation_eval(
    model: LLM,
    tokenizer: spm.SentencePieceProcessor,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu",
) -> list[dict]:

    model.eval()
    results = []

    print(f"\nRunning generation eval on {len(EVAL_PROMPTS)} prompts...")
    print("=" * 60)

    for prompt in EVAL_PROMPTS:
        input_ids = [tokenizer.bos_id()] + \
                    tokenizer.encode(prompt, out_type=int)
        idx = torch.tensor([input_ids], dtype=torch.long).to(device)

        output = model.generate(
            idx,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
        )

        new_ids  = output[0, len(input_ids):].tolist()
        new_text = tokenizer.decode(new_ids)
        new_text = new_text.replace("⁇", "").strip()

        full_text = prompt + new_text

        results.append({
            "prompt":     prompt,
            "completion": new_text,
            "full":       full_text,
        })

        print(f"\nPrompt:     {prompt}")
        print(f"Completion: {new_text[:120]}...")

    print("\n" + "=" * 60)
    model.train()
    return results