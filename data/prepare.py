"""
Run once before training to pre-tokenize and shard the dataset.
Usage: python data/prepare.py
"""
import os
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

def prepare_dataset(
    tokenizer_path: str = "tokenizer/tokenizer.model",
    dataset_name: str = "HuggingFaceFW/fineweb",
    output_dir: str = "data/shards",
    shard_size: int = 100_000_000,   # 100M tokens per shard
    max_shards: int = 10,
    split: str = "train",
    val_fraction: float = 0.005,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    print(f"Loaded tokenizer: vocab_size={sp.vocab_size()}")

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    shard_idx = 0
    token_buffer = []
    total_tokens = 0
    val_tokens = []

    def flush_shard(tokens, name):
        arr = np.array(tokens, dtype=np.uint16)
        path = os.path.join(output_dir, f"{name}_{shard_idx:04d}.bin")
        arr.tofile(path)
        print(f"  Saved {name} shard {shard_idx}: {len(tokens):,} tokens → {path}")

    print(f"Processing dataset, max {max_shards} shards of {shard_size:,} tokens each...")

    for sample in tqdm(dataset):
        text = sample.get("text", "").strip()
        if len(text) < 50:
            continue

        ids = [sp.bos_id()] + sp.encode(text, out_type=int) + [sp.eos_id()]

        # Small fraction goes to validation
        if len(val_tokens) < shard_size * val_fraction:
            val_tokens.extend(ids)
        else:
            token_buffer.extend(ids)

        if len(token_buffer) >= shard_size:
            flush_shard(token_buffer[:shard_size], "train")
            token_buffer = token_buffer[shard_size:]
            shard_idx += 1
            total_tokens += shard_size

            if shard_idx >= max_shards:
                print(f"Reached max_shards={max_shards}, stopping.")
                break

    # Flush remaining
    if token_buffer:
        flush_shard(token_buffer, "train")
        total_tokens += len(token_buffer)

    if val_tokens:
        shard_idx = 0
        flush_shard(val_tokens, "val")

    print(f"\nDone. Total tokens: {total_tokens:,}")
    print(f"Shards saved to: {output_dir}/")


if __name__ == "__main__":
    prepare_dataset()