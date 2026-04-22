import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from typing import Iterator
import sentencepiece as spm


class PackedDataset(Dataset):
    """
    Memory-mapped dataset over pre-tokenized .bin shard files.
    Each shard is a flat uint16 array of token IDs.
    Sequences are packed to max_seq_len with no padding.
    """
    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 1024,
        split: str = "train",
    ):
        self.max_seq_len = max_seq_len
        self.data_dir = Path(data_dir)

        pattern = f"{split}_*.bin"
        self.shards = sorted(self.data_dir.glob(pattern))
        assert len(self.shards) > 0, \
            f"No shards found at {data_dir}/{pattern}. Run data/prepare.py first."

        # Memory-map all shards, compute total length
        self._mmap = []
        self._lengths = []
        for shard in self.shards:
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            self._mmap.append(m)
            # Each sample is max_seq_len+1 tokens (input + target shifted by 1)
            self._lengths.append(len(m) // (max_seq_len + 1))

        self._cumlen = np.cumsum([0] + self._lengths)
        print(f"[Dataset] {split}: {len(self)} samples across {len(self.shards)} shards")

    def __len__(self) -> int:
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int):
        # Find which shard this index falls in
        shard_idx = np.searchsorted(self._cumlen[1:], idx, side="right")
        local_idx = idx - int(self._cumlen[shard_idx])

        start = local_idx * (self.max_seq_len + 1)
        chunk = self._mmap[shard_idx][start : start + self.max_seq_len + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))

        x = chunk[:self.max_seq_len]    # input tokens
        y = chunk[1:self.max_seq_len+1] # target tokens (shifted by 1)
        return x, y


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for very large corpora.
    Tokenizes on-the-fly from HuggingFace datasets.
    Use this for initial experiments before pre-tokenizing.
    """
    def __init__(
        self,
        tokenizer_path: str,
        dataset_name: str = "HuggingFaceFW/fineweb",
        max_seq_len: int = 1024,
        seed: int = 42,
    ):
        from datasets import load_dataset
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10_000)
        self.max_seq_len = max_seq_len
        self.buffer: list[int] = []

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for sample in self.dataset:
            text = sample.get("text", "").strip()
            if not text:
                continue

            # Encode with BOS/EOS
            ids = [self.sp.bos_id()] + \
                  self.sp.encode(text, out_type=int) + \
                  [self.sp.eos_id()]
            self.buffer.extend(ids)

            # Yield packed sequences when buffer is full
            while len(self.buffer) >= self.max_seq_len + 1:
                chunk = torch.tensor(
                    self.buffer[:self.max_seq_len + 1], dtype=torch.long
                )
                self.buffer = self.buffer[self.max_seq_len:]  # slide window
                yield chunk[:self.max_seq_len], chunk[1:self.max_seq_len+1]