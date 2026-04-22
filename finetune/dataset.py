"""
Instruction fine-tuning dataset.
Supports Alpaca format: instruction + input + output
"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import sentencepiece as spm
from typing import Optional


PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_prompt(
    instruction: str,
    output: str,
    input: str = "",
    inference: bool = False,
) -> str:
    """
    Format a sample into the Alpaca prompt template.
    During inference, output is empty — model completes it.
    """
    if inference:
        output = ""

    if input.strip():
        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input,
            output=output,
        )
    else:
        prompt = PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output,
        )
    return prompt


class InstructionDataset(Dataset):
    """
    Loads Alpaca-format instruction data.
    Only computes loss on the response portion — not the prompt.
    """
    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int = 1024,
        dataset_name: str = "yahma/alpaca-cleaned",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.max_seq_len = max_seq_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        print(f"Loading dataset: {dataset_name}...")
        raw = load_dataset(dataset_name, split=split)

        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.samples = []
        skipped = 0

        for item in raw:
            instruction = item.get("instruction", "").strip()
            inp         = item.get("input", "").strip()
            output      = item.get("output", "").strip()

            if not instruction or not output:
                skipped += 1
                continue

            # Full prompt with response
            full_prompt = format_prompt(instruction, output, inp)

            # Prompt only (to know where response starts)
            prompt_only = format_prompt(instruction, "", inp, inference=True)

            # Tokenize both
            full_ids   = self.sp.encode(full_prompt,   out_type=int)
            prompt_ids = self.sp.encode(prompt_only,   out_type=int)

            # Truncate if too long
            if len(full_ids) > max_seq_len - 2:
                skipped += 1
                continue

            # Add BOS/EOS
            input_ids  = [self.sp.bos_id()] + full_ids + [self.sp.eos_id()]
            prompt_len = len(prompt_ids) + 1  # +1 for BOS

            # Target: -1 (ignore) for prompt, actual ids for response
            target_ids = [-1] * prompt_len + \
                         input_ids[prompt_len:]

            # Pad to max_seq_len
            pad_len   = max_seq_len - len(input_ids)
            input_ids  = input_ids  + [self.sp.pad_id()] * pad_len
            target_ids = target_ids + [-1]              * pad_len

            self.samples.append((
                torch.tensor(input_ids[:max_seq_len],  dtype=torch.long),
                torch.tensor(target_ids[:max_seq_len], dtype=torch.long),
            ))

        print(f"Dataset ready: {len(self.samples)} samples "
              f"(skipped {skipped})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]