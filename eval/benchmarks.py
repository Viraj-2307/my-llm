"""
Evaluate on HellaSwag and ARC-Easy benchmarks.
Both are multiple-choice — we pick the completion with lowest loss.
"""
import torch
import torch.nn.functional as F
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LLM
import sentencepiece as spm
from datasets import load_dataset


@torch.no_grad()
def score_completion(
    model: LLM,
    tokenizer: spm.SentencePieceProcessor,
    context: str,
    completion: str,
    device: str = "cpu",
) -> float:
    """
    Score a completion given a context.
    Returns average cross-entropy loss — lower = model prefers this completion.
    """
    ctx_ids  = tokenizer.encode(context,    out_type=int)
    comp_ids = tokenizer.encode(completion, out_type=int)

    # Full sequence: context + completion
    input_ids  = [tokenizer.bos_id()] + ctx_ids + comp_ids
    target_ids = [-1] * (len(ctx_ids) + 1) + comp_ids  # only score completion

    # Truncate if too long
    max_len = model.config.max_seq_len
    if len(input_ids) > max_len:
        input_ids  = input_ids[-max_len:]
        target_ids = target_ids[-max_len:]

    x = torch.tensor([input_ids],  dtype=torch.long).to(device)
    y = torch.tensor([target_ids], dtype=torch.long).to(device)

    _, loss = model(x, y)
    return loss.item()


@torch.no_grad()
def eval_hellaswag(
    model: LLM,
    tokenizer: spm.SentencePieceProcessor,
    num_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """
    HellaSwag: pick the most likely sentence completion from 4 choices.
    Random baseline = 25%. GPT-2 small = ~31%. Your target = ~27-30%.
    """
    print(f"\nEvaluating HellaSwag ({num_samples} samples)...")

    dataset = load_dataset(
        "Rowan/hellaswag", split="validation", streaming=True
    )

    correct = 0
    total   = 0

    for item in dataset:
        if total >= num_samples:
            break

        context = item["activity_label"] + ": " + item["ctx"]
        endings = item["endings"]
        label   = int(item["label"])

        # Score each ending
        scores = [
            score_completion(model, tokenizer, context, ending, device)
            for ending in endings
        ]

        predicted = scores.index(min(scores))  # lowest loss = most likely
        if predicted == label:
            correct += 1
        total += 1

        if total % 25 == 0:
            print(f"  {total}/{num_samples} | accuracy: {correct/total*100:.1f}%")

    accuracy = correct / total * 100

    return {
        "benchmark":  "HellaSwag",
        "accuracy":   round(accuracy, 2),
        "correct":    correct,
        "total":      total,
        "random_baseline": 25.0,
    }


@torch.no_grad()
def eval_arc_easy(
    model: LLM,
    tokenizer: spm.SentencePieceProcessor,
    num_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """
    ARC-Easy: multiple choice science questions.
    Random baseline = 25%. GPT-2 small = ~43%. Your target = ~30-35%.
    """
    print(f"\nEvaluating ARC-Easy ({num_samples} samples)...")

    dataset = load_dataset(
        "allenai/ai2_arc", "ARC-Easy",
        split="test", streaming=True
    )

    correct = 0
    total   = 0

    for item in dataset:
        if total >= num_samples:
            break

        question = item["question"]
        choices  = item["choices"]["text"]
        labels   = item["choices"]["label"]
        answer   = item["answerKey"]

        # Find correct index
        try:
            correct_idx = labels.index(answer)
        except ValueError:
            continue

        # Score each choice
        scores = [
            score_completion(model, tokenizer, question, choice, device)
            for choice in choices
        ]

        predicted = scores.index(min(scores))
        if predicted == correct_idx:
            correct += 1
        total += 1

        if total % 25 == 0:
            print(f"  {total}/{num_samples} | accuracy: {correct/total*100:.1f}%")

    accuracy = correct / total * 100

    return {
        "benchmark":  "ARC-Easy",
        "accuracy":   round(accuracy, 2),
        "correct":    correct,
        "total":      total,
        "random_baseline": 25.0,
    }