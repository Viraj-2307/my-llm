"""
Master evaluation script.
Usage: python eval/run_eval.py --checkpoint checkpoints/ckpt_0000500.pt
"""
import torch
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LLM
from config import SMALL
import sentencepiece as spm

from perplexity      import compute_perplexity
from benchmarks      import eval_hellaswag, eval_arc_easy
from generation_eval import run_generation_eval


def load_model(checkpoint_path: str, device: str) -> LLM:
    print(f"Loading model from {checkpoint_path}...")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["model_config"]
    model  = LLM(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded: {model.num_params():,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--data_dir",    default="data/shards")
    parser.add_argument("--tokenizer",   default="tokenizer/tokenizer.model")
    parser.add_argument("--skip_bench",  action="store_true",
                        help="Skip HellaSwag/ARC (faster eval)")
    args = parser.parse_args()

    # ── Load model + tokenizer ─────────────────────────────────────────
    model     = load_model(args.checkpoint, args.device)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer)

    results = {}

    # ── 1. Perplexity ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("1. PERPLEXITY")
    print("="*60)
    ppl_results = compute_perplexity(
        model    = model,
        data_dir = args.data_dir,
        device   = args.device,
    )
    results["perplexity"] = ppl_results
    print(f"\nPerplexity: {ppl_results['perplexity']}")
    print(f"Avg loss:   {ppl_results['avg_loss']}")

    # ── 2. Benchmarks ──────────────────────────────────────────────────
    if not args.skip_bench:
        print("\n" + "="*60)
        print("2. BENCHMARKS")
        print("="*60)

        hellaswag = eval_hellaswag(model, tokenizer,
                                   num_samples=100, device=args.device)
        arc_easy  = eval_arc_easy(model, tokenizer,
                                   num_samples=100, device=args.device)

        results["hellaswag"] = hellaswag
        results["arc_easy"]  = arc_easy

    # ── 3. Generation quality ──────────────────────────────────────────
    print("\n" + "="*60)
    print("3. GENERATION QUALITY")
    print("="*60)
    gen_results = run_generation_eval(
        model     = model,
        tokenizer = tokenizer,
        device    = args.device,
    )
    results["generation"] = gen_results

    # ── Final report ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Parameters:  {model.num_params():,}")
    print(f"Perplexity:  {results['perplexity']['perplexity']}")
    print(f"Avg Loss:    {results['perplexity']['avg_loss']}")

    if not args.skip_bench:
        print(f"HellaSwag:   {results['hellaswag']['accuracy']}% "
              f"(random={results['hellaswag']['random_baseline']}%)")
        print(f"ARC-Easy:    {results['arc_easy']['accuracy']}% "
              f"(random={results['arc_easy']['random_baseline']}%)")

    # Save report
    Path("eval/reports").mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"eval/reports/eval_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()