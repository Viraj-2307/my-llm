"""
Entry point for instruction fine-tuning.
Usage:
  CPU:   python finetune/run_finetune.py --checkpoint checkpoints/ckpt_0000500.pt
  GPU:   python finetune/run_finetune.py --checkpoint checkpoints/ckpt_0000500.pt --device cuda
"""
import torch
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LLM
from config import SMALL
from finetune.lora import inject_lora, freeze_base_model
from finetune.dataset import InstructionDataset
from finetune.trainer import finetune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--tokenizer",   default="tokenizer/tokenizer.model")
    parser.add_argument("--rank",        type=int,   default=8)
    parser.add_argument("--alpha",       type=float, default=16.0)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--max_samples", type=int,   default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load base model ────────────────────────────────────────────────
    print(f"Loading base model from {args.checkpoint}...")
    ckpt  = torch.load(args.checkpoint, map_location=args.device,
                       weights_only=False)
    model = LLM(ckpt["model_config"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Base model loaded: {model.num_params():,} params")

    # ── Inject LoRA ────────────────────────────────────────────────────
    model = inject_lora(
        model,
        rank   = args.rank,
        alpha  = args.alpha,
        target_modules = ["wq", "wk", "wv", "wo"],
    )
    model = freeze_base_model(model)

    # ── Load instruction dataset ───────────────────────────────────────
    dataset = InstructionDataset(
        tokenizer_path = args.tokenizer,
        max_seq_len    = ckpt["model_config"].max_seq_len,
        max_samples    = args.max_samples,
    )

    # ── Fine-tune ──────────────────────────────────────────────────────
    model = finetune(
        model      = model,
        dataset    = dataset,
        learning_rate = args.lr,
        batch_size    = args.batch_size,
        num_epochs    = args.epochs,
        device        = args.device,
    )

    print("\nDone! LoRA weights saved to finetune/checkpoints/")
    print("To use the fine-tuned model, run:")
    print("  python serve.py --lora finetune/checkpoints/lora_epoch_3.pt")


if __name__ == "__main__":
    main()