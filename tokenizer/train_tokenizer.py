import sentencepiece as spm
import os
from datasets import load_dataset
from pathlib import Path

def dump_corpus_for_tokenizer(
    output_file: str = "tokenizer/corpus.txt",
    num_samples: int = 500_000,
    dataset_name: str = "HuggingFaceFW/fineweb",
):
    """Stream samples from dataset and write raw text to disk."""
    Path("tokenizer").mkdir(exist_ok=True)
    print(f"Streaming {num_samples} samples from {dataset_name}...")

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            text = sample.get("text", "").strip()
            if len(text) > 100:               # skip very short docs
                f.write(text[:2048] + "\n")   # cap per-doc length

    print(f"Corpus written to {output_file}")


def train_tokenizer(
    corpus_file: str = "tokenizer/corpus.txt",
    output_prefix: str = "tokenizer/tokenizer",
    vocab_size: int = 32000,
):
    """Train BPE tokenizer using SentencePiece."""
    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",

        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",

        # Coverage and normalization
        character_coverage=0.9995,   # covers most Unicode
        normalization_rule_name="identity",  # no NFKC — preserve original text
        add_dummy_prefix=True,               # adds space prefix like LLaMA

        # Training efficiency
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
    )

    print(f"Tokenizer saved: {output_prefix}.model and {output_prefix}.vocab")


def test_tokenizer(model_path: str = "tokenizer/tokenizer.model"):
    """Quick sanity check on the trained tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    test_sentences = [
        "Hello, world!",
        "The transformer architecture revolutionized NLP.",
        "def forward(self, x): return self.linear(x)",
    ]

    print("\nTokenizer sanity check:")
    for s in test_sentences:
        tokens = sp.encode(s, out_type=str)
        ids = sp.encode(s, out_type=int)
        decoded = sp.decode(ids)
        print(f"\n  Input:   {s}")
        print(f"  Tokens:  {tokens}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {decoded}")
        assert decoded == s or decoded == " " + s, "Decode mismatch!"

    print(f"\nVocab size: {sp.vocab_size()}")
    print(f"BOS id: {sp.bos_id()}, EOS id: {sp.eos_id()}, PAD id: {sp.pad_id()}")


if __name__ == "__main__":
    dump_corpus_for_tokenizer()
    train_tokenizer()
    test_tokenizer()