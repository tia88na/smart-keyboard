"""Area B — subword tokenization: train a BPE tokenizer on the corpus.

Trains a Byte-Pair-Encoding tokenizer on the same corpus, saves it to
bpe_tokenizer.json, and prints the vocabulary size and an example tokenization.
The neural models can reuse this subword vocabulary.

    pip install tokenizers
    python train_bpe.py
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import corpus

VOCAB_SIZE = 8000


def main():
    text = corpus.load_text()
    # write a plain-text file for the trainer
    with open("corpus_plain.txt", "w", encoding="utf-8") as f:
        f.write(text.lower())

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE,
                         special_tokens=["[UNK]", "[PAD]", "<s>", "</s>"])
    tokenizer.train(["corpus_plain.txt"], trainer)
    tokenizer.save("bpe_tokenizer.json")

    example = "the weather tomorrow will be"
    enc = tokenizer.encode(example)
    print(f"BPE tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Example: {example!r}")
    print(f"  tokens: {enc.tokens}")
    print("Saved to bpe_tokenizer.json")


if __name__ == "__main__":
    main()
