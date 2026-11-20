"""Area D — dynamic INT8 quantization of the fine-tuned GPT-2.

GPT-2 uses `Conv1D` layers (not `nn.Linear`), which dynamic quantization does not
support directly. We first convert the transformer blocks' Conv1D layers to the
mathematically-equivalent nn.Linear, then apply dynamic INT8 quantization to those
blocks (leaving the embeddings and the tied output head in FP32 to preserve
accuracy). Compares word-level perplexity, top-3 accuracy, size and CPU latency
against the FP32 model on the same held-out text (R3/R4).

    pip install torch transformers nltk
    python quantize_gpt2.py
"""
import os
import math
import time
import random
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import corpus

MODEL_ID = "tia88na/gpt2-keyboard"


def conv1d_to_linear(module):
    """Recursively replace transformers Conv1D with an equivalent nn.Linear."""
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            nx, nf = child.weight.shape          # (in, out)
            lin = nn.Linear(nx, nf, bias=True)
            with torch.no_grad():
                lin.weight.copy_(child.weight.t().contiguous())
                lin.bias.copy_(child.bias)
            setattr(module, name, lin)
        else:
            conv1d_to_linear(child)


def word_perplexity(model, tok, test_text):
    ids = tok(test_text, return_tensors="pt").input_ids[0]
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(ids) - 1, 512):
            chunk = ids[i:i + 512].unsqueeze(0)
            out = model(chunk, labels=chunk)
            nll += out.loss.item() * (chunk.size(1) - 1)
    return math.exp(nll / len(test_text.split()))


def top3_and_latency(model, tok, test_sents, n=300):
    samples = []
    for s in test_sents:
        for i in range(1, len(s)):
            samples.append((s[max(0, i - 20):i], s[i]))
    random.seed(0)
    random.shuffle(samples)
    samples = samples[:n]
    correct = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for ctx_words, target in samples:
            ids = tok(" ".join(ctx_words), return_tensors="pt").input_ids
            logits = model(ids).logits[0, -1]
            preds = [tok.decode([t]).strip().lower()
                     for t in logits.topk(3).indices.tolist()]
            correct += target in preds
    latency = (time.perf_counter() - t0) / len(samples) * 1000
    return correct / len(samples), latency


def report(model, tok, test_text, test_sents, label):
    ppl = word_perplexity(model, tok, test_text)
    acc, latency = top3_and_latency(model, tok, test_sents)
    print(f"\n=== {label} ===")
    print(f"Word-level perplexity:    {ppl:.1f}")
    print(f"Top-3 next-word accuracy: {acc:.4f}")
    print(f"Latency:                  {latency:.1f} ms/suggestion (CPU)")


def size_mb(model, path):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1e6
    return mb


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _, test_sents = corpus.train_test_sentences()
    test_text = " ".join(" ".join(s) for s in test_sents)

    tok = GPT2TokenizerFast.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).eval()

    fp32_mb = size_mb(model, "gpt2_fp32.pt")
    report(model, tok, test_text, test_sents, f"GPT-2 FP32 ({fp32_mb:.0f} MB)")

    # convert the transformer blocks' Conv1D -> Linear, then quantize just them
    conv1d_to_linear(model.transformer.h)
    model.transformer.h = torch.quantization.quantize_dynamic(
        model.transformer.h, {nn.Linear}, dtype=torch.qint8)

    int8_mb = size_mb(model, "gpt2_int8.pt")
    report(model, tok, test_text, test_sents, f"GPT-2 INT8 ({int8_mb:.0f} MB)")

    os.remove("gpt2_fp32.pt")
    print("\n--- size comparison ---")
    print(f"FP32: {fp32_mb:.0f} MB  ->  INT8: {int8_mb:.0f} MB "
          f"({fp32_mb / int8_mb:.1f}x smaller)")
    print("Saved quantized weights to gpt2_int8.pt")


if __name__ == "__main__":
    main()
