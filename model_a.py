"""Area A — feedforward neural language model (next-word prediction).

A fixed-context neural LM: the previous two words -> embeddings -> hidden layer
-> softmax over the vocabulary. Trained from scratch with SGD and backpropagation.
Reports perplexity, top-3 next-word accuracy and latency on the same held-out
text (R3/R4).

    pip install torch
    python model_a.py
"""
import math
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import corpus

VOCAB_SIZE = 10000
EMB = 64
HIDDEN = 128
EPOCHS = 5
BATCH = 1024
LR = 0.3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def build_vocab(train_sents):
    freq = Counter(w for s in train_sents for w in s)
    itos = ["<unk>", "<s>", "</s>"] + [w for w, _ in freq.most_common(VOCAB_SIZE)]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def make_examples(sents, stoi):
    unk = stoi["<unk>"]
    ctx, tgt = [], []
    for s in sents:
        p = ["<s>", "<s>"] + s + ["</s>"]
        ids = [stoi.get(w, unk) for w in p]
        for i in range(2, len(ids)):
            ctx.append((ids[i - 2], ids[i - 1]))
            tgt.append(ids[i])
    return torch.tensor(ctx), torch.tensor(tgt)


class FFLM(nn.Module):
    def __init__(self, vocab, emb=EMB, hidden=HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.fc1 = nn.Linear(2 * emb, hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, x):                      # x: (B, 2)
        e = self.emb(x).view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(e)))


def main():
    train_sents, test_sents = corpus.train_test_sentences()
    itos, stoi = build_vocab(train_sents)
    Xtr, ytr = make_examples(train_sents, stoi)
    Xte, yte = make_examples(test_sents, stoi)
    print(f"vocab {len(itos)} | train ex {len(Xtr)} | test ex {len(Xte)}")

    model = FFLM(len(itos))
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    n = len(Xtr)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch+1}/{EPOCHS}  train loss {total/n:.4f}")

    # ---- evaluation on held-out ----
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        correct = 0
        for i in range(0, len(Xte), 4096):
            logits = model(Xte[i:i + 4096])
            y = yte[i:i + 4096]
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            top3 = logits.topk(3, dim=-1).indices
            correct += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
    ppl = math.exp(loss_sum / len(yte))
    acc = correct / len(yte)

    # ---- latency per single suggestion ----
    sample = Xte[:300]
    t0 = time.perf_counter()
    with torch.no_grad():
        for j in range(len(sample)):
            model(sample[j:j + 1]).topk(3, dim=-1)
    latency = (time.perf_counter() - t0) / len(sample) * 1000

    torch.save({"state": model.state_dict(), "itos": itos}, "model_a_ffn.pt")
    print("\n=== Area A — feedforward neural LM ===")
    print(f"Perplexity:               {ppl:.1f}")
    print(f"Top-3 next-word accuracy: {acc:.4f}")
    print(f"Latency:                  {latency:.4f} ms/suggestion (CPU)")


if __name__ == "__main__":
    main()
