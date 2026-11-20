"""Area B — classical baseline: trigram language models (next-word prediction).

Efficient, self-contained implementation (count-based) of BOTH add-one (Laplace)
and interpolated Kneser-Ney smoothing. Reports perplexity on the shared held-out
text, top-3 next-word accuracy and per-suggestion latency (R4). Runs in seconds.

    python ngram_baseline.py
"""
import math
import time
from collections import defaultdict, Counter

import corpus

D = 0.75  # Kneser-Ney discount


def count_ngrams(train_sents):
    uni, bi, tri = Counter(), Counter(), Counter()
    for s in train_sents:
        p = ["<s>", "<s>"] + s + ["</s>"]
        for w in p:
            uni[w] += 1
        for i in range(1, len(p)):
            bi[(p[i - 1], p[i])] += 1
        for i in range(2, len(p)):
            tri[(p[i - 2], p[i - 1], p[i])] += 1
    return uni, bi, tri


# ---------- add-one (Laplace) ----------
def laplace_perplexity(test_sents, bi, tri, V):
    logp, n = 0.0, 0
    for s in test_sents:
        p = ["<s>", "<s>"] + s + ["</s>"]
        for i in range(2, len(p)):
            c_tri = tri.get((p[i - 2], p[i - 1], p[i]), 0)
            c_bi = bi.get((p[i - 2], p[i - 1]), 0)
            logp += math.log((c_tri + 1) / (c_bi + V))
            n += 1
    return math.exp(-logp / n)


# ---------- interpolated Kneser-Ney ----------
def build_kn(bi, tri):
    follow = defaultdict(set)       # (w1,w2) -> {w3}
    pre = defaultdict(set)          # (w2,w3) -> {w1}
    for (w1, w2, w3) in tri:
        follow[(w1, w2)].add(w3)
        pre[(w2, w3)].add(w1)
    n1_follow_bi = {k: len(v) for k, v in follow.items()}
    cont_bi = {k: len(v) for k, v in pre.items()}          # N1+(.w2 w3)
    cont_ctx = defaultdict(int)                             # w2 -> sum cont_bi
    cont_follow = defaultdict(int)                          # w2 -> #distinct w3
    for (w2, w3), c in cont_bi.items():
        cont_ctx[w2] += c
        cont_follow[w2] += 1
    precede = defaultdict(set)                              # w -> {words before w}
    for (w1, w2) in bi:
        precede[w2].add(w1)
    n1_pre = {w: len(s) for w, s in precede.items()}
    total_bigrams = len(bi)
    return (n1_follow_bi, cont_bi, cont_ctx, cont_follow, n1_pre, total_bigrams)


def kn_prob(w1, w2, w3, bi, tri, kn):
    n1_follow_bi, cont_bi, cont_ctx, cont_follow, n1_pre, total_bi = kn
    p_cont = n1_pre.get(w3, 0) / total_bi if total_bi else 0
    p_cont = max(p_cont, 1e-10)

    denom2 = cont_ctx.get(w2, 0)                            # bigram (continuation) level
    if denom2 == 0:
        p_bi = p_cont
    else:
        c2 = cont_bi.get((w2, w3), 0)
        lam2 = D * cont_follow.get(w2, 0) / denom2
        p_bi = max(c2 - D, 0) / denom2 + lam2 * p_cont

    denom1 = bi.get((w1, w2), 0)                            # trigram level
    if denom1 == 0:
        return p_bi
    c1 = tri.get((w1, w2, w3), 0)
    lam1 = D * n1_follow_bi.get((w1, w2), 0) / denom1
    return max(c1 - D, 0) / denom1 + lam1 * p_bi


def kn_perplexity(test_sents, bi, tri, kn):
    logp, n = 0.0, 0
    for s in test_sents:
        p = ["<s>", "<s>"] + s + ["</s>"]
        for i in range(2, len(p)):
            prob = max(kn_prob(p[i - 2], p[i - 1], p[i], bi, tri, kn), 1e-10)
            logp += math.log(prob)
            n += 1
    return math.exp(-logp / n)


# ---------- top-3 next-word prediction (with stupid backoff) ----------
def build_predict_tables(train_sents):
    tri, bi, uni = defaultdict(Counter), defaultdict(Counter), Counter()
    for s in train_sents:
        p = ["<s>", "<s>"] + s + ["</s>"]
        for i in range(2, len(p)):
            tri[(p[i - 2], p[i - 1])][p[i]] += 1
            bi[p[i - 1]][p[i]] += 1
            uni[p[i]] += 1
    return tri, bi, uni


def top3(ctx, tri, bi, uni):
    cand = tri.get(ctx) or bi.get(ctx[1]) or uni
    return [w for w, _ in cand.most_common(3)]


def top3_accuracy(train_sents, test_sents):
    tri, bi, uni = build_predict_tables(train_sents)
    correct = total = 0
    for s in test_sents:
        p = ["<s>", "<s>"] + s + ["</s>"]
        for i in range(2, len(p)):
            total += 1
            correct += p[i] in top3((p[i - 2], p[i - 1]), tri, bi, uni)
    ctxs = [(s[j], s[j + 1]) for s in test_sents[:300] if len(s) > 2
            for j in range(len(s) - 2)][:500]
    t0 = time.perf_counter()
    for c in ctxs:
        top3(c, tri, bi, uni)
    latency = (time.perf_counter() - t0) / max(len(ctxs), 1) * 1000
    return correct / total, latency


def main():
    train_sents, test_sents = corpus.train_test_sentences()
    print(f"train {len(train_sents)} / test {len(test_sents)} sentences")

    uni, bi, tri = count_ngrams(train_sents)
    V = len(uni)
    print("Computing perplexities...")
    pp_laplace = laplace_perplexity(test_sents, bi, tri, V)
    kn = build_kn(bi, tri)
    pp_kn = kn_perplexity(test_sents, bi, tri, kn)
    acc, latency = top3_accuracy(train_sents, test_sents)

    print("\n=== Area B — trigram baseline ===")
    print(f"Perplexity (add-one):     {pp_laplace:.1f}")
    print(f"Perplexity (Kneser-Ney):  {pp_kn:.1f}")
    print(f"Top-3 next-word accuracy: {acc:.4f}")
    print(f"Latency:                  {latency:.4f} ms/suggestion (CPU)")


if __name__ == "__main__":
    main()
