"""Gradio app — Smart Keyboard: next-word prediction (SWE015 UC5).

As you type, three models suggest the top-3 next words side by side, each with its
response time: the classical n-gram, the feedforward neural LM, and the quantized
(INT8) GPT-2 that the app actually runs (R5). Click a suggestion to insert it.

    pip install torch transformers gradio nltk
    python app.py
Then open the local URL it prints (http://127.0.0.1:7860).
"""
import os
import re
import time
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import corpus

MODEL_ID = "tia88na/gpt2-keyboard"
SPECIAL = {"<s>", "</s>", "<unk>"}


# ---------- n-gram model (rebuilt from the corpus) ----------
print("Building n-gram tables...")
train_sents, _ = corpus.train_test_sentences()
_tri, _bi, _uni = defaultdict(Counter), defaultdict(Counter), Counter()
for s in train_sents:
    p = ["<s>", "<s>"] + s + ["</s>"]
    for i in range(2, len(p)):
        _tri[(p[i - 2], p[i - 1])][p[i]] += 1
        _bi[p[i - 1]][p[i]] += 1
        _uni[p[i]] += 1


VOCAB = {w for w in _uni if w not in SPECIAL and len(w) >= 2}


def ngram_top3(words):
    p = ["<s>", "<s>"] + words
    cand = _tri.get((p[-2], p[-1])) or _bi.get(p[-1]) or _uni
    out = [w for w, _ in cand.most_common(6) if w not in SPECIAL]
    return out[:3]


# ---------- feedforward neural LM (loaded from Area A) ----------
class FFLM(nn.Module):
    def __init__(self, vocab, emb=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.fc1 = nn.Linear(2 * emb, hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self.emb(x).view(x.size(0), -1))))


ffn = None
if not os.path.exists("model_a_ffn.pt"):
    print("model_a_ffn.pt not found — training the neural LM once (first run only)...")
    import model_a
    model_a.main()
try:
    print("Loading feedforward neural LM...")
    ck = torch.load("model_a_ffn.pt", map_location="cpu")
    ff_itos = ck["itos"]
    ff_stoi = {w: i for i, w in enumerate(ff_itos)}
    ffn = FFLM(len(ff_itos)).eval()
    ffn.load_state_dict(ck["state"])
except Exception as e:
    print("  (neural LM not found, skipping:", e, ")")


def ffn_top3(words):
    if ffn is None:
        return []
    p = ["<s>", "<s>"] + words
    ctx = torch.tensor([[ff_stoi.get(p[-2], 0), ff_stoi.get(p[-1], 0)]])
    with torch.no_grad():
        top = ffn(ctx)[0].topk(8).indices.tolist()
    return [ff_itos[i] for i in top if ff_itos[i] not in SPECIAL][:3]


# ---------- quantized GPT-2 (Area D) ----------
print("Loading + quantizing GPT-2 (first run downloads it)...")
gtok = GPT2TokenizerFast.from_pretrained(MODEL_ID)
gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_ID).eval()


def _conv1d_to_linear(module):
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            nx, nf = child.weight.shape
            lin = nn.Linear(nx, nf, bias=True)
            with torch.no_grad():
                lin.weight.copy_(child.weight.t().contiguous())
                lin.bias.copy_(child.bias)
            setattr(module, name, lin)
        else:
            _conv1d_to_linear(child)


_conv1d_to_linear(gpt2.transformer.h)
gpt2.transformer.h = torch.quantization.quantize_dynamic(
    gpt2.transformer.h, {nn.Linear}, dtype=torch.qint8)
print("Ready.")


def gpt2_top3(text):
    text = " ".join(text.split()[-40:])          # keep a fresh context window
    ids = gtok(text if text.strip() else " ", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = gpt2(ids).logits[0, -1]
    out = []
    for t in logits.topk(300).indices.tolist():
        dec = gtok.decode([t])
        if dec[:1] == " ":                       # only whole-word (word-start) tokens
            w = dec.strip().lower()
            if w in VOCAB and w not in out:      # keep only real corpus words
                out.append(w)
        if len(out) == 3:
            break
    return out


# ---------- UI ----------
def suggest(text):
    words = re.findall(r"[a-z']+", text.lower())
    results = []
    for fn, arg in [(gpt2_top3, text), (ngram_top3, words), (ffn_top3, words)]:
        t0 = time.perf_counter()
        preds = fn(arg)
        lat = (time.perf_counter() - t0) * 1000
        results.append(((preds + ["", "", ""])[:3], lat))
    flat = results[0][0] + results[1][0] + results[2][0]
    (g, gl), (n, nl), (f, fl) = results
    return (
        _label(0, results[0][1]),
        *[gr.update(value=w or "·", visible=bool(w)) for w in g],
        _label(1, results[1][1]),
        *[gr.update(value=w or "·", visible=bool(w)) for w in n],
        _label(2, results[2][1]),
        *[gr.update(value=w or "·", visible=bool(w)) for w in f],
        flat,
    )


METAS = [("GPT-2 INT8", "transformer · quantized", "{:.0f}", "#7c6cf0"),
         ("n-gram", "classical trigram", "{:.2f}", "#2bb3a3"),
         ("Neural LM", "feedforward net", "{:.2f}", "#e0a33e")]


def _label(i, lat):
    name, sub, fmt, color = METAS[i]
    return (f"<div class='mhead'><span class='mdot' style='background:{color}'></span>"
            f"<span class='mname'>{name}</span>"
            f"<span class='msub'>{sub}</span>"
            f"<span class='mlat'>{fmt.format(lat)} ms</span></div>")


def appender(idx):
    def _fn(cur, flat):
        w = flat[idx] if flat and idx < len(flat) else ""
        if not w:
            return cur
        sep = "" if (not cur or cur.endswith(" ")) else " "
        return cur + sep + w + " "
    return _fn


CSS = """
.gradio-container {max-width: 1080px !important; margin: auto;}
#kbtitle {text-align:center; margin-bottom:4px;}
#kbtitle h1 {margin-bottom:2px;}
#kbtitle p {color: var(--body-text-color-subdued); margin-top:0;}
.model-card {background: var(--block-background-fill);
  border:1px solid var(--border-color-primary); border-radius:16px; padding:16px;}
.mhead {display:flex; align-items:center; gap:8px; margin-bottom:12px;}
.mdot {width:10px; height:10px; border-radius:50%; display:inline-block; flex:none;}
.mname {font-weight:700; font-size:16px;}
.msub {color: var(--body-text-color-subdued); font-size:12px;}
.mlat {margin-left:auto; background: var(--primary-500); color:#fff;
  border-radius:999px; padding:2px 10px; font-size:12px; font-weight:600;}
.sugg button {border-radius:10px !important; font-weight:600 !important;}
"""


with gr.Blocks(title="Smart Keyboard",
               theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
    gr.HTML("<div id='kbtitle'><h1>⌨️ Smart Keyboard</h1>"
            "<p>Next-word prediction — three models, side by side. "
            "Type, and click a suggestion to insert it.</p></div>")
    msg = gr.Textbox(label="", lines=2, autofocus=True,
                     placeholder="start typing, e.g. the weather tomorrow will …")
    state = gr.State([""] * 9)
    cols = []
    with gr.Row(equal_height=True):
        for _ in range(3):
            with gr.Column(elem_classes="model-card"):
                lab = gr.HTML()
                btns = [gr.Button("·", elem_classes="sugg") for _ in range(3)]
                cols.append((lab, btns))

    all_btns = cols[0][1] + cols[1][1] + cols[2][1]
    outputs = []
    for lab, btns in cols:
        outputs += [lab, *btns]
    outputs.append(state)

    msg.change(suggest, msg, outputs, show_progress="hidden")
    for i, b in enumerate(all_btns):
        b.click(appender(i), [msg, state], msg, show_progress="hidden") \
         .then(suggest, msg, outputs, show_progress="hidden")

if __name__ == "__main__":
    demo.launch()
