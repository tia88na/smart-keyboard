"""Corpus loader for the Smart Keyboard (SWE015 UC5).

Uses NLTK's built-in Project Gutenberg corpus (reliable download from the NLTK
data repo, no manual work). Provides a shared train / held-out split so every
model is evaluated on the same test text (R3). No `datasets` library required.
"""
import re
import nltk

# domain corpus = classic English literature (public domain), ~900k words
BOOKS = [
    "austen-emma.txt", "austen-sense.txt", "carroll-alice.txt",
    "melville-moby_dick.txt", "chesterton-thursday.txt", "edgeworth-parents.txt",
]


def _ensure():
    try:
        nltk.data.find("corpora/gutenberg")
    except LookupError:
        print("Downloading NLTK Gutenberg corpus (first run only)...")
        nltk.download("gutenberg", quiet=True)


def load_text():
    _ensure()
    from nltk.corpus import gutenberg
    return "\n".join(gutenberg.raw(b) for b in BOOKS)


def sentences():
    """Return a list of token lists (lowercased words), one per sentence."""
    text = load_text()
    out = []
    for raw in re.split(r"[.!?]+", text):
        words = re.findall(r"[a-z']+", raw.lower())
        if len(words) >= 2:
            out.append(words)
    return out


def train_test_sentences(test_frac=0.1):
    sents = sentences()
    k = int(len(sents) * (1 - test_frac))
    return sents[:k], sents[k:]


if __name__ == "__main__":
    tr, te = train_test_sentences()
    words = sum(len(s) for s in tr)
    print(f"train: {len(tr)} sentences / {words} words")
    print(f"test:  {len(te)} sentences")
