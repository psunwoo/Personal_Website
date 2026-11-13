"""Stanford Sentiment Treebank: human sentiment for every node of every parse tree.

Words, phrases and sentences all carry a human rating in [0, 1], so the same data gives
  - object sets with a label (the words of a phrase -> the phrase's sentiment), at sizes 2..N;
  - human one-body potentials (each word's own rating);
  - a real hierarchy in scale: parent node vs its two children, all human-rated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import SetData, pad_sets
from .embed import embed_texts

RAW = "data/raw/sst"


def _spans(n_tok: int, parents: list[int]) -> tuple[list[tuple[int, int]], dict[int, list[int]]]:
    """Token span of every node; nodes 1..n_tok are leaves, parent 0 marks the root."""
    m = len(parents)
    lo, hi = [10**9] * (m + 1), [-1] * (m + 1)
    kids: dict[int, list[int]] = {}
    for i in range(1, n_tok + 1):
        lo[i] = hi[i] = i - 1
    for c, p in enumerate(parents, start=1):
        if p:
            kids.setdefault(p, []).append(c)
    for node in range(n_tok + 1, m + 1):  # internal nodes are numbered bottom-up
        for c in kids.get(node, []):
            lo[node], hi[node] = min(lo[node], lo[c]), max(hi[node], hi[c])
    return [(lo[i], hi[i]) for i in range(1, m + 1)], kids


def load(root: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """nodes: one row per tree node (split, sentence id, phrase, label, size, children's phrases/labels)."""
    r = Path(root) / RAW
    dic = pd.read_parquet(r / "dictionary_dictionary.parquet").drop_duplicates("phrase").set_index("phrase")["label"]
    rows = []
    sid = 0
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(r / f"default_{split}.parquet")
        for toks, tree in zip(df.tokens, df.tree):
            t = toks.split("|")
            par = [int(x) for x in tree.split("|")]
            spans, kids = _spans(len(t), par)
            text = [" ".join(t[a:b + 1]) for a, b in spans]
            for node, (a, b) in enumerate(spans, start=1):
                ch = sorted(kids.get(node, []), key=lambda c: spans[c - 1][0])
                rows.append((split, sid, text[node - 1], b - a + 1,
                             text[ch[0] - 1] if len(ch) == 2 else None, text[ch[1] - 1] if len(ch) == 2 else None))
            sid += 1
    nodes = pd.DataFrame(rows, columns=["split", "sid", "phrase", "size", "left", "right"])
    nodes["label"] = nodes.phrase.map(dic)
    nodes["y_left"], nodes["y_right"] = nodes.left.map(dic), nodes.right.map(dic)
    return nodes, dic.reset_index()


def phrase_table(nodes: pd.DataFrame, min_size: int = 2, max_size: int = 12) -> pd.DataFrame:
    """Unique phrases with a leak-free split: anything seen in a test (val) sentence is test (val) only."""
    p = nodes[(nodes["size"] >= min_size) & (nodes["size"] <= max_size) & nodes.label.notna()]
    rank = {"test": 0, "validation": 1, "train": 2}
    p = p.assign(r=p.split.map(rank)).sort_values("r").drop_duplicates("phrase")
    return p[["phrase", "size", "label", "split"]].reset_index(drop=True)


def phrase_sets(tab: pd.DataFrame, dic: pd.DataFrame, features: tuple[str, ...] = ("minilm", "wordsent"),
                max_size: int = 12) -> SetData:
    """One object per token: [isolated embedding] [the word's own human sentiment, missing flag] [relative position]."""
    toks = [p.split(" ") for p in tab.phrase]
    vocab = sorted({w for t in toks for w in t})
    vid = {w: i for i, w in enumerate(vocab)}
    blocks = []
    for f in features:
        if f == "wordsent":
            s = dic.set_index("phrase")["label"].reindex(vocab).to_numpy(dtype=np.float32)
            blocks.append(np.stack([np.nan_to_num(s, nan=0.5) - 0.5, np.isnan(s).astype(np.float32)], 1))
        else:
            blocks.append(embed_texts([w.lower() for w in vocab], f))
    E = np.concatenate(blocks, 1)
    seqs = []
    for t in toks:
        n = len(t)
        pos = np.arange(n, dtype=np.float32) / max(n - 1, 1)
        seqs.append(np.concatenate([E[[vid[w] for w in t]], pos[:, None], (1 - pos)[:, None]], 1))
    X, M = pad_sets(seqs, max_size)
    return SetData(X, M, tab.label.to_numpy(dtype=np.float32)[:, None], "reg", ["sentiment"], tab.phrase.to_numpy())
