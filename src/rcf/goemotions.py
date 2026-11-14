"""GoEmotions (Demszky et al. 2020, Apache-2.0): Reddit comments, 27 emotions + neutral, with rater ids.

Used for two things:
  - a fourth language test bed whose target is a distribution over emotion categories;
  - the only open data here with rater identities, so per-user couplings can be tested.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SetData, pad_sets
from .embed import embed_texts
from .lexical import load_norms

RAW = "data/raw/goemotions/raw.parquet"
META = ["text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"]
TOKEN = re.compile(r"n't|[a-z0-9]+|[!?]")


def tokenise(text: str) -> list[str]:
    return TOKEN.findall(text.lower().replace("n't", " n't").replace("’", "'"))


def load(root: str = ".") -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    """comments: one row per comment (tokens, split); ratings: one row per (comment, rater) with a label distribution."""
    d = pd.read_parquet(Path(root) / RAW)
    emo = [c for c in d.columns if c not in META]
    Y = d[emo].to_numpy(dtype=np.float32)
    keep = Y.sum(1) > 0
    d, Y = d[keep].reset_index(drop=True), Y[keep]
    Y = Y / Y.sum(1, keepdims=True)
    com = d.drop_duplicates("id")[["id", "text"]].reset_index(drop=True)
    com["tokens"] = com.text.map(tokenise)
    com = com[com.tokens.str.len() >= 2].reset_index(drop=True)
    r = np.random.default_rng(0).random(len(com))
    com["split"] = np.where(r < 0.8, "train", np.where(r < 0.9, "val", "test"))
    cid = {c: i for i, c in enumerate(com.id)}
    ok = d.id.isin(cid)
    rat = pd.DataFrame(dict(comment=d.id[ok].map(cid).to_numpy(), rater=d.rater_id[ok].to_numpy()))
    return com, rat, emo, Y[ok.to_numpy()]


def comment_sets(com: pd.DataFrame, rat: pd.DataFrame, Y: np.ndarray, names: list[str], L: int = 24,
                 rows: np.ndarray | None = None) -> SetData:
    """One object per token: [isolated MiniLM embedding, human V/A/D of the word + missing flag, relative position x2].
    Target: the comment's label distribution averaged over its raters.
    rows: build only these comments (the full array needs about 2 GB)."""
    vocab = sorted({w for t in com.tokens for w in t})
    vid = {w: i for i, w in enumerate(vocab)}
    E = embed_texts(vocab, "minilm")
    vad = load_norms()["vad_words"].set_index("term").reindex(vocab)[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
    N = np.concatenate([np.nan_to_num(vad), np.isnan(vad[:, :1]).astype(np.float32)], 1)
    F = np.concatenate([E, N], 1)
    rows = np.arange(len(com)) if rows is None else rows
    seqs = []
    for t in com.tokens.iloc[rows]:
        t = t[:L]
        n = len(t)
        pos = np.arange(n, dtype=np.float32) / max(n - 1, 1)
        seqs.append(np.concatenate([F[[vid[w] for w in t]], pos[:, None], (1 - pos)[:, None]], 1))
    X, M = pad_sets(seqs, L)
    tgt = np.zeros((len(com), Y.shape[1]), dtype=np.float64)
    cnt = np.zeros(len(com))
    np.add.at(tgt, rat.comment.to_numpy(), Y)
    np.add.at(cnt, rat.comment.to_numpy(), 1)
    return SetData(X, M, (tgt / cnt[:, None]).astype(np.float32)[rows], "dist", names, com.id.to_numpy()[rows])
