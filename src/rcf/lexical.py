"""Language test bed: do the human norms of a two-word expression follow additively from its words?

Sources (all human ratings, no model-generated labels):
  NRC VAD Lexicon v2.1      valence / arousal / dominance for words and ~10k two-word expressions
  Brysbaert et al.          concreteness for ~37k words
  Muraki et al. (HF mirror) concreteness for ~62k multiword expressions, with per-item SD and N
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import SetData
from .embed import embed_texts

NRC = "data/raw/nrc_vad/NRC-VAD-Lexicon-v2.1"
CONC = "data/raw/concreteness"
# split-half reliability r reported for the whole lexicon (Mohammad 2025, Table 2); Spearman-Brown -> full-set reliability
NRC_SHR = dict(valence=0.99, arousal=0.98, dominance=0.96)


def load_norms(root: str = ".") -> dict[str, pd.DataFrame]:
    r = Path(root)
    uni = pd.read_csv(r / NRC / "Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt", sep="\t", keep_default_na=False)
    mwe = pd.read_csv(r / NRC / "MWE/mwe-NRC-VAD-Lexicon-v2.1.txt", sep="\t", keep_default_na=False)
    cw = pd.read_csv(r / CONC / "concreteness_ratings.csv", keep_default_na=False)
    cw = cw[cw.Bigram == 0].assign(term=lambda d: d.Word.str.lower())[["term", "Conc.M"]].drop_duplicates("term")
    cp = pd.read_json(r / CONC / "concreteness_phrases.jsonl", lines=True)
    cp = cp.assign(term=cp.Expression.str.lower()).drop_duplicates("term")
    return dict(vad_words=uni, vad_mwe=mwe, conc_words=cw, conc_phrases=cp)


def bigram_dataset(target: str, features: tuple[str, ...], norms: dict | None = None, require_norms: bool = False,
                   root: str = ".") -> tuple[SetData, pd.DataFrame]:
    """target in {"vad", "conc"}.  features: any of "glove", "minilm", "norms"; word position is always appended.

    Each word is one object.  "norms" gives the word's own human V, A, D and concreteness, with
    a missing-indicator per norm, i.e. the human one-body potentials themselves.
    """
    norms = norms or load_norms(root)
    vw = norms["vad_words"].set_index("term")
    cw = norms["conc_words"].set_index("term")["Conc.M"]
    if target == "vad":
        df = norms["vad_mwe"].copy()
        ycols = ["valence", "arousal", "dominance"]
        noise = None
    else:
        df = norms["conc_phrases"].rename(columns={"Mean_C": "concreteness"}).copy()
        ycols = ["concreteness"]
    w = df.term.str.split(" ")
    df = df[w.str.len() == 2].copy()
    df["w1"], df["w2"] = df.term.str.split(" ").str[0], df.term.str.split(" ").str[1]
    df["rated"] = df.w1.isin(vw.index) & df.w2.isin(vw.index)
    if require_norms:
        df = df[df.rated]
    df = df.reset_index(drop=True)
    vocab = sorted(set(df.w1) | set(df.w2))
    vid = {t: i for i, t in enumerate(vocab)}
    blocks = []
    for f in features:
        if f == "norms":
            V = vw.reindex(vocab)[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
            Cn = ((cw.reindex(vocab).to_numpy(dtype=np.float32) - 3.0) / 2.0)[:, None]
            N = np.concatenate([V, Cn], 1)
            miss = np.isnan(N).astype(np.float32)[:, [0, 3]]
            blocks.append(np.concatenate([np.nan_to_num(N), miss], 1))
        else:
            blocks.append(embed_texts(vocab, f, cache_dir=str(Path(root) / "data/cache")))
    E = np.concatenate(blocks, 1)
    i1, i2 = df.w1.map(vid).to_numpy(), df.w2.map(vid).to_numpy()
    pos = np.eye(2, dtype=np.float32)
    X = np.stack([np.concatenate([E[i1], np.tile(pos[0], (len(df), 1))], 1),
                  np.concatenate([E[i2], np.tile(pos[1], (len(df), 1))], 1)], 1).astype(np.float32)
    y = df[ycols].to_numpy(dtype=np.float32)
    noise = (df.SD_C.to_numpy() ** 2 / df.N_Rate.to_numpy())[:, None].astype(np.float32) if target == "conc" else None
    data = SetData(X, np.ones((len(df), 2), dtype=bool), y, "reg", ycols, df.term.to_numpy(), None, noise)
    return data, df


def whole_phrase_dataset(df: pd.DataFrame, y: np.ndarray, model: str, names: list[str], root: str = ".") -> SetData:
    """Reference: the expression embedded as one string (the encoder composes the words itself)."""
    E = embed_texts(df.term.tolist(), model, cache_dir=str(Path(root) / "data/cache"))
    return SetData(E[:, None, :], np.ones((len(df), 1), dtype=bool), y, "reg", names, df.term.to_numpy())


def ceiling(target: str, data: SetData) -> np.ndarray:
    """Irreducible held-out MSE per node, in units of label variance."""
    if target == "conc":
        return (data.noise_var.mean(0) / data.y.var(0)).astype(np.float64)
    r = np.array([NRC_SHR[n] for n in data.names])
    return 1.0 - 2 * r / (1 + r)
