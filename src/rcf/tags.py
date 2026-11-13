"""Language-derived object sets for images: zero-shot concept tags with human word norms.

Vocabulary: concrete, well-known English nouns (Brysbaert concreteness >= 4, known by >= 95 %)
that also have human valence/arousal/dominance norms (NRC VAD v2.1).  Each image receives its
top-k tags by CLIP image-text similarity.  Each tag is one object, described in isolation by

  "minilm"  a text embedding of the word alone
  "norms"   the word's human V, A, D and concreteness
  + the tag's rank-normalised CLIP score

Caveat (tracked as a confound): the tags come from a language-supervised model.  The labels and
the word norms are human.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import SetData
from .embed import embed_texts
from .emoset import EMOTIONS, onehot
from .lexical import load_norms

CLIP = "openai/clip-vit-base-patch32"


def vocabulary(min_freq: int = 100, root: str = ".") -> pd.DataFrame:
    cw = pd.read_csv(Path(root) / "data/raw/concreteness/concreteness_ratings.csv", keep_default_na=False)
    cw = cw[(cw.Bigram == 0) & (cw.Dom_Pos == "Noun") & (cw["Conc.M"] >= 4.0) & (cw.Percent_known >= 0.95)
            & (cw.SUBTLEX >= min_freq)]
    cw = cw.assign(term=cw.Word.str.lower())[["term", "Conc.M"]].drop_duplicates("term")
    vad = load_norms(root)["vad_words"]
    v = cw.merge(vad, on="term").rename(columns={"Conc.M": "concreteness"}).sort_values("term").reset_index(drop=True)
    return v


@torch.inference_mode()
def clip_text(words: list[str], cache: str = "data/cache/clip_text_vocab.npz") -> np.ndarray:
    p = Path(cache)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if list(z["words"]) == words:
            return z["E"]
    from transformers import CLIPModel, CLIPTokenizer

    tok, model = CLIPTokenizer.from_pretrained(CLIP), CLIPModel.from_pretrained(CLIP).eval()
    out = []
    for s in range(0, len(words), 256):
        t = tok([f"a photo of a {w}" for w in words[s:s + 256]], padding=True, return_tensors="pt")
        out.append(model.text_projection(model.text_model(**t).pooler_output).numpy())
    E = np.concatenate(out).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, words=np.array(words), E=E)
    return E


def tag_images(clip_img: np.ndarray, T: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Top-k vocabulary indices and cosine similarities per image."""
    I = clip_img.astype(np.float32)
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    idx, sc = [], []
    for s in range(0, len(I), 4096):
        S = I[s:s + 4096] @ Tn.T
        top = np.argpartition(-S, k, axis=1)[:, :k]
        ss = np.take_along_axis(S, top, 1)
        o = np.argsort(-ss, axis=1)
        idx.append(np.take_along_axis(top, o, 1))
        sc.append(np.take_along_axis(ss, o, 1))
    return np.concatenate(idx), np.concatenate(sc)


def tag_sets(feats: dict[str, dict], k: int = 8, features: tuple[str, ...] = ("minilm", "norms")) -> dict[str, SetData]:
    voc = vocabulary()
    words = voc.term.tolist()
    T = clip_text(words)
    blocks = []
    if "minilm" in features:
        blocks.append(embed_texts(words, "minilm"))
    if "norms" in features:
        N = voc[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
        blocks.append(np.concatenate([N, ((voc.concreteness.to_numpy(dtype=np.float32) - 4.5) * 2)[:, None]], 1))
    E = np.concatenate(blocks, 1)
    out = {}
    for s, f in feats.items():
        idx, sc = tag_images(f["clip"], T, k)
        z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-6)  # within-image, so no global score leaks
        X = np.concatenate([E[idx], z[..., None]], -1).astype(np.float32)
        d = SetData(X, np.ones(idx.shape, dtype=bool), onehot(f["label"]), "dist", EMOTIONS, f["image_id"])
        d.tag_idx = idx
        out[s] = d
    return out
