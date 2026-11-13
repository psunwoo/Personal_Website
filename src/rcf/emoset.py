"""EmoSet-118K test bed: evoked emotion (8 Mikels classes, human labelled) of real images.

Two ways of turning an image into a set of objects, both strictly one-body
(each object is embedded with no access to the others):

  tiles       g x g grid of image tiles, each passed alone through DINOv2 ViT-S/14
  attributes  EmoSet's own attributes (scene, objects, facial expression, human
              action, brightness, colourfulness), each named item embedded alone
              as text

Whole-image embeddings (DINOv2, CLIP) are any-order references.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import SetData, pad_sets
from .embed import embed_texts

EMOTIONS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]
KEYS = ["d1", "d2", "d3", "d4", "clip", "label", "brightness", "colorfulness", "facial_expression", "human_action",
        "scene", "object", "image_id"]


def load(split: str, root: str = "data/features/emoset", keys: list[str] | None = None) -> dict[str, np.ndarray]:
    files = sorted(Path(root).glob(f"{split}-*.npz"))
    if not files:
        raise FileNotFoundError(f"no extracted shards for split {split!r} in {root}")
    keys = keys or KEYS
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def onehot(label: np.ndarray, C: int = 8) -> np.ndarray:
    return np.eye(C, dtype=np.float32)[label.astype(int)]


class Standardiser:
    """z-score (and optional PCA) fitted on training objects only."""

    def __init__(self, X: np.ndarray, pca: int = 0):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.P = None
        if pca and pca < X.shape[1]:
            Z = (X[:: max(len(X) // 200_000, 1)] - self.mu) / self.sd
            _, s, Vt = np.linalg.svd(Z, full_matrices=False)
            self.P = Vt[:pca].T.astype(np.float32)
            self.explained = float((s[:pca] ** 2).sum() / (s ** 2).sum())

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float32) - self.mu) / self.sd
        return Z if self.P is None else Z @ self.P


def tile_sets(feats: dict, grid: int, std: Standardiser) -> SetData:
    """One object per tile: [standardised tile embedding, one-hot tile position]."""
    T = std(feats[f"d{grid}"])  # n, g*g, d
    n, L, _ = T.shape
    pos = np.broadcast_to(np.eye(L, dtype=np.float32), (n, L, L))
    X = np.concatenate([T, pos], -1).astype(np.float32)
    return SetData(X, np.ones((n, L), dtype=bool), onehot(feats["label"]), "dist", EMOTIONS, feats["image_id"])


def whole_set(feats: dict, key: str, std: Standardiser) -> SetData:
    X = std(feats[key])[:, None, :].astype(np.float32)
    return SetData(X, np.ones((len(X), 1), dtype=bool), onehot(feats["label"]), "dist", EMOTIONS, feats["image_id"])


TYPES = ["scene", "object", "facial_expression", "human_action", "brightness", "colorfulness"]


def attribute_items(feats: dict) -> list[list[tuple[str, str, float]]]:
    """Per image: list of (type, name, value).  Missing attributes are simply absent from the set."""
    out = []
    for i in range(len(feats["label"])):
        it = []
        if feats["scene"][i]:
            it.append(("scene", str(feats["scene"][i]).replace("_", " "), 0.0))
        for o in json.loads(str(feats["object"][i]) or "[]") or []:
            it.append(("object", str(o).lower(), 0.0))
        if feats["facial_expression"][i]:
            it.append(("facial_expression", str(feats["facial_expression"][i]) + " face", 0.0))
        if feats["human_action"][i]:
            it.append(("human_action", str(feats["human_action"][i]).replace("_", " "), 0.0))
        for k in ("brightness", "colorfulness"):
            v = float(feats[k][i])
            if np.isfinite(v):
                it.append((k, k, v))
        out.append(it)
    return out


def attribute_sets(feats: dict, model: str = "minilm", L: int = 12, use: tuple[str, ...] = tuple(TYPES)) -> SetData:
    """One object per attribute: [isolated text embedding of its name, one-hot type, scalar value]."""
    items = [[t for t in it if t[0] in use] for it in attribute_items(feats)]
    vocab = sorted({t[1] for it in items for t in it})
    E = embed_texts(vocab, model)
    vid = {t: i for i, t in enumerate(vocab)}
    tid = {t: i for i, t in enumerate(TYPES)}
    seqs = []
    for it in items:
        a = np.zeros((max(len(it), 1), E.shape[1] + len(TYPES) + 1), dtype=np.float32)
        for j, (ty, name, val) in enumerate(it):
            a[j, : E.shape[1]] = E[vid[name]]
            a[j, E.shape[1] + tid[ty]] = 1.0
            a[j, -1] = val
        seqs.append(a if it else a[:0])
    X, M = pad_sets([s if len(s) else np.zeros((0, E.shape[1] + len(TYPES) + 1), np.float32) for s in seqs], L)
    return SetData(X, M, onehot(feats["label"]), "dist", EMOTIONS, feats["image_id"])
