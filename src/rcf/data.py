"""Common container for 'a set of objects -> concept activations' datasets."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SetData:
    """n sets of up to L objects with D features each.

    task = "dist": y rows are probability vectors over C concept nodes.
    task = "reg":  y rows are real-valued concept activations.
    """

    X: np.ndarray  # (n, L, D) float32, zero padded
    mask: np.ndarray  # (n, L) bool, True where an object is present
    y: np.ndarray  # (n, C) float32
    task: str
    names: list[str] = field(default_factory=list)  # concept node names
    ids: np.ndarray | None = None
    groups: np.ndarray | None = None  # for grouped splits / cluster bootstrap
    noise_var: np.ndarray | None = None  # (n, C) variance of the label mean (reg)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, idx: np.ndarray) -> "SetData":
        pick = lambda a: None if a is None else a[idx]
        return SetData(self.X[idx], self.mask[idx], self.y[idx], self.task, self.names,
                       pick(self.ids), pick(self.groups), pick(self.noise_var))


def pad_sets(seqs: list[np.ndarray], L: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """List of (n_i, D) arrays -> padded (n, L, D) and mask (n, L)."""
    L = L or max(len(s) for s in seqs)
    D = seqs[0].shape[1]
    X = np.zeros((len(seqs), L, D), dtype=np.float32)
    M = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        k = min(len(s), L)
        X[i, :k] = s[:k]
        M[i, :k] = True
    return X, M


def kfold(n: int, k: int, seed: int = 0, groups: np.ndarray | None = None) -> np.ndarray:
    """Fold index per item; items sharing a group share a fold."""
    rng = np.random.default_rng(seed)
    if groups is None:
        f = np.arange(n) % k
        rng.shuffle(f)
        return f
    ug, inv = np.unique(groups, return_inverse=True)
    gf = np.arange(len(ug)) % k
    rng.shuffle(gf)
    return gf[inv]
