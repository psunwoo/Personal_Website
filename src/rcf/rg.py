"""Q2: a linear real-space-mutual-information coarse-graining and its relevance spectrum.

RSMI (Koch-Janusz & Ringel) keeps the block variables that carry the most
information about the block's environment.  In the linear-Gaussian case the
information-bottleneck solution is given by canonical correlation analysis: the
block directions ordered by decreasing canonical correlation rho_k with the
environment, with I_k = -1/2 log(1 - rho_k^2) nats carried by direction k.
The sequence rho_k^2 is the relevance spectrum.
"""
from __future__ import annotations

import numpy as np


def _whiten(X: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / len(X)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0, None) + reg * w.max()
    Wh = V / np.sqrt(w)  # columns: whitening directions
    return mu, Wh, Xc @ Wh


class CCA:
    """Regularised CCA.  proj_x(X)[:, k] is the k-th most environment-predictable block coordinate."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, reg: float = 1e-3):
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        self.mx, Wx, Xw = _whiten(X, reg)
        self.my, Wy, Yw = _whiten(Y, reg)
        U, s, Vt = np.linalg.svd(Xw.T @ Yw / len(X), full_matrices=False)
        self.rho = np.clip(s, 0, 1)
        self.A = Wx @ U  # x-side canonical directions (d_x, k)
        self.B = Wy @ Vt.T

    def proj_x(self, X: np.ndarray, k: int | None = None, bottom: bool = False) -> np.ndarray:
        A = self.A[:, -k:] if (bottom and k) else self.A[:, :k]
        return (X.astype(np.float64) - self.mx) @ A

    def heldout_rho(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Canonical correlations re-measured on data not used for fitting (removes over-fit inflation)."""
        zx = (X.astype(np.float64) - self.mx) @ self.A
        zy = (Y.astype(np.float64) - self.my) @ self.B
        zx, zy = zx - zx.mean(0), zy - zy.mean(0)
        return (zx * zy).sum(0) / np.sqrt((zx ** 2).sum(0) * (zy ** 2).sum(0) + 1e-12)


def block_environment(T: np.ndarray, grid: int, env: str = "ring") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T: (n, g*g, d) tile embeddings in row-major order.

    Returns block features X (n*g*g, d), environment features Y, and the image index of each row.
    env = "ring": mean embedding of the 8-neighbourhood;  "all": mean of all other tiles;
    env = "both": concatenation of the two.
    """
    n, L, d = T.shape
    G = T.reshape(n, grid, grid, d)
    tot = G.sum((1, 2), keepdims=True)
    rest = ((tot - G) / (L - 1)).reshape(n * L, d)
    pad = np.zeros((n, grid + 2, grid + 2, d), dtype=T.dtype)
    cnt = np.zeros((grid + 2, grid + 2), dtype=np.float32)
    pad[:, 1:-1, 1:-1] = G
    cnt[1:-1, 1:-1] = 1
    ring = np.zeros_like(G)
    k = np.zeros((grid, grid), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                ring += pad[:, 1 + dy: 1 + dy + grid, 1 + dx: 1 + dx + grid]
                k += cnt[1 + dy: 1 + dy + grid, 1 + dx: 1 + dx + grid]
    ring = (ring / k[None, :, :, None]).reshape(n * L, d)
    Y = {"ring": ring, "all": rest, "both": np.concatenate([ring, rest], 1)}[env]
    return T.reshape(n * L, d), Y, np.repeat(np.arange(n), L)


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between two k-dim subspaces (1 = identical, k/d = random)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return float((np.linalg.svd(Qa.T @ Qb, compute_uv=False) ** 2).mean())


def participation_ratio(v: np.ndarray) -> float:
    return float(v.sum() ** 2 / (v ** 2).sum())
