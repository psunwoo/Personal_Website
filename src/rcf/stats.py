"""Uncertainty for model comparisons: paired two-way bootstrap, ceilings, pass rule."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Delta:
    mean: float  # mean paired loss difference L_a - L_b (positive: b is better)
    lo: float  # 2.5 %
    hi: float  # 97.5 %
    seed_sd: float  # sd of the per-seed mean difference

    def as_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(la: np.ndarray, lb: np.ndarray, B: int = 4000, seed: int = 0,
                     groups: np.ndarray | None = None) -> Delta:
    """la, lb: (S seeds, n items) held-out losses of two models on the same items.

    Resamples seeds and items (or whole groups of items) with replacement at
    once, so the interval reflects both sources of noise.
    """
    D = np.asarray(la, dtype=np.float64) - np.asarray(lb, dtype=np.float64)
    S, n = D.shape
    rng = np.random.default_rng(seed)
    if groups is not None:  # cluster bootstrap: collapse items to group sums
        ug, inv = np.unique(groups, return_inverse=True)
        G = np.zeros((S, len(ug)))
        np.add.at(G.T, inv, D.T)
        cnt = np.bincount(inv).astype(np.float64)
    else:
        G, cnt = D, np.ones(n)
    m = G.shape[1]
    out = np.empty(B)
    for b in range(B):
        w = rng.multinomial(m, np.full(m, 1.0 / m)).astype(np.float64)
        s = rng.integers(S, size=S)
        out[b] = (G[s] @ w).mean() / (cnt @ w)
    return Delta(float(D.mean()), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)),
                 float(D.mean(1).std(ddof=1)) if S > 1 else float("nan"))


def mean_ci(l: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    d = paired_bootstrap(l, np.zeros_like(l), B, seed)
    return d.mean, d.lo, d.hi


def pass_rule(delta: Delta, L1: float, Lceil: float, frac: float = 0.10) -> dict:
    """Pre-registered rule: lower CI bound > 0 AND gain >= frac of the headroom L1 - Lceil."""
    headroom = L1 - Lceil
    share = delta.mean / headroom if headroom > 0 else float("nan")
    return dict(significant=bool(delta.lo > 0), headroom=float(headroom), share_of_headroom=float(share),
                passed=bool(delta.lo > 0 and headroom > 0 and share >= frac))


def dist_ceiling(y_emp: np.ndarray, p_true: np.ndarray) -> float:
    """Cross-entropy of the empirical label distribution under the true one (synthetic data only)."""
    return float(-(y_emp * np.log(np.clip(p_true, 1e-12, None))).sum(1).mean())
