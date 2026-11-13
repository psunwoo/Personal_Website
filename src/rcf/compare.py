"""Fit several body orders on identical splits and seeds; return per-item held-out losses."""
from __future__ import annotations

import numpy as np

from .data import SetData, kfold
from .train import fit, select_hp, test_loss


def _inner_split(n: int, seed: int, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    p = np.random.default_rng(1000 + seed).permutation(n)
    k = max(int(frac * n), 1)
    return p[k:], p[:k]


def crossfit(data: SetData, kinds: list[str], folds: int = 5, seeds: int = 5, tune: bool = True,
             verbose: bool = True, **kw) -> dict:
    """Out-of-fold losses for every item.  Returns {kind: (seeds, n[, C])} plus chosen hyper-parameters."""
    f = kfold(len(data), folds, seed=0, groups=data.groups)
    losses = {k: [] for k in kinds}
    hps: dict[str, dict] = {}
    for k in kinds:
        tr0 = data.subset(np.where(f != 0)[0])
        a, b = _inner_split(len(tr0), 0)
        hps[k] = select_hp(tr0.subset(a), tr0.subset(b), k, **kw) if tune else {}
        for s in range(seeds):
            out = None
            for j in range(folds):
                te = np.where(f == j)[0]
                tr = data.subset(np.where(f != j)[0])
                a, b = _inner_split(len(tr), s)
                ft = fit(tr.subset(a), tr.subset(b), k, hps[k], seed=s, **kw)
                l = test_loss(ft, data.subset(te))
                if out is None:
                    out = np.zeros((len(data),) + l.shape[1:], dtype=np.float32)
                out[te] = l
            losses[k].append(out)
            if verbose:
                print(f"  {k:9s} seed {s}: loss {out.mean():.4f}", flush=True)
        losses[k] = np.stack(losses[k])
    return dict(losses=losses, hp=hps)


def holdout(train: SetData, val: SetData, test: SetData, kinds: list[str], seeds: int = 5, tune: bool = True,
            verbose: bool = True, keep_models: bool = False, **kw) -> dict:
    """Fixed train/val/test split.  Returns {kind: (seeds, n_test[, C])}."""
    losses, hps, models = {}, {}, {}
    for k in kinds:
        hps[k] = select_hp(train, val, k, **kw) if tune else {}
        ls = []
        for s in range(seeds):
            ft = fit(train, val, k, hps[k], seed=s, **kw)
            ls.append(test_loss(ft, test))
            if keep_models:
                models.setdefault(k, []).append(ft)
            if verbose:
                print(f"  {k:9s} seed {s}: loss {ls[-1].mean():.4f} (epochs {ft.epochs})", flush=True)
        losses[k] = np.stack(ls)
    return dict(losses=losses, hp=hps, models=models)
