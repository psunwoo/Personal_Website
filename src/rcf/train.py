"""Training, prediction and per-item held-out losses."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import SetData
from .models import SetModel

DEFAULT_HP = dict(K=64, hidden=128, rank=16, dropout=0.0, wd=1e-4, lr=2e-3)
HP_GRID = [
    dict(),
    dict(wd=1e-2, dropout=0.1),
    dict(K=128, hidden=256),
    dict(K=128, hidden=256, wd=1e-2, dropout=0.1),
]


@dataclass
class Fit:
    model: SetModel
    val_loss: float
    epochs: int
    mu: np.ndarray | None
    sd: np.ndarray | None


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def item_loss(z: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    """Per-item loss, (n, C) for regression (squared error per node), (n,) for dist (cross-entropy)."""
    if task == "dist":
        return -(y * F.log_softmax(z, dim=-1)).sum(-1)
    return (z - y) ** 2


def fit(train: SetData, val: SetData, kind: str, hp: dict | None = None, seed: int = 0,
        max_epochs: int = 80, patience: int = 8, batch: int = 256) -> Fit:
    hp = {**DEFAULT_HP, **(hp or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    task = train.task
    mu = sd = None
    ytr, yva = train.y, val.y
    if task == "reg":  # standardise targets with training statistics
        mu, sd = train.y.mean(0), train.y.std(0) + 1e-8
        ytr, yva = (train.y - mu) / sd, (val.y - mu) / sd
    Xtr, Mtr, Ytr = _t(train.X), _t(train.mask), _t(ytr.astype(np.float32))
    Xva, Mva, Yva = _t(val.X), _t(val.mask), _t(yva.astype(np.float32))
    model = SetModel(train.X.shape[-1], train.y.shape[-1], kind, hp["K"], hp["hidden"], hp["rank"], hp["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best, best_state, bad, ep = np.inf, None, 0, 0
    n = len(Xtr)
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = _t(perm[s:s + batch])
            loss = item_loss(model(Xtr[idx], Mtr[idx]), Ytr[idx], task).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(torch.cat([item_loss(model(Xva[s:s + 2048], Mva[s:s + 2048]), Yva[s:s + 2048], task).reshape(-1)
                                  for s in range(0, len(Xva), 2048)]).mean())
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return Fit(model, best, ep, mu, sd)


@torch.no_grad()
def predict(f: Fit, X: np.ndarray, mask: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Model output in standardised units (reg) or logits (dist)."""
    Xt, Mt = _t(X), _t(mask)
    return torch.cat([f.model(Xt[s:s + chunk], Mt[s:s + chunk]) for s in range(0, len(Xt), chunk)]).numpy()


def test_loss(f: Fit, test: SetData) -> np.ndarray:
    """Per-item held-out loss.  Regression: (n, C) squared error in units of the training label variance."""
    z = _t(predict(f, test.X, test.mask))
    y = test.y if f.mu is None else (test.y - f.mu) / f.sd
    return item_loss(z, _t(y.astype(np.float32)), test.task).numpy()


def select_hp(train: SetData, val: SetData, kind: str, grid: list[dict] | None = None, **kw) -> dict:
    grid = HP_GRID if grid is None else grid
    scores = [fit(train, val, kind, g, seed=0, **kw).val_loss for g in grid]
    return grid[int(np.argmin(scores))]
