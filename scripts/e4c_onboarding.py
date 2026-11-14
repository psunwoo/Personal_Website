"""E4c: how many ratings from a person are needed before a personal coupling pays off?

Same setting as e4_personal.py (frozen shared additive model, GoEmotions).  Each modelled rater's
training ratings are cut down to n; the personal layer is fitted and scored on held-out comments.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from rcf import goemotions as ge
from rcf.stats import paired_bootstrap

spec = importlib.util.spec_from_file_location("personal", "scripts/e4_personal.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

com, rat, emo, Y = ge.load()
z = np.load("runs/e4_frozen.npz")
fz = dict(logits=torch.from_numpy(z["add_logits"]), A=torch.from_numpy(z["add_A"]))
cnt = rat.rater.value_counts()
big = cnt[cnt >= 500].index.to_numpy()
umap = {r: i + 1 for i, r in enumerate(big)}
u_all = rat.rater.map(umap).fillna(0).to_numpy(dtype=np.int64)
c_all = rat.comment.to_numpy(dtype=np.int64)
split = com.split.to_numpy()[c_all]
n_users = len(big) + 1


def rows(train_mask: np.ndarray) -> dict:
    out = {}
    for s in ("train", "val", "test"):
        m = (split == s) & (train_mask if s == "train" else True)
        out[s] = (torch.from_numpy(c_all[m]), torch.from_numpy(u_all[m]), torch.from_numpy(Y[m]))
    return out


full = np.ones(len(u_all), dtype=bool)
groups = c_all[split == "test"]
shared = np.stack([P.fit_eval("shared", fz, rows(full), n_users, s)[0] for s in range(3)])
res = dict(n_raters=int(len(big)), shared_loss=float(shared.mean()), curve=[])
tr_idx = {u: np.where((split == "train") & (u_all == u))[0] for u in range(1, n_users)}
for n in (10, 20, 50, 100, 200, 500, 10**6):
    row = dict(n=int(min(n, max(len(v) for v in tr_idx.values()))) if n == 10**6 else n)
    for kind in ("field", "gain", "lowrank"):
        Ls = []
        for s in range(3):
            rng = np.random.default_rng(100 + s)
            mask = np.zeros(len(u_all), dtype=bool)
            mask[(split == "train") & (u_all == 0)] = True
            for u, ix in tr_idx.items():
                mask[rng.permutation(ix)[:n]] = True
            Ls.append(P.fit_eval(kind, fz, rows(mask), n_users, s)[0])
        d = paired_bootstrap(shared, np.stack(Ls), B=500, groups=groups)
        row[kind] = d.as_dict()
    res["curve"].append(row)
    print(f"n={'all' if n == 10**6 else n:>4}: " + "  ".join(f"{k} {row[k]['mean']:+.4f} [{row[k]['lo']:+.4f},{row[k]['hi']:+.4f}]" for k in ("field", "gain", "lowrank")), flush=True)
Path("docs/results/e4c_onboarding.json").write_text(json.dumps(res, indent=1))
