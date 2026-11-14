"""E4e: train the shared additive set model and the per-rater couplings together, on individual ratings.

The two-stage result (e4_personal.py) froze a shared model fitted to comment-level averages.  Here
everything is fitted end to end on (comment, rater) rows:
    mode shared : additive set model only
    mode joint  : additive set model + per-rater field, response strength and rank-4 content readout
Usage: python scripts/e4e_joint.py shared|joint
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from rcf import goemotions as ge
from rcf.models import SetModel

torch.set_num_threads(1)
mode = sys.argv[1]
C, R = 28, 4
com, rat, emo, Y = ge.load()
data = ge.comment_sets(com, rat, Y, emo)
X, M = torch.from_numpy(data.X), torch.from_numpy(data.mask)
cnt = rat.rater.value_counts()
big = cnt[cnt >= 500].index.to_numpy()
umap = {r: i + 1 for i, r in enumerate(big)}
u_all = torch.from_numpy(rat.rater.map(umap).fillna(0).to_numpy(dtype=np.int64))
c_all = torch.from_numpy(rat.comment.to_numpy(dtype=np.int64))
split = com.split.to_numpy()[c_all.numpy()]
Yt = torch.from_numpy(Y)
rows = {s: np.where(split == s)[0] for s in ("train", "val", "test")}


class Joint(nn.Module):
    def __init__(self, personal: bool):
        super().__init__()
        self.set = SetModel(X.shape[-1], C, "add")
        self.personal = personal
        n = len(big) + 1
        self.bu, self.au, self.Uu = nn.Embedding(n, C), nn.Embedding(n, C), nn.Embedding(n, R * C)
        self.V = nn.Linear(64, R, bias=False)
        for e in (self.bu, self.au, self.Uu):
            nn.init.zeros_(e.weight)

    def forward(self, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x, m = X[c], M[c]
        mf = m.float().unsqueeze(-1)
        n = mf.sum(1).clamp(min=1)
        A = (self.set.phi(x) * mf).sum(1) / n
        s = self.set.size(torch.cat([torch.log(n), 1.0 / n], 1))
        F = torch.exp(s[:, :C]) * self.set.lin(A) + s[:, C:]
        if not self.personal:
            return F
        return (1.0 + self.au(u)) * F + self.bu(u) + torch.einsum("br,brc->bc", self.V(A), self.Uu(u).view(-1, R, C))


def ce(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(y * F_.log_softmax(z, -1)).sum(-1)


def evaluate(m: Joint, idx: np.ndarray) -> np.ndarray:
    m.eval()
    with torch.no_grad():
        return torch.cat([ce(m(c_all[idx[s:s + 4096]], u_all[idx[s:s + 4096]]), Yt[idx[s:s + 4096]]) for s in range(0, len(idx), 4096)]).numpy()


out = []
for seed in range(3):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    m = Joint(mode == "joint")
    pers = [p for e in (m.bu, m.au, m.Uu) for p in e.parameters()]
    opt = torch.optim.Adam([dict(params=list(m.set.parameters()) + list(m.V.parameters()), weight_decay=1e-6),
                            dict(params=pers, weight_decay=1e-3 / 4096 * 512)], lr=2e-3)
    best, state, bad = np.inf, None, 0
    for ep in range(25):
        m.train()
        p = g.permutation(rows["train"])
        for s in range(0, len(p), 512):
            i = p[s:s + 512]
            loss = ce(m(c_all[i], u_all[i]), Yt[i]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = float(evaluate(m, rows["val"]).mean())
        if vl < best - 1e-5:
            best, state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 3:
                break
    m.load_state_dict(state)
    tl = evaluate(m, rows["test"])
    out.append(tl)
    print(f"[{mode}] seed {seed}: test loss {tl.mean():.4f} (epochs {ep + 1}, val {best:.4f})", flush=True)
np.savez_compressed(f"runs/e4e_{mode}_losses.npz", loss=np.stack(out).astype(np.float32), comment=c_all.numpy()[rows["test"]])
