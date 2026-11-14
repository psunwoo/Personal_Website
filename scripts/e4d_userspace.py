"""E4d: do personal couplings live in a low-dimensional user space, and does that speed up onboarding?

Setting as in e4_personal.py (frozen shared additive model on GoEmotions; a rating is
comment x rater -> label distribution).  The 62 modelled raters are split into 47 "development"
raters and 15 "new" raters.

A. User space.  On the development raters fit
       full field      z = a F + b + b_u                         (28 free numbers per rater)
       k-dim field     z = a F + b + B e_u,  e_u in R^k          (B shared, 28 x k)
       k-dim full      field, response strength and content readout all driven by e_u
   and look at the spectrum of the fitted fields b_u across raters.

B. Cold start, done properly.  A new rater contributes n ratings.  Nothing else about that rater
   is used: the regularisation strength is chosen by simulating the same cold start on development
   raters, and there is no personal validation set.  Scored on the new raters' ratings of comments
   never used in training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from rcf import goemotions as ge
from rcf.stats import paired_bootstrap

torch.set_num_threads(1)
C, R = 28, 4
com, rat, emo, Y = ge.load()
z = np.load("runs/e4_frozen.npz")
Fz, Az = torch.from_numpy(z["add_logits"]), torch.from_numpy(z["add_A"])
K = Az.shape[1]
cnt = rat.rater.value_counts()
big = cnt[cnt >= 500].index.to_numpy()
rng = np.random.default_rng(0)
perm = rng.permutation(len(big))
new_r, dev_r = big[perm[:15]], big[perm[15:]]
c_all = torch.from_numpy(rat.comment.to_numpy(dtype=np.int64))
r_all = rat.rater.to_numpy()
split = com.split.to_numpy()[c_all.numpy()]
Yt = torch.from_numpy(Y)


def ce(zl: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(y * F_.log_softmax(zl, -1)).sum(-1)


class UserSpace(nn.Module):
    """theta_u = B e_u.  parts: which couplings e_u drives ("b": field, "a": response strength, "U": content readout)."""

    def __init__(self, n_users: int, k: int, parts: str, free_field: bool = False):
        super().__init__()
        self.parts, self.k, self.free = parts, k, free_field
        self.a, self.b = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C))
        self.E = nn.Embedding(n_users, C if free_field else k)
        nn.init.normal_(self.E.weight, std=0.01)
        if not free_field:
            self.Bb = nn.Linear(k, C, bias=False)
            self.Ba = nn.Linear(k, C, bias=False)
            self.BU = nn.Linear(k, R * C, bias=False)
            self.V = nn.Linear(K, R, bias=False)

    def logits(self, c: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        Fc = Fz[c]
        zl = self.a * Fc + self.b
        if self.free:
            return zl + e
        zl = zl + self.Bb(e)
        if "a" in self.parts:
            zl = zl + self.Ba(e) * Fc
        if "U" in self.parts:
            zl = zl + torch.einsum("br,brc->bc", self.V(Az[c]), self.BU(e).view(-1, R, C))
        return zl

    def forward(self, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.logits(c, self.E(u))


def rows(raters: np.ndarray, sp: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = {r: i for i, r in enumerate(raters)}
    m = np.isin(r_all, raters) & np.isin(split, sp)
    return c_all[m], torch.from_numpy(np.array([idx[r] for r in r_all[m]], dtype=np.int64)), Yt[m]


def fit_dev(k: int, parts: str, free: bool = False, seed: int = 0, epochs: int = 40) -> tuple[UserSpace, float]:
    torch.manual_seed(seed)
    m = UserSpace(len(dev_r), k, parts, free)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-5)
    (c, u, y), (cv, uv, yv) = rows(dev_r, ("train",)), rows(dev_r, ("val",))
    best, state, bad = np.inf, None, 0
    g = np.random.default_rng(seed)
    for _ in range(epochs):
        p = torch.from_numpy(g.permutation(len(c)))
        for s in range(0, len(c), 4096):
            i = p[s:s + 4096]
            loss = ce(m(c[i], u[i]), y[i]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(ce(m(cv, uv), yv).mean())
        if vl < best - 1e-5:
            best, state, bad = vl, {a: b.clone() for a, b in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(state)
    return m, best


def cold_start(m: UserSpace, raters: np.ndarray, n: int, lam: float, prior_sd: torch.Tensor, seed: int, eval_sp: tuple[str, ...]):
    """Fit only e_u for each rater from n training ratings (MAP with a Gaussian prior); return per-rating loss on eval_sp."""
    g = np.random.default_rng(1000 + seed)
    c, u, y = rows(raters, ("train",))
    keep = np.zeros(len(c), dtype=bool)
    for j in range(len(raters)):
        ix = np.where(u.numpy() == j)[0]
        keep[g.permutation(ix)[:n]] = True
    c, u, y = c[keep], u[keep], y[keep]
    e = torch.zeros(len(raters), prior_sd.shape[0], requires_grad=True)
    opt = torch.optim.LBFGS([e], lr=0.5, max_iter=200, history_size=20, line_search_fn="strong_wolfe")
    for p_ in m.parameters():
        p_.requires_grad_(False)

    def closure():
        opt.zero_grad()
        loss = ce(m.logits(c, e[u]), y).sum() + 0.5 * lam * ((e / prior_sd) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    ce_, ue, ye = rows(raters, eval_sp)
    with torch.no_grad():
        return ce(m.logits(ce_, e.detach()[ue]), ye).numpy(), ce(m.logits(ce_, torch.zeros_like(e)[ue]), ye).numpy(), ce_.numpy()


res = dict(n_dev=len(dev_r), n_new=len(new_r), user_space={}, cold_start=[])
# ---- A. user space on development raters ----
models = {}
free, v_free = fit_dev(C, "b", free=True)
models["field_full"] = free
with torch.no_grad():
    Bu = free.E.weight.numpy()
    sv = np.linalg.svd(Bu - Bu.mean(0), compute_uv=False) ** 2
    res["user_space"]["field_variance_explained_by_top_k"] = {str(k): float(sv[:k].sum() / sv.sum()) for k in (1, 2, 4, 8, 16)}
    res["user_space"]["participation_ratio"] = float(sv.sum() ** 2 / (sv ** 2).sum())
    shared_val = float(ce(free.logits(*[rows(dev_r, ("val",))[0], torch.zeros(len(rows(dev_r, ("val",))[0]), C)]), rows(dev_r, ("val",))[2]).mean())
print("spectrum of rater fields:", res["user_space"], flush=True)
res["user_space"]["val_gain"] = {"field_full": shared_val - v_free}
for k in (1, 2, 4, 8):
    for parts in ("b", "baU"):
        m, v = fit_dev(k, parts)
        models[f"{'field' if parts == 'b' else 'full'}_k{k}"] = m
        res["user_space"]["val_gain"][f"{'field' if parts == 'b' else 'full'}_k{k}"] = shared_val - v
print("validation gain on development raters:", {a: round(b, 4) for a, b in res["user_space"]["val_gain"].items()}, flush=True)

# ---- B. cold start on new raters ----
ns = (5, 10, 20, 50, 100, 200)
lams = (0.3, 1.0, 3.0, 10.0)
for name, m in models.items():
    with torch.no_grad():
        sd = m.E.weight.std(0) + 1e-6
    for n in ns:
        # pick the prior strength by simulating the cold start on development raters (their validation ratings)
        sim = {lam: float(cold_start(m, dev_r, n, lam, sd, 0, ("val",))[0].mean()) for lam in lams}
        lam = min(sim, key=sim.get)
        L1, L0, grp = [], [], None
        for s in range(3):
            a, b, grp = cold_start(m, new_r, n, lam, sd, s, ("val", "test"))
            L1.append(a)
            L0.append(b)
        d = paired_bootstrap(np.stack(L0), np.stack(L1), B=500, groups=grp)
        res["cold_start"].append(dict(model=name, n=n, lam=lam, gain=d.as_dict(), shared_loss=float(np.mean(L0))))
        print(f"  {name:11s} n={n:4d} lam={lam:4.1f}  gain {d.mean:+.4f} [{d.lo:+.4f},{d.hi:+.4f}]", flush=True)
Path("docs/results/e4d_userspace.json").write_text(json.dumps(res, indent=1))
