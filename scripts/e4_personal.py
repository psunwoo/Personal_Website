"""E4, stage 2: per-rater couplings on top of a frozen shared model (GoEmotions, 28 emotion labels).

A rating is (comment, rater, label distribution).  The shared model gives logits F(S) for the
comment; a rater-specific layer is fitted on the training comments and scored on held-out comments:

  shared       z = a * F + b                               (global recalibration only)
  field        z = a * F + b + b_u                         rater-specific field on the concept nodes
  gain         z = (a + a_u) * F + b + b_u                 + rater-specific response strength
  lowrank      gain + U_u V A(S)                           + rater-specific readout of content (rank r)
  opstrength   gain, with operators scaled per rater:  m_u(x) = 1 + s_u * (m(x) - 1)

Controls: the same models with rater identities shuffled (any gain there is over-fitting), and the
split-half reliability of s_u across raters (is "how strongly negation acts" a stable personal trait?).
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


class Personal(nn.Module):
    def __init__(self, kind: str, n_users: int, K: int, per_node_s: bool = False):
        super().__init__()
        self.kind = kind
        self.a, self.b = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C))
        self.bu = nn.Embedding(n_users, C)
        self.au = nn.Embedding(n_users, C)
        self.su = nn.Embedding(n_users, C if per_node_s else 1)
        self.V = nn.Linear(K, R, bias=False)
        self.Uu = nn.Embedding(n_users, R * C)
        for e in (self.bu, self.au, self.su, self.Uu):
            nn.init.zeros_(e.weight)

    def forward(self, fz: dict, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if self.kind in ("opstrength", "opstrength_only"):
            s = 1.0 + self.su(u)  # B,1 or B,C
            prod = (1.0 + s.unsqueeze(1) * fz["T"][c].float()).prod(1)
            F = fz["gain"][c] * (fz["lin"][c] * prod + fz["lin_b"]) + fz["bias"][c]
        else:
            F = fz["logits"][c]
        z = self.a * F + self.b
        if self.kind in ("field", "gain", "lowrank", "opstrength"):
            z = z + self.bu(u)
        if self.kind in ("gain", "lowrank", "opstrength"):
            z = z + self.au(u) * F
        if self.kind == "lowrank":
            z = z + torch.einsum("br,brc->bc", self.V(fz["A"][c]), self.Uu(u).view(-1, R, C))
        return z


def ce(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(y * F_.log_softmax(z, -1)).sum(-1)


def fit_eval(kind: str, fz: dict, rows: dict, n_users: int, seed: int, epochs: int = 40, wd: float = 1e-3, **kw):
    torch.manual_seed(seed)
    m = Personal(kind, n_users, fz["A"].shape[1], **kw)
    shared = [m.a, m.b] + list(m.V.parameters())
    personal = [p for e in (m.bu, m.au, m.su, m.Uu) for p in e.parameters()]
    opt = torch.optim.Adam([dict(params=shared, weight_decay=0.0), dict(params=personal, weight_decay=wd)], lr=5e-3)
    c, u, y = rows["train"]
    best, state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(c)))
        for s in range(0, len(c), 4096):
            i = perm[s:s + 4096]
            loss = ce(m(fz, c[i], u[i]), y[i]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(torch.cat([ce(m(fz, rows["val"][0][s:s + 8192], rows["val"][1][s:s + 8192]), rows["val"][2][s:s + 8192])
                                  for s in range(0, len(rows["val"][0]), 8192)]).mean())
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(state)
    with torch.no_grad():
        tl = torch.cat([ce(m(fz, rows["test"][0][s:s + 8192], rows["test"][1][s:s + 8192]), rows["test"][2][s:s + 8192])
                        for s in range(0, len(rows["test"][0]), 8192)]).numpy()
    return tl, m, best


WDS = (1e-3, 1e-4, 1e-5, 1e-6)


def tuned(kind: str, fz: dict, rows: dict, n_users: int, seeds: int = 3, **kw) -> tuple[np.ndarray, float]:
    """Pick the L2 strength on the personal parameters by validation loss (seed 0), then run the seeds."""
    if kind == "shared":
        return np.stack([fit_eval(kind, fz, rows, n_users, s, **kw)[0] for s in range(seeds)]), 0.0
    wd = min(WDS, key=lambda w: fit_eval(kind, fz, rows, n_users, 0, wd=w, **kw)[2])
    return np.stack([fit_eval(kind, fz, rows, n_users, s, wd=wd, **kw)[0] for s in range(seeds)]), wd


def main(bases: tuple[str, ...] = ("add", "gate"), tag: str = "") -> None:
    com, rat, emo, Y = ge.load()
    z = np.load("runs/e4_frozen.npz")
    cnt = rat.rater.value_counts()
    big = cnt[cnt >= 500].index.to_numpy()
    umap = {r: i + 1 for i, r in enumerate(big)}  # 0 = all small raters pooled
    u_all = rat.rater.map(umap).fillna(0).to_numpy(dtype=np.int64)
    c_all = rat.comment.to_numpy(dtype=np.int64)
    split = com.split.to_numpy()[c_all]
    n_users = len(big) + 1

    def rows_for(u_vec: np.ndarray, train_mask: np.ndarray | None = None) -> dict:
        out = {}
        for s in ("train", "val", "test"):
            m = split == s
            if s == "train" and train_mask is not None:
                m = m & train_mask
            out[s] = (torch.from_numpy(c_all[m]), torch.from_numpy(u_vec[m]), torch.from_numpy(Y[m]))
        return out

    res = dict(n_ratings={s: int((split == s).sum()) for s in ("train", "val", "test")}, n_raters_modelled=int(len(big)), bases={})
    test_groups = c_all[split == "test"]
    rng = np.random.default_rng(0)
    u_shuf = u_all[rng.permutation(len(u_all))]
    for base in bases:
        fz = dict(logits=torch.from_numpy(z[f"{base}_logits"]), A=torch.from_numpy(z[f"{base}_A"]))
        kinds = ["shared", "field", "gain", "lowrank"]
        if base in ("gate", "sgate"):
            if base == "gate":
                fz.update(T=torch.from_numpy(z["gate_T"]), lin=torch.from_numpy(z["gate_lin"]), lin_b=torch.zeros(C))
            else:  # scalar factor per token, acting on the deviation from the baseline
                fz.update(T=torch.from_numpy(z["sgate_T"]).unsqueeze(-1), lin=fz["A"] @ torch.from_numpy(z["sgate_w"]).T,
                          lin_b=torch.from_numpy(z["sgate_b"]))
            fz.update(gain=torch.from_numpy(z[f"{base}_gain"]), bias=torch.from_numpy(z[f"{base}_bias"]))
            chk = fz["gain"] * (fz["lin"] * (1.0 + fz["T"].float()).prod(1) + fz["lin_b"]) + fz["bias"]
            print("gate reconstruction max abs error:", float((chk - fz["logits"]).abs().max()), flush=True)
            kinds += ["opstrength"]
        L, wds = {}, {}
        for k in kinds:
            L[k], wds[k] = tuned(k, fz, rows_for(u_all), n_users)
        for k in ("field", "lowrank"):
            L[k + "_shuffled"], wds[k + "_shuffled"] = tuned(k, fz, rows_for(u_shuf), n_users)
        out = {}
        for k, v in L.items():
            row = dict(loss=float(v.mean()), l2=wds[k])
            if k != "shared":
                row["vs_shared"] = paired_bootstrap(L["shared"], v, B=1000, groups=test_groups).as_dict()
            out[k] = row
            d = row.get("vs_shared")
            print(f"  [{base}] {k:18s} loss {row['loss']:.4f}" + ("" if not d else f"  gain {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"), flush=True)
        for a, b in (("field", "gain"), ("gain", "lowrank")) + ((("gain", "opstrength"),) if base in ("gate", "sgate") else ()):
            out[f"{b}_minus_{a}"] = paired_bootstrap(L[a], L[b], B=1000, groups=test_groups).as_dict()
            print(f"  [{base}] {b} over {a}: {out[f'{b}_minus_{a}']}", flush=True)
        if base in ("gate", "sgate"):
            # is operator strength a stable personal trait?  fit s_u on two disjoint halves of the training ratings
            half = rng.random(len(u_all)) < 0.5
            s_half = []
            for hm in (half, ~half):
                _, m, _ = fit_eval("opstrength", fz, rows_for(u_all, hm), n_users, 0, wd=wds["opstrength"])
                s_half.append(1.0 + m.su.weight.detach().numpy()[1:, 0])
            _, m, _ = fit_eval("opstrength", fz, rows_for(u_all), n_users, 0, wd=wds["opstrength"])
            s_full = 1.0 + m.su.weight.detach().numpy()[1:, 0]
            out["operator_strength"] = dict(mean=float(s_full.mean()), sd=float(s_full.std()), min=float(s_full.min()), max=float(s_full.max()),
                                            split_half_r=float(np.corrcoef(s_half[0], s_half[1])[0, 1]), n_raters=int(len(s_full)))
            print("  operator strength per rater:", out["operator_strength"], flush=True)
            bu = m.bu.weight.detach().numpy()[1:]
            out["field_sd_by_node"] = {e: float(v) for e, v in sorted(zip(emo, bu.std(0)), key=lambda t: -t[1])[:8]}
            print("  rater field, largest spread:", out["field_sd_by_node"], flush=True)
        res["bases"][base] = out
    Path(f"docs/results/e4_personal{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main(tuple(sys.argv[1].split(",")), tag="_" + sys.argv[1].replace(",", "+"))
    else:
        main()
