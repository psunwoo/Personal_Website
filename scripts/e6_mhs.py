"""E6: per-user couplings on a second dataset with many people and few ratings each.

Measuring Hate Speech (Kennedy et al., CC-BY-4.0): 135,556 ratings of 39,565 comments by 7,912
annotators (median 17 ratings each), on ten ordinal survey items (sentiment, respect, insult, ...),
used here as ten concept nodes.  Only annotator ids are used, no demographic fields.

Shared model: cross-fitted ridge regression from a sentence embedding of the comment to the ten
nodes; variant "+crowd" adds the (shrunk, leave-one-out) mean residual of the other people who
rated the same comment.  The personal layer is a field b_u on the nodes.  Because the model is
linear-Gaussian, the cold-start estimate of b_u from n ratings is closed-form:

    b_u = S_b (S_b + S_e / n)^-1 * (mean residual of the n ratings)

with S_b the covariance of fields across people and S_e the within-person noise covariance, both
estimated on development annotators.  Restricting S_b to its top k eigen-directions gives a k-dim
user space; using only its diagonal gives independent shrinkage per node.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from rcf.embed import embed_texts

ITEMS = ["sentiment", "respect", "insult", "humiliate", "status", "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"]
d = pd.read_parquet("data/raw/mhs/mhs.parquet", columns=["comment_id", "annotator_id", "text"] + ITEMS)
Yr = d[ITEMS].to_numpy(dtype=np.float64)
Yr = (Yr - Yr.mean(0)) / Yr.std(0)
com = d.drop_duplicates("comment_id")[["comment_id", "text"]].reset_index(drop=True)
cidx = d.comment_id.map({c: i for i, c in enumerate(com.comment_id)}).to_numpy()
E = embed_texts(com.text.tolist(), "minilm")
rng = np.random.default_rng(0)

# shared model, cross-fitted by comment
fold = rng.integers(5, size=len(com))[cidx]
F = np.zeros_like(Yr)
for j in range(5):
    tr, te = fold != j, fold == j
    F[te] = Ridge(alpha=10.0).fit(E[cidx[tr]], Yr[tr]).predict(E[cidx[te]])
res0 = Yr - F

# people: development vs new
ann = d.annotator_id.to_numpy()
ua, uinv = np.unique(ann, return_inverse=True)
is_new_u = rng.random(len(ua)) < 0.3
is_new = is_new_u[uinv]

# "+crowd": leave-one-out mean residual of the other raters of the same comment, shrunk
S = np.zeros((len(com), len(ITEMS)))
N = np.zeros(len(com))
np.add.at(S, cidx, res0)
np.add.at(N, cidx, 1)
m_other = N[cidx] - 1
loo = np.where(m_other[:, None] > 0, (S[cidx] - res0) / np.maximum(m_other, 1)[:, None], 0.0)
dev_rows = ~is_new
best = min((float(((res0 - (m_other / (m_other + k))[:, None] * loo)[dev_rows] ** 2).mean()), k) for k in (0.25, 0.5, 1, 2, 4, 8))
kappa = best[1]
res1 = res0 - (m_other / (m_other + kappa))[:, None] * loo
out = dict(n_ratings=int(len(d)), n_comments=int(len(com)), n_people=int(len(ua)), n_new_people=int(is_new_u.sum()), crowd_kappa=kappa,
           r2_shared={"ridge": float(1 - (res0 ** 2).mean()), "ridge+crowd": float(1 - (res1 ** 2).mean())}, bases={})
print("shared R2:", out["r2_shared"], "kappa", kappa, flush=True)

for base, res in (("ridge", res0), ("ridge+crowd", res1)):
    # variance components on development people
    C = len(ITEMS)
    Sw, dof, means, ns = np.zeros((C, C)), 0, [], []
    order = np.argsort(uinv, kind="stable")
    bounds = np.searchsorted(uinv[order], np.arange(len(ua) + 1))
    rows_of = [order[bounds[i]:bounds[i + 1]] for i in range(len(ua))]
    for i in np.where(~is_new_u)[0]:
        r = res[rows_of[i]]
        if len(r) < 2:
            continue
        mu = r.mean(0)
        Sw += (r - mu).T @ (r - mu)
        dof += len(r) - 1
        means.append(mu)
        ns.append(len(r))
    Se = Sw / dof
    means, ns = np.array(means), np.array(ns)
    Sb = np.cov(means.T) - Se * np.mean(1.0 / ns)
    w, V = np.linalg.eigh(Sb)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    spec = dict(eigen_share=(w / w.sum()).round(4).tolist(), participation_ratio=float(w.sum() ** 2 / (w ** 2).sum()),
                field_sd_by_node={k: float(np.sqrt(max(v, 0))) for k, v in zip(ITEMS, np.diag(Sb))},
                noise_sd_by_node={k: float(np.sqrt(v)) for k, v in zip(ITEMS, np.diag(Se))},
                top_direction={k: float(v) for k, v in zip(ITEMS, V[:, 0])})
    print(f"[{base}] share of between-person variance by direction: {np.round(w / w.sum(), 3)[:6]}  PR {spec['participation_ratio']:.2f}", flush=True)

    priors = {"none (raw mean)": None, "diagonal": np.diag(np.diag(Sb).clip(min=0)), "full covariance": (V * w) @ V.T}
    for k in (1, 2, 4):
        priors[f"rank {k}"] = (V[:, :k] * w[:k]) @ V[:, :k].T
    curves = {}
    for n in (1, 2, 4, 8, 12):
        gains = {p: [] for p in priors}
        for i in np.where(is_new_u)[0]:
            ix = rows_of[i]
            if len(ix) < n + 3:
                continue
            ix = rng.permutation(ix)
            fit, ev = res[ix[:n]], res[ix[n:]]
            rbar = fit.mean(0)
            base_mse = (ev ** 2).mean()
            for p, Sp in priors.items():
                b = rbar if Sp is None else Sp @ np.linalg.solve(Sp + Se / n, rbar)
                gains[p].append(base_mse - ((ev - b) ** 2).mean())
        row = {}
        for p, g in gains.items():
            g = np.array(g)
            bs = [g[rng.integers(len(g), size=len(g))].mean() for _ in range(1000)]
            row[p] = dict(mean=float(g.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), n_people=int(len(g)))
        dd = np.array(gains["full covariance"]) - np.array(gains["diagonal"])
        bs = [dd[rng.integers(len(dd), size=len(dd))].mean() for _ in range(1000)]
        row["full_minus_diagonal"] = dict(mean=float(dd.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)))
        curves[str(n)] = row
        print(f"  n={n:2d} ({row['diagonal']['n_people']} people): " + "  ".join(f"{p} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for p, v in row.items() if p in ("diagonal", "full covariance", "full_minus_diagonal")), flush=True)
    base_mse_all = float((res[is_new] ** 2).mean())
    out["bases"][base] = dict(spectrum=spec, residual_mse_new_people=base_mse_all, cold_start=curves)
Path("docs/results/e6_mhs.json").write_text(json.dumps(out, indent=1))
