"""E1 (model-free variant): does a two-word expression's human norm follow from its words' human norms?

Inputs are only the human norms of each word (V, A, D, concreteness).  The same gradient-boosting
learner is fitted twice:
  order 1:  f(norms of word 1) + g(norms of word 2)     (no interaction across words)
  order 2:  F(norms of word 1, norms of word 2)
5-fold cross-fitting, 3 seeds, paired bootstrap.  No embeddings, no language model anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from rcf import lexical
from rcf.data import kfold
from rcf.stats import paired_bootstrap, pass_rule


def run(target: str) -> dict:
    data, df = lexical.bigram_dataset(target, ("norms",), require_norms=True)
    X = data.X[:, :, :4].reshape(len(data), 8)  # V, A, D, C of word 1 then word 2
    ok = ~np.isnan(X).any(1)
    cst = [{0, 1, 2, 3}, {4, 5, 6, 7}]
    folds = kfold(len(data), 5)
    ceil = lexical.ceiling(target, data)
    out = dict(n=int(len(data)), nodes=data.names, ceiling_mse=ceil.tolist(), models={})
    sq = {"order1": [], "order2": []}
    for name, c in (("order1", cst), ("order2", None)):
        for seed in range(3):
            err = np.zeros_like(data.y)
            for j in range(5):
                tr, te = folds != j, folds == j
                for t in range(data.y.shape[1]):
                    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, interaction_cst=c, random_state=seed,
                                                      early_stopping=True, validation_fraction=0.1)
                    m.fit(X[tr], data.y[tr, t])
                    err[te, t] = (data.y[te, t] - m.predict(X[te])) ** 2
            sq[name].append(err / data.y.var(0))
        sq[name] = np.stack(sq[name])
        out["models"][name] = dict(r2=(1 - sq[name].mean((0, 1))).tolist())
    out["models"]["order2"]["vs_order1"] = []
    for t in range(data.y.shape[1]):
        d = paired_bootstrap(sq["order1"][:, :, t], sq["order2"][:, :, t], B=2000)
        out["models"]["order2"]["vs_order1"].append({**d.as_dict(), **pass_rule(d, float(sq["order1"][:, :, t].mean()), float(ceil[t]))})
    # where the interaction sits: items whose error drops most when words may interact (first node)
    gain = (sq["order1"] - sq["order2"]).mean(0)[:, 0]
    top = np.argsort(-gain)[:25]
    out["top_items"] = [dict(term=str(df.term[i]), y=float(data.y[i, 0]), w1=float(X[i, 0 if target == "vad" else 3]),
                             w2=float(X[i, 4 if target == "vad" else 7]), gain=float(gain[i])) for i in top]
    srt = np.sort(gain)[::-1]
    pos = np.clip(srt, 0, None)
    out["gain_share_top5pct"] = float(pos[: len(pos) // 20].sum() / pos.sum())
    return out


if __name__ == "__main__":
    res = {t: run(t) for t in ("vad", "conc")}
    Path("docs/results/e1_norms_gbm.json").write_text(json.dumps(res, indent=1))
    for t, r in res.items():
        print(t, "n", r["n"], "nodes", r["nodes"], "ceiling", np.round(r["ceiling_mse"], 3))
        print("  order1 R2", np.round(r["models"]["order1"]["r2"], 3), " order2 R2", np.round(r["models"]["order2"]["r2"], 3))
        for nm, v in zip(r["nodes"], r["models"]["order2"]["vs_order1"]):
            print(f"   {nm}: d {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] share {v['share_of_headroom']:.3f} pass={v['passed']}")
        print("  top items:", [(i["term"], round(i["w1"], 2), round(i["w2"], 2), round(i["y"], 2)) for i in r["top_items"][:14]])
