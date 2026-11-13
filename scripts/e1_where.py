"""Where do second-order effects sit in two-word expressions?  Uses saved out-of-fold losses."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(tag: str, better: str = "poly2", node: int = 0) -> None:
    z = np.load(f"runs/e1_{tag}_losses.npz")
    terms = z["terms"]
    g = (z["add"][:, :, node] - z[better][:, :, node]).mean(0)  # per-item gain from allowing interaction
    res_add = z["add"][:, :, node].mean(0)
    order = np.argsort(-g)
    pos = np.clip(np.sort(g)[::-1], 0, None)
    by_word = defaultdict(list)
    for t, gi in zip(terms, g):
        w1, w2 = t.split(" ")
        by_word[("first", w1)].append(gi)
        by_word[("second", w2)].append(gi)
    words = sorted(((np.mean(v), len(v), k) for k, v in by_word.items() if len(v) >= 12), reverse=True)
    out = dict(tag=tag, model=better, n=len(g), mean_gain=float(g.mean()),
               share_of_gain_in_top_5pct=float(pos[: len(pos) // 20].sum() / pos.sum()),
               share_of_items_with_positive_gain=float((g > 0).mean()),
               corr_gain_vs_additive_error=float(np.corrcoef(g, res_add)[0, 1]),
               top_items=[dict(term=str(terms[i]), gain=float(g[i]), additive_sq_err=float(res_add[i])) for i in order[:30]],
               top_words=[dict(slot=k[0], word=k[1], n=n, mean_gain=float(m)) for m, n, k in words[:20]])
    Path(f"docs/results/e1_where_{tag}.json").write_text(json.dumps(out, indent=1))
    print(tag, "mean gain %.4f; top 5%% of items carry %.0f%% of the gain; positive for %.0f%% of items" %
          (out["mean_gain"], 100 * out["share_of_gain_in_top_5pct"], 100 * out["share_of_items_with_positive_gain"]))
    print(" items:", [d["term"] for d in out["top_items"][:20]])
    print(" words:", [(d["slot"], d["word"], d["n"], round(d["mean_gain"], 3)) for d in out["top_words"][:12]])


if __name__ == "__main__":
    main(*sys.argv[1:2], **({"better": sys.argv[2]} if len(sys.argv) > 2 else {}))
