"""E0: calibrate the Q1 estimator on synthetic data with a planted pair interaction.

Answers two questions before any real data is touched:
  1. With no interaction (eta = 0), how often does the pass rule fire?  (false positives)
  2. How small an interaction-variance share eta can be detected at a given sample size?

Appends one JSON line per (n_train, eta, replicate) to docs/results/e0_synthetic.jsonl; resumable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rcf import synthetic
from rcf.compare import holdout
from rcf.stats import dist_ceiling, paired_bootstrap, pass_rule

KINDS = ["add", "poly2", "pair"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/e0_synthetic.jsonl")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["n_train"], r["eta"], r["rep"]) for r in map(json.loads, out.read_text().splitlines())}
    grid = []
    for rep in range(6):  # null replicates first: false-positive rate
        grid.append((4000, 0.0, rep))
    for n in (4000, 16000):
        for eta in (0.01, 0.03, 0.1, 0.3):
            grid.append((n, eta, 0))
    grid.append((16000, 0.0, 0))
    grid.append((16000, 0.0, 1))
    for n, eta, rep in grid:
        if (n, eta, rep) in done:
            continue
        t0 = time.time()
        base = 100 * rep + int(1000 * eta) + n
        tr, _ = synthetic.make(n, eta, seed=base + 1)
        va, _ = synthetic.make(max(n // 8, 500), eta, seed=base + 2)
        te, info = synthetic.make(4000, eta, seed=base + 3)
        r = holdout(tr, va, te, KINDS, seeds=args.seeds, tune=False, verbose=False)
        L = {k: float(v.mean()) for k, v in r["losses"].items()}
        ceil = dist_ceiling(te.y, info["p_true"])
        row = dict(n_train=n, eta=eta, rep=rep, seeds=args.seeds, loss=L, ceiling=ceil, sec=None)
        for k in KINDS[1:]:
            d = paired_bootstrap(r["losses"]["add"], r["losses"][k], B=2000, seed=rep)
            row[k] = {**d.as_dict(), **pass_rule(d, L["add"], ceil)}
        row["sec"] = round(time.time() - t0, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
