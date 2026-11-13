"""E1: body-order test on two-word expressions (human affect and concreteness norms).

For every feature set, fits the additive model (order 1) and second-order models on identical
cross-fitting folds and seeds, and applies the pre-registered pass rule per concept node.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rcf import lexical
from rcf.compare import crossfit
from rcf.stats import paired_bootstrap, pass_rule

KINDS = ["add", "poly2", "pair", "deepsets"]


def summarise(losses: dict, ceil: np.ndarray, names: list[str], base: str = "add") -> dict:
    out = {"nodes": names, "ceiling_mse": ceil.tolist(), "models": {}}
    for k, L in losses.items():  # L: (S, n, C) squared error in units of label variance
        row = {"mse": L.mean((0, 1)).tolist(), "r2": (1 - L.mean((0, 1))).tolist()}
        if k != base and base in losses:
            row["vs_add"] = []
            for c in range(L.shape[2]):
                d = paired_bootstrap(losses[base][:, :, c], L[:, :, c], B=2000)
                row["vs_add"].append({**d.as_dict(), **pass_rule(d, float(losses[base][:, :, c].mean()), float(ceil[c]))})
        out["models"][k] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="vad")
    ap.add_argument("--features", default="glove")  # comma list
    ap.add_argument("--require-norms", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--whole", default="", help="comma list of whole-phrase reference encoders")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kinds", default=",".join(KINDS))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    feats = tuple(args.features.split(","))
    tag = args.tag or f"{args.target}_{'+'.join(feats)}{'_rated' if args.require_norms else ''}"
    t0 = time.time()
    data, df = lexical.bigram_dataset(args.target, feats, require_norms=args.require_norms)
    ceil = lexical.ceiling(args.target, data)
    print(f"[{tag}] n={len(data)} d={data.X.shape[-1]} ceiling={np.round(ceil, 4)}", flush=True)
    res = crossfit(data, args.kinds.split(","), folds=args.folds, seeds=args.seeds, batch=args.batch)
    losses = dict(res["losses"])
    for m in [w for w in args.whole.split(",") if w]:
        wd = lexical.whole_phrase_dataset(df, data.y, m, data.names)
        losses[f"whole_{m}"] = crossfit(wd, ["add"], folds=args.folds, seeds=args.seeds, batch=args.batch)["losses"]["add"]
    summ = summarise(losses, ceil, data.names)
    summ.update(tag=tag, n=len(data), d=int(data.X.shape[-1]), folds=args.folds, seeds=args.seeds, hp=res["hp"],
                sec=round(time.time() - t0, 1))
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path(f"docs/results/e1_{tag}.json").write_text(json.dumps(summ, indent=1))
    Path("runs").mkdir(exist_ok=True)
    np.savez_compressed(f"runs/e1_{tag}_losses.npz", terms=df.term.to_numpy().astype(str),
                        **{k: v.astype(np.float32) for k, v in losses.items()})
    for k, row in summ["models"].items():
        extra = "" if "vs_add" not in row else "  d_vs_add=" + " ".join(
            f"{v['mean']:+.4f}[{v['lo']:+.4f},{v['hi']:+.4f}]{'*' if v['passed'] else ''}" for v in row["vs_add"])
        print(f"  {k:14s} R2=" + " ".join(f"{x:.3f}" for x in row["r2"]) + extra, flush=True)


if __name__ == "__main__":
    main()
