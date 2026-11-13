"""E1c: Stanford Sentiment Treebank.

Part A (model-free, human labels only).  For every internal tree node, predict the parent's human
sentiment from its two children's human sentiments, with the same gradient-boosting learner
  additive:     f(y_left) + g(y_right)            (interactions between the inputs forbidden)
  interacting:  F(y_left, y_right)
The gap is the interaction between the affect of the parts; what neither explains is information
the one-dimensional sentiment variable does not carry across one coarse-graining step.

Part B.  Words of a phrase as an object set (sizes 2..12): order 1 vs 2 vs 3 vs any-order models.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from rcf import sst
from rcf.compare import holdout
from rcf.data import SetData
from rcf.embed import embed_texts
from rcf.stats import paired_bootstrap, pass_rule

GRID = [dict(), dict(K=128, hidden=256, wd=1e-2, dropout=0.1)]


def part_a(nodes) -> dict:
    inner = nodes[nodes.left.notna()].drop_duplicates("phrase")
    rank = {"test": 0, "validation": 1, "train": 2}
    inner = nodes[nodes.left.notna()].assign(r=lambda d: d.split.map(rank)).sort_values("r").drop_duplicates("phrase")
    inner = inner.assign(n_left=inner.left.str.count(" ") + 1, n_right=inner.right.str.count(" ") + 1)
    tr, te = inner[inner.split == "train"], inner[inner.split == "test"]
    out = {"n_train": len(tr), "n_test": len(te)}
    var = float(te.label.var())
    sq = {}
    for name, cols, cst in [
        ("children_additive", ["y_left", "y_right"], [{0}, {1}]),
        ("children_interacting", ["y_left", "y_right"], None),
        ("children+sizes_additive", ["y_left", "y_right", "n_left", "n_right"], [{0}, {1}, {2}, {3}]),
        ("children+sizes_interacting", ["y_left", "y_right", "n_left", "n_right"], None),
    ]:
        errs = []
        for seed in range(3):
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, interaction_cst=cst, random_state=seed,
                                              early_stopping=True, validation_fraction=0.1)
            m.fit(tr[cols].to_numpy(), tr.label.to_numpy())
            errs.append((te.label.to_numpy() - m.predict(te[cols].to_numpy())) ** 2)
        sq[name] = np.stack(errs) / var
        out[name] = dict(r2=float(1 - sq[name].mean()))
    for a, b in [("children_additive", "children_interacting"), ("children+sizes_additive", "children+sizes_interacting")]:
        d = paired_bootstrap(sq[a], sq[b], B=2000)
        out[b]["vs_additive"] = d.as_dict()
    # by size of the parent phrase
    size = te["size"].to_numpy()
    out["by_size"] = {}
    for lo, hi in [(2, 2), (3, 4), (5, 8), (9, 16), (17, 60)]:
        m = (size >= lo) & (size <= hi)
        out["by_size"][f"{lo}-{hi}"] = dict(n=int(m.sum()), r2_additive=float(1 - sq["children_additive"][:, m].mean() * var / te.label[m].var()),
                                           r2_interacting=float(1 - sq["children_interacting"][:, m].mean() * var / te.label[m].var()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ab")
    ap.add_argument("--features", default="minilm,wordsent")
    ap.add_argument("--kinds", default="add,poly2,poly3,deepsets,settrans")
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-test", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="", help="suffix for output files, so a run does not overwrite another")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    nodes, dic = sst.load()
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    if "a" in args.part:
        a = part_a(nodes)
        Path("docs/results/e1c_sst_tree.json").write_text(json.dumps(a, indent=1))
        print(json.dumps(a, indent=1), flush=True)
    if "b" not in args.part:
        return
    feats = tuple(args.features.split(","))
    tab = sst.phrase_table(nodes)
    rng = np.random.default_rng(0)
    idx = {s: np.where(tab.split == s)[0] for s in ("train", "validation", "test")}
    idx["train"] = rng.permutation(idx["train"])[: args.max_train]
    idx["test"] = rng.permutation(idx["test"])[: args.max_test]
    idx["validation"] = rng.permutation(idx["validation"])[:8000]
    data = sst.phrase_sets(tab, dic, feats)
    tr, va, te = (data.subset(idx[s]) for s in ("train", "validation", "test"))
    kinds = args.kinds.split(",")
    print(f"[sst {feats}] train {len(tr)} val {len(va)} test {len(te)} d={data.X.shape[-1]}", flush=True)
    kw = dict(batch=512, max_epochs=40, patience=5)
    r = holdout(tr, va, te, kinds, seeds=args.seeds, grid=GRID, **kw)
    losses = {k: v[:, :, 0] for k, v in r["losses"].items()}
    E = embed_texts(tab.phrase.tolist(), "mpnet")  # any-order reference: the phrase encoded as one string
    wd = SetData(E[:, None, :], np.ones((len(E), 1), dtype=bool), data.y, "reg", data.names)
    losses["whole_mpnet"] = holdout(wd.subset(idx["train"]), wd.subset(idx["validation"]), wd.subset(idx["test"]), ["add"],
                                    seeds=args.seeds, grid=GRID, **kw)["losses"]["add"][:, :, 0]
    refs = [k for k in ("settrans", "deepsets", "whole_mpnet") if k in losses]
    ceil_key = min(refs, key=lambda k: losses[k].mean())
    Lceil = float(losses[ceil_key].mean())
    size = tab["size"].to_numpy()[idx["test"]]
    summ = dict(features=feats, n=dict(train=len(tr), val=len(va), test=len(te)), ceiling_model=ceil_key, ceiling_mse=Lceil,
                hp=r["hp"], models={}, by_size={})
    for k, L in losses.items():
        row = dict(mse=float(L.mean()), r2=float(1 - L.mean()))
        if k != "add":
            d = paired_bootstrap(losses["add"], L, B=2000)
            row["vs_add"] = {**d.as_dict(), **pass_rule(d, float(losses["add"].mean()), Lceil)}
        summ["models"][k] = row
        v = row.get("vs_add")
        print(f"  {k:12s} R2 {row['r2']:.4f}" + ("" if not v else f"  d {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] share {v['share_of_headroom']:.2f} pass={v['passed']}"), flush=True)
    for lo, hi in [(2, 2), (3, 3), (4, 5), (6, 8), (9, 12)]:
        m = (size >= lo) & (size <= hi)
        summ["by_size"][f"{lo}-{hi}"] = dict(n=int(m.sum()), **{k: float(L[:, m].mean()) for k, L in losses.items()},
                                            **{f"d_{k}": paired_bootstrap(losses["add"][:, m], losses[k][:, m], B=1000).as_dict()
                                               for k in losses if k != "add"})
    summ["sec"] = round(time.time() - t0, 1)
    tag = "+".join(feats) + (f"_{args.tag}" if args.tag else "")
    Path(f"docs/results/e1c_sst_sets_{tag}.json").write_text(json.dumps(summ, indent=1))
    np.savez_compressed(f"runs/e1c_sst_{tag}_losses.npz", size=size, phrase=tab.phrase.to_numpy()[idx["test"]].astype(str),
                        **{k: v.astype(np.float32) for k, v in losses.items()})


if __name__ == "__main__":
    main()
