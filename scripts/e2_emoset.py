"""E2: body-order test on EmoSet-118K (evoked emotion of real images).

--rep tiles2|tiles3|tiles4   objects = isolated image tiles (DINOv2), with tile position
--rep attrs                  objects = EmoSet attributes embedded as isolated text
--rep tags|tags_norms|tags_text
                             objects = CLIP zero-shot concept tags; described by a text embedding
                             and/or the word's human norms (V, A, D, concreteness)

Any-order references: Deep Sets, Set Transformer on the same objects, and whole-image DINOv2 / CLIP
embeddings.  The whole-image DINOv2 model defines the ceiling in the pass rule.
With --holdout the official train/val/test split is used; otherwise the listed splits are pooled
and cross-fitted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rcf import emoset
from rcf.compare import crossfit, holdout
from rcf.data import SetData
from rcf.stats import paired_bootstrap, pass_rule


def concat(parts: list[SetData]) -> SetData:
    return SetData(np.concatenate([p.X for p in parts]), np.concatenate([p.mask for p in parts]),
                   np.concatenate([p.y for p in parts]), parts[0].task, parts[0].names,
                   np.concatenate([p.ids for p in parts]))


def build(rep: str, feats: dict[str, dict], fit_split: str, pca: int) -> dict[str, SetData]:
    if rep.startswith("tiles"):
        g = int(rep[-1])
        std = emoset.Standardiser(feats[fit_split][f"d{g}"], pca)
        return {s: emoset.tile_sets(f, g, std) for s, f in feats.items()}
    if rep in ("whole_d1", "whole_clip"):
        key = rep.split("_")[1]
        std = emoset.Standardiser(feats[fit_split][key])
        return {s: emoset.whole_set(f, key, std) for s, f in feats.items()}
    if rep == "attrs":
        return {s: emoset.attribute_sets(f) for s, f in feats.items()}
    if rep in ("tags", "tags_norms", "tags_text"):
        from rcf import tags
        fs = {"tags": ("minilm", "norms"), "tags_norms": ("norms",), "tags_text": ("minilm",)}[rep]
        return tags.tag_sets(feats, features=fs)
    raise ValueError(rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--kinds", default="add,poly2,poly3,pair,deepsets,settrans")
    ap.add_argument("--refs", default="whole_d1,whole_clip")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--grid", default="small", help="small: 2 configurations, full: 4")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tag = args.tag or f"{args.rep}_{'holdout' if args.holdout else 'crossfit'}"
    lock = Path(f"runs/e2_{tag}.lock")
    if Path(f"docs/results/e2_{tag}.json").exists() or lock.exists():  # lets several runners share one queue
        print(f"[{tag}] already done or running, skipping")
        return
    lock.touch()
    splits = ["train", "val", "test"] if args.holdout else args.splits.split(",")
    feats = {s: emoset.load(s) for s in splits}
    if args.holdout and args.max_train:
        idx = np.random.default_rng(0).permutation(len(feats["train"]["label"]))[: args.max_train]
        feats["train"] = {k: v[idx] for k, v in feats["train"].items()}
    kw = dict(batch=args.batch, max_epochs=args.epochs, patience=args.patience,
              grid=None if args.grid == "full" else [dict(), dict(K=128, hidden=256, wd=1e-2, dropout=0.1)])
    kinds = args.kinds.split(",")
    losses, acc, hp = {}, {}, {}

    def run(rep: str, ks: list[str], prefix: str = "") -> None:
        sets = build(rep, feats, splits[0], args.pca)
        if args.holdout:
            r = holdout(sets["train"], sets["val"], sets["test"], ks, seeds=args.seeds, tune=not args.no_tune, **kw)
        else:
            r = crossfit(concat([sets[s] for s in splits]), ks, folds=args.folds, seeds=args.seeds,
                         tune=not args.no_tune, **kw)
        for k in ks:
            losses[prefix + k], acc[prefix + k], hp[prefix + k] = r["losses"][k], r["acc"][k], r["hp"][k]

    print(f"[{args.rep}] n={ {s: len(f['label']) for s, f in feats.items()} }", flush=True)
    run(args.rep, kinds)
    for ref in [r for r in args.refs.split(",") if r]:
        run(ref, ["add"], prefix=ref + ":")
    ceil_key = "whole_d1:add" if "whole_d1:add" in losses else None
    Lceil = float(losses[ceil_key].mean()) if ceil_key else float("nan")
    summ = dict(rep=args.rep, mode="holdout" if args.holdout else "crossfit", n={s: len(f["label"]) for s, f in feats.items()},
                seeds=args.seeds, ceiling_model=ceil_key, ceiling_loss=Lceil, hp=hp, models={})
    for k, L in losses.items():
        row = dict(loss=float(L.mean()), acc=float(np.mean(acc[k])), acc_sd=float(np.std(acc[k])))
        if k != "add" and "add" in losses:
            d = paired_bootstrap(losses["add"], L, B=2000)
            row["vs_add"] = {**d.as_dict(), **pass_rule(d, float(losses["add"].mean()), Lceil)}
        summ["models"][k] = row
        v = row.get("vs_add")
        print(f"  {k:16s} loss {row['loss']:.4f} acc {row['acc']:.3f}" + ("" if not v else
              f"  d_vs_add {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] share {v['share_of_headroom']:.2f} pass={v['passed']}"), flush=True)
    summ["sec"] = round(time.time() - t0, 1)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path(f"docs/results/e2_{tag}.json").write_text(json.dumps(summ, indent=1))
    np.savez_compressed(f"runs/e2_{tag}_losses.npz", **{k.replace(":", "__"): v.astype(np.float32) for k, v in losses.items()})
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
