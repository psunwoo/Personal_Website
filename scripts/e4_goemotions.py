"""E4, stage 1: body order and operators on GoEmotions (distribution over 28 emotion labels per comment).

Also freezes one additive and one gated model and stores what stage 2 (per-rater couplings) needs:
per-comment logits, pooled one-body features, and for the gated model the per-token operator terms.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rcf import goemotions as ge
from rcf.compare import holdout
from rcf.data import SetData
from rcf.embed import embed_texts
from rcf.stats import paired_bootstrap, pass_rule
from rcf.train import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="add,gate,scope_l,poly2,deepsets")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-whole", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    com, rat, emo, Y = ge.load()
    idx = {s: np.where(com.split == s)[0] for s in ("train", "val", "test")}
    tr, va, te = (ge.comment_sets(com, rat, Y, emo, rows=idx[s]) for s in ("train", "val", "test"))
    print(f"[goemotions] comments {len(com)} train {len(tr)} val {len(va)} test {len(te)} d={tr.X.shape[-1]}", flush=True)
    kw = dict(batch=512, max_epochs=40, patience=5)
    kinds = args.kinds.split(",")
    r = holdout(tr, va, te, kinds, seeds=args.seeds, tune=False, keep_models=True, **kw)
    losses, acc = dict(r["losses"]), dict(r["acc"])
    if not args.no_whole:
        E = embed_texts(com.text.tolist(), "mpnet")
        ws = {s: SetData(E[idx[s]][:, None, :], np.ones((len(idx[s]), 1), dtype=bool), d.y, "dist", emo)
              for s, d in (("train", tr), ("val", va), ("test", te))}
        w = holdout(ws["train"], ws["val"], ws["test"], ["add"], seeds=args.seeds, tune=False, **kw)
        losses["whole_mpnet"], acc["whole_mpnet"] = w["losses"]["add"], w["acc"]["add"]
    # entropy of the test targets: the cross-entropy no model can beat on these empirical distributions
    p = np.clip(te.y, 1e-12, 1)
    floor = float(-(te.y * np.log(p)).sum(1).mean())
    refs = [k for k in ("deepsets", "whole_mpnet") if k in losses]
    ceil_key = min(refs, key=lambda k: losses[k].mean())
    Lceil = float(losses[ceil_key].mean())
    summ = dict(n=dict(train=len(tr), val=len(va), test=len(te)), ceiling_model=ceil_key, ceiling_loss=Lceil,
                target_entropy=floor, models={})
    for k, L in losses.items():
        row = dict(loss=float(L.mean()), acc=float(np.mean(acc[k])))
        if k != "add":
            d = paired_bootstrap(losses["add"], L, B=2000)
            row["vs_add"] = {**d.as_dict(), **pass_rule(d, float(losses["add"].mean()), Lceil)}
        summ["models"][k] = row
        v = row.get("vs_add")
        print(f"  {k:12s} loss {row['loss']:.4f} top-1 {row['acc']:.3f}" + ("" if not v else
              f"  d {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] share {v['share_of_headroom']:.2f} pass={v['passed']}"), flush=True)
    summ["sec"] = round(time.time() - t0, 1)
    Path("docs/results/e4_goemotions.json").write_text(json.dumps(summ, indent=1))
    np.savez_compressed("runs/e4_goemotions_losses.npz", **{k: v.astype(np.float32) for k, v in losses.items()})

    # freeze shared models for stage 2 (arrays indexed by comment, in the order of `com`)
    out = {}
    n_all = len(com)

    def scatter(parts: list[np.ndarray]) -> np.ndarray:
        full = np.zeros((n_all,) + parts[0].shape[1:], dtype=parts[0].dtype)
        for s, p_ in zip(("train", "val", "test"), parts):
            full[idx[s]] = p_
        return full

    for k in ("add", "gate"):
        if k not in r["models"]:
            continue
        f = r["models"][k][0]
        m = f.model
        logits, A, T, gain, bias = [], [], [], [], []
        with torch.no_grad():
            for d in (tr, va, te):
                logits.append(predict(f, d.X, d.mask).astype(np.float32))
                a_, t_ = [], []
                for s in range(0, len(d), 2048):
                    x, mk = torch.from_numpy(d.X[s:s + 2048]), torch.from_numpy(d.mask[s:s + 2048])
                    mf = mk.float().unsqueeze(-1)
                    a_.append(((m.phi(x) * mf).sum(1) / mf.sum(1).clamp(min=1)).numpy())
                    if k == "gate":
                        t_.append((m.factors(x, mk) - 1.0).numpy().astype(np.float16))  # 2 tanh(delta); 0 where absent
                A.append(np.concatenate(a_).astype(np.float32))
                if k == "gate":
                    T.append(np.concatenate(t_))
                    nn_ = torch.from_numpy(d.mask).float().sum(1, keepdim=True).clamp(min=1)
                    sz = m.size(torch.cat([torch.log(nn_), 1.0 / nn_], 1)).numpy()
                    gain.append(np.exp(sz[:, :28]).astype(np.float32))
                    bias.append(sz[:, 28:].astype(np.float32))
        out[f"{k}_logits"], out[f"{k}_A"] = scatter(logits), scatter(A)
        if k == "gate":
            out["gate_T"], out["gate_gain"], out["gate_bias"] = scatter(T), scatter(gain), scatter(bias)
            with torch.no_grad():
                out["gate_lin"] = m.lin(torch.from_numpy(out["gate_A"])).numpy().astype(np.float32)
    np.savez("runs/e4_frozen.npz", **out)
    print("frozen:", {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
