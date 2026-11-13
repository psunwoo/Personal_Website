"""E3 (Q2): relevance spectrum of a linear RSMI coarse-graining on image tiles.

A. Spectrum.  For block size g in {4x4, 3x3, 2x2 tiles}: canonical correlations rho_k between a
   block's embedding and its environment (the surrounding tiles), fitted on one set of images and
   re-measured on held-out images.  Null: environment taken from a different image.  Stability:
   overlap of the top-k subspaces fitted on two disjoint halves.
B. Conjecture test.  Keep only k directions per block, chosen as (i) most relevant (top CCA),
   (ii) highest variance (PCA), (iii) least relevant (bottom CCA), (iv) random.  Decode an abstract
   label (evoked emotion) and two global sensory ones (brightness, colourfulness) linearly from
   the kept coordinates.  "Abstract concepts are the relevant directions, sensory detail is
   irrelevant" predicts: emotion survives truncation to the top relevant directions better than the
   sensory attributes do, and better than under PCA truncation.
C. Flow.  Two coarse-graining steps (4x4 -> 2x2 -> 1) and the spectrum at each step.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from rcf import emoset
from rcf.rg import CCA, block_environment, participation_ratio, subspace_overlap


def zscore(fit: np.ndarray, *others: np.ndarray):
    """Standardise with fit-split statistics, then remove the mean of each block position.

    Each position has its own average appearance (sky at the top, ground at the bottom).  That mean
    is shared by all images, so leaving it in creates block-environment correlation even between
    blocks of different images.
    """
    fit = fit.astype(np.float32)
    mu, sd = fit.reshape(-1, fit.shape[-1]).mean(0), fit.reshape(-1, fit.shape[-1]).std(0) + 1e-6
    z = [((a - mu) / sd).astype(np.float32) for a in (fit, *others)]
    if fit.ndim == 3:
        pos = z[0].mean(0, keepdims=True)
        z = [a - pos for a in z]
    return z


HITS: list[np.ndarray] = []  # per-image correctness of the most recent decode() calls


def acc_diff(a: np.ndarray, b: np.ndarray, B: int = 2000) -> dict:
    """Paired bootstrap over evaluation images for an accuracy difference a - b."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(B)])
    return dict(mean=float(d.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)))


def decode(Ztr: np.ndarray, Zte: np.ndarray, tr: dict, te: dict) -> dict:
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    clf = LogisticRegression(C=0.1, max_iter=300).fit(Ztr, tr["label"])
    P = clf.predict_proba(Zte)
    HITS.append(P.argmax(1) == te["label"])
    out = dict(emotion_acc=float((P.argmax(1) == te["label"]).mean()),
               emotion_ce=float(-np.log(np.clip(P[np.arange(len(P)), te["label"]], 1e-12, None)).mean()))
    for k in ("brightness", "colorfulness"):
        ok_tr, ok_te = np.isfinite(tr[k]), np.isfinite(te[k])
        r = Ridge(alpha=10.0).fit(Ztr[ok_tr], tr[k][ok_tr])
        res = te[k][ok_te] - r.predict(Zte[ok_te])
        out[k + "_r2"] = float(1 - (res ** 2).mean() / te[k][ok_te].var())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-splits", default="test,val")
    ap.add_argument("--eval-splits", default="")
    ap.add_argument("--max-fit", type=int, default=40000)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    keys = ["d1", "d2", "d3", "d4", "label", "brightness", "colorfulness"]

    def get(splits: str) -> dict:
        parts = [emoset.load(s, keys=keys) for s in splits.split(",")]
        return {k: np.concatenate([p[k] for p in parts]) for k in keys}

    rng = np.random.default_rng(0)
    if args.eval_splits:
        F, E = get(args.fit_splits), get(args.eval_splits)
        if len(F["label"]) > args.max_fit:
            i = rng.permutation(len(F["label"]))[: args.max_fit]
            F = {k: v[i] for k, v in F.items()}
    else:  # split the available images 70 / 30
        A = get(args.fit_splits)
        p = rng.permutation(len(A["label"]))
        c = int(0.7 * len(p))
        F, E = {k: v[p[:c]] for k, v in A.items()}, {k: v[p[c:]] for k, v in A.items()}
    tr = dict(label=F["label"].astype(int), brightness=F["brightness"], colorfulness=F["colorfulness"])
    te = dict(label=E["label"].astype(int), brightness=E["brightness"], colorfulness=E["colorfulness"])
    nF, nE = len(tr["label"]), len(te["label"])
    print(f"fit images {nF}, eval images {nE}", flush=True)
    ks = [int(k) for k in args.ks.split(",")]
    res = dict(n_fit=nF, n_eval=nE, spectrum={}, decode={}, flow={})

    for g in (4, 3, 2):
        Tf, Te = zscore(F[f"d{g}"], E[f"d{g}"])
        Xf, Yf, _ = block_environment(Tf, g)
        Xe, Ye, _ = block_environment(Te, g)
        cca = CCA(Xf, Yf)
        rho = cca.heldout_rho(Xe, Ye)
        perm = rng.permutation(nE)
        _, Ynull, _ = block_environment(Te[perm], g)
        rho_null = cca.heldout_rho(Xe, Ynull)
        r2 = np.clip(rho, 0, 1) ** 2
        info = -0.5 * np.log(1 - np.clip(r2, 0, 0.999999))
        h = nF // 2
        ca = CCA(*block_environment(Tf[:h], g)[:2])
        cb = CCA(*block_environment(Tf[h:], g)[:2])
        ratio = r2[:40] / np.maximum(r2[1:41], 1e-9)
        spec = dict(rho2=r2.tolist(), rho2_null_max=float(np.max(rho_null ** 2)),
                    rho2_null_p99=float(np.quantile(rho_null ** 2, 0.99)),
                    n_above={str(t): int((r2 > t).sum()) for t in (0.5, 0.25, 0.1, 0.05)},
                    info_nats_total=float(info.sum()), info_share_top={str(k): float(info[:k].sum() / info.sum()) for k in ks},
                    participation_ratio=participation_ratio(info), largest_gap_ratio=float(ratio.max()),
                    largest_gap_after_k=int(ratio.argmax() + 1),
                    stability={str(k): subspace_overlap(ca.A[:, :k], cb.A[:, :k]) for k in ks},
                    stability_chance={str(k): k / Xf.shape[1] for k in ks})
        res["spectrum"][str(g)] = spec
        print(f"g={g}: rho2 top8 {np.round(r2[:8], 3)} n>0.25: {spec['n_above']['0.25']} PR {spec['participation_ratio']:.1f} "
              f"gap x{spec['largest_gap_ratio']:.2f} after k={spec['largest_gap_after_k']} null max {spec['rho2_null_max']:.4f}", flush=True)

        # B. truncation test
        Xc = Xf - Xf.mean(0)
        _, _, Vt = np.linalg.svd(Xc[:: max(len(Xc) // 100_000, 1)], full_matrices=False)
        L, d = g * g, Xf.shape[1]
        rows = {"full": decode(Tf.reshape(nF, -1), Te.reshape(nE, -1), tr, te)}
        for k in ks:
            projs = {"relevant": cca.A[:, :k], "pca": Vt[:k].T, "irrelevant": cca.A[:, -k:]}
            for j in range(3):
                projs[f"random{j}"] = np.linalg.qr(np.random.default_rng(100 + j).normal(size=(d, k)))[0]
            hit = {}
            for name, P in projs.items():
                P = P.astype(np.float32)
                Zf = ((Tf.reshape(-1, d) - cca.mx.astype(np.float32)) @ P).reshape(nF, L * k)
                Ze = ((Te.reshape(-1, d) - cca.mx.astype(np.float32)) @ P).reshape(nE, L * k)
                rows[f"{name}@{k}"] = decode(Zf, Ze, tr, te)
                hit[name] = HITS[-1]
            rows[f"relevant_minus_pca@{k}"] = acc_diff(hit["relevant"], hit["pca"])
            rows[f"relevant_minus_random@{k}"] = acc_diff(hit["relevant"], hit["random0"])
            rnd = [rows.pop(f"random{j}@{k}") for j in range(3)]
            rows[f"random@{k}"] = {m: float(np.mean([r[m] for r in rnd])) for m in rnd[0]}
            dd = rows[f"relevant_minus_pca@{k}"]
            print(f"  k={k:3d} rel-pca acc {dd['mean']:+.3f} [{dd['lo']:+.3f},{dd['hi']:+.3f}] " + " | ".join(f"{nm}: acc {rows[f'{nm}@{k}']['emotion_acc']:.3f} br {rows[f'{nm}@{k}']['brightness_r2']:.2f} "
                                            f"col {rows[f'{nm}@{k}']['colorfulness_r2']:.2f}" for nm in ("relevant", "pca", "irrelevant", "random")), flush=True)
        print(f"  full: {rows['full']}", flush=True)
        res["decode"][str(g)] = rows

    # C. two-step flow on the 4x4 grid
    Tf, Te = zscore(F["d4"], E["d4"])
    d = Tf.shape[-1]
    quad = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    for k1 in (8, 16, 32):
        out = {}
        for method in ("relevant", "pca"):
            Xf, Yf, _ = block_environment(Tf, 4)
            c1 = CCA(Xf, Yf)
            if method == "relevant":
                P1 = c1.A[:, :k1]
            else:
                P1 = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)[2][:k1].T
            Bf = ((Tf.reshape(-1, d) - c1.mx) @ P1).reshape(nF, 16, k1)[:, quad].reshape(nF, 4, 4 * k1)
            Be = ((Te.reshape(-1, d) - c1.mx) @ P1).reshape(nE, 16, k1)[:, quad].reshape(nE, 4, 4 * k1)
            Bf, Be = zscore(Bf, Be)
            X2f, Y2f, _ = block_environment(Bf, 2)
            X2e, Y2e, _ = block_environment(Be, 2)
            c2 = CCA(X2f, Y2f)
            r2 = np.clip(c2.heldout_rho(X2e, Y2e), 0, 1) ** 2
            k2 = k1
            P2 = c2.A[:, :k2] if method == "relevant" else np.linalg.svd(X2f - X2f.mean(0), full_matrices=False)[2][:k2].T
            Zf = ((Bf.reshape(-1, 4 * k1) - c2.mx) @ P2).reshape(nF, 4 * k2)
            Ze = ((Be.reshape(-1, 4 * k1) - c2.mx) @ P2).reshape(nE, 4 * k2)
            out[method] = dict(step1=decode(Bf.reshape(nF, -1), Be.reshape(nE, -1), tr, te), step2=decode(Zf, Ze, tr, te),
                               step2_rho2=r2.tolist())
            out[method + "_hit"] = HITS[-1]
            print(f"flow k={k1} {method}: step1 acc {out[method]['step1']['emotion_acc']:.3f} -> step2 acc {out[method]['step2']['emotion_acc']:.3f}; "
                  f"step2 rho2 top5 {np.round(r2[:5], 3)}", flush=True)
        out["step2_relevant_minus_pca"] = acc_diff(out.pop("relevant_hit"), out.pop("pca_hit"))
        print(f"   step2 relevant - pca accuracy: {out['step2_relevant_minus_pca']}", flush=True)
        res["flow"][str(k1)] = out
    D1f, D1e = zscore(F["d1"], E["d1"])
    res["whole_d1"] = decode(D1f, D1e, tr, te)
    print("whole d1:", res["whole_d1"], flush=True)
    res["sec"] = round(time.time() - t0, 1)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path(f"docs/results/e3_rg{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
