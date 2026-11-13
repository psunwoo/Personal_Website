"""Print the markdown tables used in docs/RESULTS.md from docs/results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

R = Path("docs/results")


def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def ci(v, s=1.0, f="{:+.4f}"):
    return f"{f.format(s * v['mean'])} [{f.format(s * v['lo'])}, {f.format(s * v['hi'])}]"


def e0():
    rows = [json.loads(l) for l in (R / "e0_synthetic.jsonl").read_text().splitlines()]
    print("| training sets | planted share $\\eta$ | headroom | $\\Delta_2$ (ACE form) | % of headroom | passes |\n|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["n_train"], r["eta"], r["rep"])):
        if r["rep"] > 0:
            continue
        p = r["poly2"]
        print(f"| {r['n_train']:,} | {r['eta']} | {p['headroom']:.4f} | {ci(p)} | {100 * p['share_of_headroom']:.0f}% | {'yes' if p['passed'] else 'no'} |")
    nul = [r for r in rows if r["eta"] == 0]
    print(f"\nNull replicates: {len(nul)}; passes: poly2 {sum(r['poly2']['passed'] for r in nul)}, pair {sum(r['pair']['passed'] for r in nul)}; "
          f"lower CI bound above 0: poly2 {sum(r['poly2']['significant'] for r in nul)}, pair {sum(r['pair']['significant'] for r in nul)}.\n")


def e1():
    print("| features | node | additive $R^2$ | $\\Delta_2$ ACE form | $\\Delta$ Deep Sets | ceiling $R^2$ | % of headroom (ACE) | passes |\n|---|---|---|---|---|---|---|---|")
    for tag in ("vad_glove", "vad_minilm", "vad_glove+norms_rated", "conc_minilm"):
        d = load(f"e1_{tag}.json")
        if not d:
            continue
        for i, node in enumerate(d["nodes"]):
            a, p, ds = d["models"]["add"], d["models"]["poly2"]["vs_add"][i], d["models"]["deepsets"]["vs_add"][i]
            print(f"| {tag} | {node} | {a['r2'][i]:.3f} | {ci(p)} | {ci(ds)} | {1 - d['ceiling_mse'][i]:.3f} | {100 * p['share_of_headroom']:.1f}% | {'yes' if p['passed'] else 'no'} |")
    for tag in ("vad_minilm", "conc_minilm"):
        d = load(f"e1_{tag}.json")
        if d:
            for k in [k for k in d["models"] if k.startswith("whole")]:
                print(f"\n{tag} {k}: R2 {[round(x, 3) for x in d['models'][k]['r2']]}; vs additive {[ci(v) for v in d['models'][k]['vs_add']]}")
    g = load("e1_norms_gbm.json")
    if g:
        print("\n| target | node | order-1 $R^2$ | order-2 $R^2$ | $\\Delta_2$ |\n|---|---|---|---|---|")
        for t, r in g.items():
            for i, node in enumerate(r["nodes"]):
                print(f"| {t} (n={r['n']:,}) | {node} | {r['models']['order1']['r2'][i]:.3f} | {r['models']['order2']['r2'][i]:.3f} | {ci(r['models']['order2']['vs_order1'][i])} |")


def sst():
    a = load("e1c_sst_tree.json")
    if a:
        print(f"\nTree nodes: train {a['n_train']:,}, test {a['n_test']:,}")
        for k in ("children_additive", "children_interacting", "children+sizes_additive", "children+sizes_interacting"):
            print(f"  {k}: R2 {a[k]['r2']:.4f}" + (f"  vs additive {ci(a[k]['vs_additive'])}" if "vs_additive" in a[k] else ""))
        print("  by size:", {k: (v["n"], round(v["r2_additive"], 3), round(v["r2_interacting"], 3)) for k, v in a["by_size"].items()})
    for tag in ("minilm+wordsent", "minilm"):
        d = load(f"e1c_sst_sets_{tag}.json")
        if not d:
            continue
        print(f"\nSST sets [{tag}] n={d['n']} ceiling model {d['ceiling_model']} (R2 {1 - d['ceiling_mse']:.3f})")
        print("| model | $R^2$ | $\\Delta$ vs additive | % of headroom | passes |\n|---|---|---|---|---|")
        for k, m in d["models"].items():
            v = m.get("vs_add")
            print(f"| {k} | {m['r2']:.4f} | " + ("| | |" if not v else f"{ci(v)} | {100 * v['share_of_headroom']:.0f}% | {'yes' if v['passed'] else 'no'} |"))
        print("by size:", {k: {m: ci(v[f'd_{m}']) for m in ("poly2", "poly3") if f"d_{m}" in v} for k, v in d["by_size"].items()})


def e2():
    base = load("e2_tiles2_crossfit.json")
    if not base:
        return
    ceil = base["ceiling_loss"]
    print(f"\nwhole-image DINOv2: loss {ceil:.4f} acc {base['models']['whole_d1:add']['acc']:.3f}; whole-image CLIP: loss "
          f"{base['models']['whole_clip:add']['loss']:.4f} acc {base['models']['whole_clip:add']['acc']:.3f}")
    print("| objects | additive loss (acc) | $\\Delta_2$ ACE | $\\Delta_3$ ACE | $\\Delta$ explicit pair | $\\Delta$ Deep Sets | $\\Delta$ Set Transformer | % of headroom ($\\Delta_2$) | passes |\n|---|---|---|---|---|---|---|---|---|")
    for rep in ("tiles2", "tiles3", "tiles4", "tags", "tags_norms", "attrs"):
        d = load(f"e2_{rep}_crossfit.json")
        if not d:
            continue
        m = d["models"]
        h = m["add"]["loss"] - ceil
        cell = lambda k: ci(m[k]["vs_add"]) if k in m else "n/a"
        v = m["poly2"]["vs_add"]
        ok = v["lo"] > 0 and v["mean"] / h >= 0.1
        print(f"| {rep} | {m['add']['loss']:.4f} ({m['add']['acc']:.3f}) | {cell('poly2')} | {cell('poly3')} | {cell('pair')} | {cell('deepsets')} | {cell('settrans')} | {100 * v['mean'] / h:.1f}% | {'yes' if ok else 'no'} |")


def e3(tag="test"):
    d = load(f"e3_rg_{tag}.json")
    if not d:
        return
    print(f"\nQ2 [{tag}] fit {d['n_fit']:,} eval {d['n_eval']:,} images")
    print("| block | top $\\rho^2$ | directions with $\\rho^2>0.25$ | $>0.05$ | participation ratio | largest $\\rho^2_k/\\rho^2_{k+1}$ (k<=40) | null max | top-8 subspace overlap (chance) | top-32 overlap (chance) |\n|---|---|---|---|---|---|---|---|---|")
    for g in ("4", "3", "2"):
        s = d["spectrum"][g]
        print(f"| {g}x{g} | {s['rho2'][0]:.2f} | {s['n_above']['0.25']} | {s['n_above']['0.05']} | {s['participation_ratio']:.0f} | {s['largest_gap_ratio']:.2f} | {s['rho2_null_max']:.4f} | "
              f"{s['stability']['8']:.2f} ({s['stability_chance']['8']:.2f}) | {s['stability']['32']:.2f} ({s['stability_chance']['32']:.2f}) |")
    for g in ("4", "2"):
        rows = d["decode"][g]
        ks = sorted({int(k.split("@")[1]) for k in rows if k.startswith("relevant@")})
        print(f"\nTruncation, {g}x{g} tiles (full: acc {rows['full']['emotion_acc']:.3f}, brightness {rows['full']['brightness_r2']:.2f}, colourfulness {rows['full']['colorfulness_r2']:.2f})")
        print("| k | emotion acc: relevant | PCA | random | least relevant | relevant - PCA | brightness $R^2$: relevant | PCA | colourfulness $R^2$: relevant | PCA |\n|---|---|---|---|---|---|---|---|---|---|")
        for k in ks:
            r = lambda n, m: rows[f"{n}@{k}"][m]
            dd = rows.get(f"relevant_minus_pca@{k}")
            print(f"| {k} | {r('relevant', 'emotion_acc'):.3f} | {r('pca', 'emotion_acc'):.3f} | {r('random', 'emotion_acc'):.3f} | {r('irrelevant', 'emotion_acc'):.3f} | "
                  f"{ci(dd, f='{:+.3f}') if dd else ''} | {r('relevant', 'brightness_r2'):.2f} | {r('pca', 'brightness_r2'):.2f} | {r('relevant', 'colorfulness_r2'):.2f} | {r('pca', 'colorfulness_r2'):.2f} |")
    print("\nFlow (4x4 -> 2x2 -> image), emotion accuracy after step 2:")
    for k, v in d["flow"].items():
        print(f"  k={k}: relevant {v['relevant']['step2']['emotion_acc']:.3f}, PCA {v['pca']['step2']['emotion_acc']:.3f}, diff {ci(v['step2_relevant_minus_pca'], f='{:+.3f}') if 'step2_relevant_minus_pca' in v else ''}; "
              f"step-2 top rho2 relevant {v['relevant']['step2_rho2'][0]:.2f}")
    print("whole image, linear:", d["whole_d1"])


if __name__ == "__main__":
    import sys

    e0(); e1(); sst(); e2(); e3(sys.argv[1] if len(sys.argv) > 1 else "test")
