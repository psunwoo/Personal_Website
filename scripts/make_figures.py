"""Figures for docs/RESULTS.md, built from the JSON files in docs/results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("docs/results")
OUT = R / "figs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def load(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def q1_rows() -> list[tuple[str, float, float, float, bool]]:
    """(label, share of headroom, lo, hi, passed) for the pre-specified second-order model (poly2)."""
    rows = []
    for tag, lab in (("vad_glove", "Two-word expressions"), ("conc_minilm", "Two-word expressions")):
        d = load(f"e1_{tag}.json")
        if not d:
            continue
        for node, v in zip(d["nodes"], d["models"]["poly2"]["vs_add"]):
            h = v["headroom"]
            rows.append((f"{lab}: {node}", v["mean"] / h, v["lo"] / h, v["hi"] / h, v["passed"]))
    d = load("e1c_sst_sets_minilm+wordsent.json")
    if d:
        v = d["models"]["poly2"]["vs_add"]
        rows.append(("Treebank phrases (2-12 words): sentiment", v["mean"] / v["headroom"], v["lo"] / v["headroom"], v["hi"] / v["headroom"], v["passed"]))
    base = load("e2_tiles2_crossfit.json")
    ceil = base["ceiling_loss"] if base else None
    for rep, lab in (("tiles2", "Image, 2x2 tiles"), ("tiles3", "Image, 3x3 tiles"), ("tiles4", "Image, 4x4 tiles"),
                     ("tags", "Image, concept tags"), ("attrs", "Image, dataset attributes")):
        d = load(f"e2_{rep}_crossfit.json")
        if not d or ceil is None:
            continue
        v = d["models"]["poly2"]["vs_add"]
        h = d["models"]["add"]["loss"] - ceil
        rows.append((f"{lab}: emotion", v["mean"] / h, v["lo"] / h, v["hi"] / h, v["lo"] > 0 and v["mean"] / h >= 0.1))
    return rows


def fig_q1() -> None:
    rows = q1_rows()
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 0.44 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, lo, hi, ok) in zip(y, rows):
        ax.plot([100 * lo, 100 * hi], [yi, yi], color=C[0], lw=2, solid_capstyle="round")
        ax.plot(100 * m, yi, "o", color=C[0], ms=8, mec=SURFACE, mew=2)
        ax.text(100 * hi + 1.2, yi, f"{100 * m:.1f}%", va="center", fontsize=9, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axvline(10, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(10.6, len(rows) - 0.5, "pass threshold (10%)", fontsize=9, color=INK2, va="bottom")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("held-out gain of the second-order model, % of headroom (95% CI)")
    ax.set_title("Q1: second-order gain over the additive model")
    fig.savefig(OUT / "q1_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_e0() -> None:
    p = R / "e0_synthetic.jsonl"
    if not p.exists():
        return
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, n in enumerate((4000, 16000)):
        rr = sorted([r for r in rows if r["n_train"] == n and r["rep"] == 0], key=lambda r: r["eta"])
        x = [max(r["eta"], 0.004) for r in rr]
        m = [100 * r["poly2"]["share_of_headroom"] for r in rr]
        lo = [100 * r["poly2"]["lo"] / r["poly2"]["headroom"] for r in rr]
        hi = [100 * r["poly2"]["hi"] / r["poly2"]["headroom"] for r in rr]
        ax.fill_between(x, lo, hi, color=C[i], alpha=0.15, lw=0)
        ax.plot(x, m, "-o", color=C[i], mec=SURFACE, mew=2, ms=8, label=f"{n:,} training sets")
        ax.text(x[-1] * 1.08, m[-1], f"{n // 1000}k", color=INK2, va="center", fontsize=9)
    ax.axhline(10, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.0042, 11.5, "pass threshold", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.004, 0.01, 0.03, 0.1, 0.3], ["0", "0.01", "0.03", "0.1", "0.3"])
    ax.set_xlim(0.0035, 0.42)
    ax.set_xlabel("planted interaction share of logit variance")
    ax.set_ylabel("gain, % of headroom")
    ax.set_title("Calibration: what the test can detect")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "e0_power.png", dpi=200)
    plt.close(fig)


def fig_q2(tag: str) -> None:
    d = load(f"e3_rg_{tag}.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = {"4": "4x4 tiles (56 px)", "3": "3x3 tiles (70 px)", "2": "2x2 tiles (112 px)"}
    for i, g in enumerate(("4", "3", "2")):
        r2 = np.array(d["spectrum"][g]["rho2"])
        ax.plot(np.arange(1, len(r2) + 1), r2, color=C[i], label=names[g])
    null = max(d["spectrum"][g]["rho2_null_p99"] for g in ("4", "3", "2"))
    ax.axhline(null, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(1.05, null + 0.025, "null: environment taken from another image", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("direction rank k (of 384)")
    ax.set_ylabel("held-out canonical correlation, squared")
    ax.set_title("Q2: the relevance spectrum decays smoothly, with no gap")
    ax.legend(loc="upper right", title="block size")
    fig.tight_layout()
    fig.savefig(OUT / "q2_spectrum.png", dpi=200)
    plt.close(fig)

    rows = d["decode"]["4"]
    ks = sorted({int(k.split("@")[1]) for k in rows if "@" in k})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    panels = [("emotion_acc", "evoked emotion (accuracy)"), ("brightness_r2", "brightness ($R^2$)"), ("colorfulness_r2", "colourfulness ($R^2$)")]
    series = [("relevant", "most relevant (top CCA)"), ("pca", "highest variance (PCA)"), ("random", "random"), ("irrelevant", "least relevant")]
    for ax, (m, title) in zip(axes, panels):
        for i, (s, lab) in enumerate(series):
            v = [rows[f"{s}@{k}"][m] for k in ks]
            ax.plot(ks, v, "-o", color=C[i], mec=SURFACE, mew=1.5, ms=6, label=lab)
        ax.axhline(rows["full"][m], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_title(title)
        ax.set_xlabel("directions kept per tile")
    axes[0].legend(loc="upper left", fontsize=8)
    for ax, (m, _) in zip(axes, panels):
        ax.text(ks[-1], rows["full"][m], "all 384 ", fontsize=8, color=INK2, va="top", ha="right")
    fig.suptitle("Q2: keep only k directions per 56-px tile, then decode linearly", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "q2_truncation.png", dpi=200)
    plt.close(fig)


def fig_shortlist() -> None:
    d = load("e5b_short_list.json")
    if not d:
        return
    full = d["full_gate_gain"]["mean"]
    xs = [r["k"] for r in d["lists"]] + [3000]
    m = [100 * r["gain"]["mean"] / full for r in d["lists"]] + [100.0]
    lo = [100 * r["gain"]["lo"] / full for r in d["lists"]] + [100 * d["full_gate_gain"]["lo"] / full]
    hi = [100 * r["gain"]["hi"] / full for r in d["lists"]] + [100 * d["full_gate_gain"]["hi"] / full]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(xs, lo, hi, color=C[0], alpha=0.15, lw=0)
    ax.plot(xs, m, "-o", color=C[0], mec=SURFACE, mew=2, ms=8)
    for x, v in zip(xs, m):
        ax.text(x, v + 9, f"{v:.0f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs[:-1]] + ["all words"])
    ax.minorticks_off()
    ax.set_ylim(0, 150)
    ax.set_xlabel("words allowed to act as operators (strongest first)")
    ax.set_ylabel("% of the full operator gain")
    ax.set_title("A short list gives roughly half of the operator gain")
    fig.tight_layout()
    fig.savefig(OUT / "operators_short_list.png", dpi=200)
    plt.close(fig)


def fig_onboarding() -> None:
    d, p = load("e4c_onboarding.json"), load("e4_personal.json")
    if not d or not p:
        return
    op = p["bases"]["add"]["shared"]["loss"] - p["bases"]["gate"]["shared"]["loss"]
    xs = [r["n"] if i < len(d["curve"]) - 1 else 2700 for i, r in enumerate(d["curve"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    names = [("lowrank", "+ personal reading of content"), ("gain", "+ personal response strength"), ("field", "personal field only")]
    for i, (k, lab) in enumerate(names):
        m = [r[k]["mean"] for r in d["curve"]]
        ax.fill_between(xs, [r[k]["lo"] for r in d["curve"]], [r[k]["hi"] for r in d["curve"]], color=C[i], alpha=0.15, lw=0)
        ax.plot(xs, m, "-o", color=C[i], mec=SURFACE, mew=1.5, ms=6, label=lab)
    ax.axhline(op, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(xs[-1], op - 0.006, "gain from operators (shared model)", fontsize=9, color=INK2, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs[:-1]] + ["all"])
    ax.minorticks_off()
    ax.set_xlabel("ratings available from the person")
    ax.set_ylabel("held-out gain, nats per rating")
    ax.set_title("Personal couplings: what a person's own ratings buy")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "personal_onboarding.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    fig_e0()
    fig_q1()
    fig_q2(sys.argv[1] if len(sys.argv) > 1 else "test")
    fig_shortlist()
    fig_onboarding()
    print(sorted(p.name for p in OUT.glob("*.png")))
