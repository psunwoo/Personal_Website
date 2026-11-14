"""E5: anatomy of the learned operators (treebank), and whether they transfer.

A. Sparsity.  Fit the gated model, rank words by |m - 1|, then switch off every operator except
   the top k (m = 1 for the rest) and re-score the held-out phrases without refitting.  How short
   can the operator list be?
B. Transfer.  Take the factor each word learned on the treebank (movie-review sentiment) and use
   it, frozen, on NRC two-word expressions (valence norms from a different rating study):
       add          additive model
       add+feat     additive model that also sees the treebank factor as an input feature
       fgate        additive model multiplied by the frozen treebank factors
       fgate_shuf   as fgate, with the factors shuffled across words (control)
       gate         operators learned on the NRC data themselves
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcf import lexical, sst
from rcf.compare import crossfit
from rcf.data import SetData
from rcf.stats import paired_bootstrap
from rcf.train import fit

torch.set_num_threads(1)
FE = ("minilm", "wordsent")
nodes, dic = sst.load()
tab = sst.phrase_table(nodes)
rng = np.random.default_rng(0)
idx = {s: np.where(tab.split == s)[0] for s in ("train", "validation", "test")}
idx["train"] = rng.permutation(idx["train"])[:60000]
idx["test"] = rng.permutation(idx["test"])[:20000]
idx["validation"] = rng.permutation(idx["validation"])[:8000]
tr, va, te = (sst.phrase_sets(tab.iloc[idx[s]].reset_index(drop=True), dic, FE) for s in ("train", "validation", "test"))
cnt = Counter(w for p in tab.phrase.iloc[idx["train"]] for w in p.split(" "))
words = sorted(w for w, c in cnt.items() if c >= 5)
single = pd.DataFrame(dict(phrase=words, size=1, label=0.5, split="train"))
W = sst.phrase_sets(single, dic, FE, max_size=1)
W.X[:, :, -2:] = 0.5
te_tokens = [p.split(" ") for p in tab.phrase.iloc[idx["test"]]]
yte = torch.from_numpy(te.y)

fits = [fit(tr, va, "gate", seed=s, batch=512, max_epochs=40, patience=5) for s in range(3)]
with torch.no_grad():
    M = np.stack([1.0 + 2.0 * torch.tanh(f.model.delta(torch.from_numpy(W.X)))[:, 0, 0].numpy() for f in fits])
m_word = M.mean(0)
rank = np.argsort(-np.abs(m_word - 1.0))
order = {words[i]: r for r, i in enumerate(rank)}


def loss_with_top(f, k: int) -> np.ndarray:
    """Held-out squared error with only the k highest-ranked operator words left on."""
    keep = np.zeros(te.mask.shape, dtype=bool)
    for i, t in enumerate(te_tokens):
        for j, w in enumerate(t[: te.mask.shape[1]]):
            keep[i, j] = order.get(w, 10**9) < k
    with torch.no_grad():
        m, x, mk = f.model, torch.from_numpy(te.X), torch.from_numpy(te.mask)
        n = mk.float().sum(1, keepdim=True).clamp(min=1)
        A = (m.phi(x) * mk.float().unsqueeze(-1)).sum(1) / n
        fac = torch.where(torch.from_numpy(keep).unsqueeze(-1), m.factors(x, mk), torch.ones(1))
        z = m.lin(A) * fac.prod(1)
        sz = m.size(torch.cat([torch.log(n), 1.0 / n], 1))
        z = torch.exp(sz[:, :1]) * z + sz[:, 1:]
        y = (yte - torch.from_numpy(f.mu)) / torch.from_numpy(f.sd)
        return ((z - y) ** 2)[:, 0].numpy()


ks = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, len(words) + 1]
curve = {k: np.stack([loss_with_top(f, k) for f in fits]) for k in ks}
full, none = curve[ks[-1]].mean(), curve[0].mean()
out = dict(n_words=len(words), loss_no_operators=float(none), loss_all_operators=float(full), sparsity=[],
           ranked=[dict(word=words[i], factor=float(m_word[i]), seed_sd=float(M[:, i].std()), count=int(cnt[words[i]])) for i in rank[:60]])
for k in ks:
    d = paired_bootstrap(curve[0], curve[k], B=500)
    out["sparsity"].append(dict(k=int(min(k, len(words))), loss=float(curve[k].mean()), share_of_gain=float((none - curve[k].mean()) / (none - full)),
                                gain=d.as_dict()))
    print(f"top {k:5d} operators: loss {curve[k].mean():.4f}  share of operator gain {100 * out['sparsity'][-1]['share_of_gain']:.0f}%", flush=True)
print("ranked:", [(r["word"], round(r["factor"], 2)) for r in out["ranked"][:40]], flush=True)
Path("docs/results/e5_operators.json").write_text(json.dumps(out, indent=1))

# ---- B. transfer to NRC two-word expressions (valence) ----
data, df = lexical.bigram_dataset("vad", ("glove",))
vocab = sorted(set(df.w1) | set(df.w2))
sv = pd.DataFrame(dict(phrase=vocab, size=1, label=0.5, split="train"))
V = sst.phrase_sets(sv, dic, FE, max_size=1)
V.X[:, :, -2:] = 0.5
with torch.no_grad():
    mt = np.mean([1.0 + 2.0 * torch.tanh(f.model.delta(torch.from_numpy(V.X)))[:, 0, 0].numpy() for f in fits], 0)
vid = {w: i for i, w in enumerate(vocab)}
fac = np.stack([mt[df.w1.map(vid).to_numpy()], mt[df.w2.map(vid).to_numpy()]], 1).astype(np.float32)  # n, 2
perm = np.random.default_rng(1).permutation(len(vocab))
fac_shuf = np.stack([mt[perm][df.w1.map(vid).to_numpy()], mt[perm][df.w2.map(vid).to_numpy()]], 1).astype(np.float32)
yv = data.y[:, :1]


def with_feat(f: np.ndarray) -> SetData:
    return SetData(np.concatenate([data.X, f[..., None]], -1), data.mask, yv, "reg", ["valence"], data.ids)


plain = SetData(data.X, data.mask, yv, "reg", ["valence"], data.ids)
kw = dict(folds=5, seeds=3, tune=False, verbose=False)
L = dict(add=crossfit(plain, ["add"], **kw)["losses"]["add"][:, :, 0], gate=crossfit(plain, ["gate"], **kw)["losses"]["gate"][:, :, 0])
r = crossfit(with_feat(fac), ["add", "fgate"], **kw)["losses"]
L["add+feat"], L["fgate"] = r["add"][:, :, 0], r["fgate"][:, :, 0]
L["fgate_shuf"] = crossfit(with_feat(fac_shuf), ["fgate"], **kw)["losses"]["fgate"][:, :, 0]
tr_out = dict(n=len(data), share_of_nrc_words_with_factor_below_0p7=float((mt < 0.7).mean()), models={})
for k, v in L.items():
    row = dict(r2=float(1 - v.mean()))
    if k != "add":
        row["vs_add"] = paired_bootstrap(L["add"], v, B=2000).as_dict()
    tr_out["models"][k] = row
    d = row.get("vs_add")
    print(f"  transfer {k:11s} R2 {row['r2']:.4f}" + ("" if not d else f"  d {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"), flush=True)
tr_out["fgate_vs_add+feat"] = paired_bootstrap(L["add+feat"], L["fgate"], B=2000).as_dict()
tr_out["fgate_vs_gate"] = paired_bootstrap(L["gate"], L["fgate"], B=2000).as_dict()
print("  fgate over add+feat:", tr_out["fgate_vs_add+feat"], " fgate over in-domain gate:", tr_out["fgate_vs_gate"], flush=True)
out["transfer"] = tr_out
Path("docs/results/e5_operators.json").write_text(json.dumps(out, indent=1))
