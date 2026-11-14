"""E5b: how short can the operator list be, when the model is refitted with that list only?

Rank words by |m - 1| in the unrestricted gated model (words seen >= 25 times in training), then
refit a gated model in which only the top-k words may carry a factor.  Reports the held-out gain
over the additive model for each k, next to the unrestricted gate.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcf import sst
from rcf.data import SetData
from rcf.stats import paired_bootstrap
from rcf.train import fit, test_loss

torch.set_num_threads(1)
FE = ("minilm", "wordsent")
nodes, dic = sst.load()
tab = sst.phrase_table(nodes)
rng = np.random.default_rng(0)
idx = {s: np.where(tab.split == s)[0] for s in ("train", "validation", "test")}
idx["train"] = rng.permutation(idx["train"])[:60000]
idx["test"] = rng.permutation(idx["test"])[:20000]
idx["validation"] = rng.permutation(idx["validation"])[:8000]
sets = {s: sst.phrase_sets(tab.iloc[idx[s]].reset_index(drop=True), dic, FE) for s in idx}
toks = {s: [p.split(" ") for p in tab.phrase.iloc[idx[s]]] for s in idx}
cnt = Counter(w for t in toks["train"] for w in t)
words = sorted(w for w, c in cnt.items() if c >= 25)
W = sst.phrase_sets(pd.DataFrame(dict(phrase=words, size=1, label=0.5, split="train")), dic, FE, max_size=1)
W.X[:, :, -2:] = 0.5
kw = dict(batch=512, max_epochs=40, patience=5)
S = 3
full = [fit(sets["train"], sets["validation"], "gate", seed=s, **kw) for s in range(S)]
L = dict(add=np.stack([test_loss(fit(sets["train"], sets["validation"], "add", seed=s, **kw), sets["test"])[:, 0] for s in range(S)]),
         gate=np.stack([test_loss(f, sets["test"])[:, 0] for f in full]))
with torch.no_grad():
    m = np.mean([1.0 + 2.0 * torch.tanh(f.model.delta(torch.from_numpy(W.X)))[:, 0, 0].numpy() for f in full], 0)
rank = [words[i] for i in np.argsort(-np.abs(m - 1.0))]
fac = dict(zip(words, m))


def flagged(s: str, allowed: set[str]) -> SetData:
    d = sets[s]
    flag = np.zeros(d.mask.shape + (1,), dtype=np.float32)
    for i, t in enumerate(toks[s]):
        for j, w in enumerate(t[: d.mask.shape[1]]):
            flag[i, j, 0] = w in allowed
    # the position features stay at -3, -2 ... keep them where the scoped models expect them is not needed here
    return SetData(np.concatenate([d.X, flag], -1), d.mask, d.y, d.task, d.names)


out = dict(n_candidate_words=len(words), ranking=[dict(word=w, factor=float(fac[w]), count=int(cnt[w])) for w in rank[:60]], lists=[])
g_full = paired_bootstrap(L["add"], L["gate"], B=1000)
print(f"additive R2 {1 - L['add'].mean():.4f}; unrestricted gate gain {g_full.mean:+.4f} [{g_full.lo:+.4f},{g_full.hi:+.4f}]", flush=True)
for k in (5, 10, 20, 50, 200):
    allowed = set(rank[:k])
    tr, va, te = (flagged(s, allowed) for s in ("train", "validation", "test"))
    Lk = np.stack([test_loss(fit(tr, va, "mgate", seed=s, **kw), te)[:, 0] for s in range(S)])
    d = paired_bootstrap(L["add"], Lk, B=1000)
    share_tok = float(np.mean([w in allowed for t in toks["test"] for w in t]))
    out["lists"].append(dict(k=k, words=rank[:k] if k <= 20 else None, gain=d.as_dict(), share_of_full_gate_gain=float(d.mean / g_full.mean),
                             share_of_test_tokens=share_tok))
    print(f"top {k:4d} words as operators: gain {d.mean:+.4f} [{d.lo:+.4f},{d.hi:+.4f}] = {100 * d.mean / g_full.mean:.0f}% of the full gate; "
          f"{100 * share_tok:.1f}% of tokens", flush=True)
out["additive_r2"], out["full_gate_gain"] = float(1 - L["add"].mean()), g_full.as_dict()
print("ranking:", [(w, round(float(fac[w]), 2)) for w in rank[:30]], flush=True)
Path("docs/results/e5b_short_list.json").write_text(json.dumps(out, indent=1))
