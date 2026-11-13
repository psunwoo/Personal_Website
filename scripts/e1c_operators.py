"""Which words does the gated-additive model treat as operators?

Fits the gate model on treebank phrases and reads off, for every frequent word, its multiplicative
factor m(word) = 1 + 2 tanh(delta) on the pooled sentiment (m = 1: no effect; m < 0: flips the sign;
0 < m < 1: damps; m > 1: amplifies), next to the word's own human sentiment.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rcf import sst
from rcf.train import fit

torch.set_num_threads(1)
nodes, dic = sst.load()
tab = sst.phrase_table(nodes)
rng = np.random.default_rng(0)
tr = rng.permutation(np.where(tab.split == "train")[0])[:40000]
va = rng.permutation(np.where(tab.split == "validation")[0])[:6000]
data = sst.phrase_sets(tab, dic, ("minilm", "wordsent"))
facs = []
cnt = Counter(w for p in tab.phrase[tr] for w in p.split(" "))
words = [w for w, c in cnt.items() if c >= 25]
probe = sst.phrase_sets(tab.iloc[:1].assign(phrase=" ".join(words[:1])), dic, ("minilm", "wordsent"))  # shape template
single = sst.pd.DataFrame(dict(phrase=words, size=1, label=0.5, split="train"))
S = sst.phrase_sets(single, dic, ("minilm", "wordsent"), max_size=1)
S.X[:, :, -2:] = 0.5  # neutral position
for seed in range(3):
    f = fit(data.subset(tr), data.subset(va), "gate", seed=seed, batch=512, max_epochs=40, patience=5)
    with torch.no_grad():
        m = 1.0 + 2.0 * torch.tanh(f.model.delta(torch.from_numpy(S.X)))[:, 0, 0].numpy()
    facs.append(m)
m = np.mean(facs, 0)
sd = np.std(facs, 0)
own = dic.set_index("phrase")["label"].reindex(words).to_numpy()
rows = [dict(word=w, factor=float(a), seed_sd=float(b), count=int(cnt[w]), own_sentiment=float(o)) for w, a, b, o in zip(words, m, sd, own)]
rows.sort(key=lambda r: r["factor"])
out = dict(n_words=len(rows), share_within_0p1_of_one=float(np.mean(np.abs(m - 1) < 0.1)),
           share_below_half=float(np.mean(m < 0.5)), share_negative=float(np.mean(m < 0)), share_above_1p5=float(np.mean(m > 1.5)),
           lowest=rows[:30], highest=rows[-20:][::-1])
Path("docs/results/e1c_operators.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if not isinstance(v, list)})
print("lowest:", [(r["word"], round(r["factor"], 2), round(r["seed_sd"], 2)) for r in out["lowest"]])
print("highest:", [(r["word"], round(r["factor"], 2), round(r["seed_sd"], 2)) for r in out["highest"]])
