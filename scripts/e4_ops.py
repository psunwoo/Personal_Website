"""Which emotion nodes do the strongest operators act on?  (GoEmotions gated model, frozen in stage 1.)

For each word, the mean factor m = 1 + T per emotion node over all its occurrences.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rcf import goemotions as ge

com, rat, emo, Y = ge.load()
T = np.load("runs/e4_frozen.npz")["gate_T"].astype(np.float32)  # comments, L, 28
L = T.shape[1]
acc, cnt = defaultdict(lambda: np.zeros(len(emo))), defaultdict(int)
for i, toks in enumerate(com.tokens):
    for j, w in enumerate(toks[:L]):
        acc[w] += T[i, j]
        cnt[w] += 1
words = [w for w, c in cnt.items() if c >= 50]
M = np.stack([1.0 + acc[w] / cnt[w] for w in words])  # words, 28
strength = np.abs(M - 1).mean(1)
order = np.argsort(-strength)
out = dict(n_words=len(words), share_mean_abs_dev_below_0p05=float((strength < 0.05).mean()), top=[])
for i in order[:25]:
    lo, hi = np.argsort(M[i])[:3], np.argsort(-M[i])[:3]
    out["top"].append(dict(word=words[i], count=int(cnt[words[i]]), mean_abs_dev=float(strength[i]),
                           most_damped={emo[k]: float(M[i, k]) for k in lo}, most_amplified={emo[k]: float(M[i, k]) for k in hi}))
    print(f"{words[i]:12s} n={cnt[words[i]]:5d} |m-1|={strength[i]:.2f}  damps " + ", ".join(f"{emo[k]} {M[i, k]:.2f}" for k in lo)
          + "  | amplifies " + ", ".join(f"{emo[k]} {M[i, k]:.2f}" for k in hi))
Path("docs/results/e4_ops.json").write_text(json.dumps(out, indent=1))
