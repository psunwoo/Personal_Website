"""E4b: a scalar-factor gate on GoEmotions, for a clean reading of the operators.

With one factor per emotion node (the "gate" of stage 1) the factors are entangled with the sign
of each node's baseline logit and do not read as negation.  Here each token has a single factor m
acting on the deviation of all 28 logits from their baseline.  Reports the gain over the additive
model (paired with stage 1), the word factors, their agreement with the treebank's factors, and
freezes the model for the per-rater stage.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcf import goemotions as ge
from rcf import sst
from rcf.stats import paired_bootstrap
from rcf.train import fit, predict, test_loss

torch.set_num_threads(1)
com, rat, emo, Y = ge.load()
idx = {s: np.where(com.split == s)[0] for s in ("train", "val", "test")}
tr, va, te = (ge.comment_sets(com, rat, Y, emo, rows=idx[s]) for s in ("train", "val", "test"))
kw = dict(batch=512, max_epochs=60, patience=5)
fits = [fit(tr, va, "sgate", seed=s, **kw) for s in range(3)]
Ls = np.stack([test_loss(f, te) for f in fits])
Ladd = np.load("runs/e4_goemotions_losses.npz")["add"]
d = paired_bootstrap(Ladd, Ls, B=2000)
dg = paired_bootstrap(np.load("runs/e4_goemotions_losses.npz")["gate"], Ls, B=2000)
print(f"sgate loss {Ls.mean():.4f}; gain over additive {d.mean:+.4f} [{d.lo:+.4f},{d.hi:+.4f}]; over per-node gate {dg.mean:+.4f} [{dg.lo:+.4f},{dg.hi:+.4f}]",
      "epochs", [f.epochs for f in fits], flush=True)

# word factors: mean over occurrences and seeds
acc, cnt = defaultdict(float), Counter()
frozen = {}
n_all = len(com)
for si, f in enumerate(fits):
    m = f.model
    Tfull = np.zeros((n_all, tr.X.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for s_, dset in (("train", tr), ("val", va), ("test", te)):
            T = np.concatenate([(m.factors(torch.from_numpy(dset.X[a:a + 4096]), torch.from_numpy(dset.mask[a:a + 4096])) - 1.0)[..., 0].numpy()
                                for a in range(0, len(dset), 4096)])
            Tfull[idx[s_]] = T
    for i, toks in enumerate(com.tokens):
        for j, w in enumerate(toks[: Tfull.shape[1]]):
            acc[w] += Tfull[i, j] / len(fits)
            if si == 0:
                cnt[w] += 1
    if si == 0:  # freeze seed 0 for the per-rater stage
        with torch.no_grad():
            parts = {k: np.zeros((n_all,) + shp, dtype=np.float32) for k, shp in (("logits", (28,)), ("A", (64,)), ("gain", (28,)), ("bias", (28,)))}
            for s_, dset in (("train", tr), ("val", va), ("test", te)):
                x, mk = torch.from_numpy(dset.X), torch.from_numpy(dset.mask)
                parts["logits"][idx[s_]] = predict(f, dset.X, dset.mask)
                mf = mk.float().unsqueeze(-1)
                parts["A"][idx[s_]] = np.concatenate([((m.phi(x[a:a + 4096]) * mf[a:a + 4096]).sum(1) / mf[a:a + 4096].sum(1).clamp(min=1)).numpy()
                                                      for a in range(0, len(dset), 4096)])
                nn_ = mk.float().sum(1, keepdim=True).clamp(min=1)
                sz = m.size(torch.cat([torch.log(nn_), 1.0 / nn_], 1)).numpy()
                parts["gain"][idx[s_]], parts["bias"][idx[s_]] = np.exp(sz[:, :28]), sz[:, 28:]
            frozen = dict(sgate_logits=parts["logits"], sgate_A=parts["A"], sgate_gain=parts["gain"], sgate_bias=parts["bias"], sgate_T=Tfull.astype(np.float16),
                          sgate_w=m.lin.weight.detach().numpy(), sgate_b=m.lin.bias.detach().numpy())
old = dict(np.load("runs/e4_frozen.npz"))
old.update(frozen)
np.savez("runs/e4_frozen.npz", **old)

words = [w for w, c in cnt.items() if c >= 50]
mfac = np.array([1.0 + acc[w] / cnt[w] for w in words])
order = np.argsort(mfac)
out = dict(loss=float(Ls.mean()), gain_over_additive=d.as_dict(), gain_over_per_node_gate=dg.as_dict(), n_words=len(words),
           share_within_0p1_of_one=float((np.abs(mfac - 1) < 0.1).mean()), share_below_half=float((mfac < 0.5).mean()),
           lowest=[dict(word=words[i], factor=float(mfac[i]), count=int(cnt[words[i]])) for i in order[:30]],
           highest=[dict(word=words[i], factor=float(mfac[i]), count=int(cnt[words[i]])) for i in order[::-1][:20]])
print("lowest:", [(words[i], round(float(mfac[i]), 2)) for i in order[:30]], flush=True)
print("highest:", [(words[i], round(float(mfac[i]), 2)) for i in order[::-1][:20]], flush=True)

# agreement with the treebank's factors on shared words
FE = ("minilm", "wordsent")
nodes, dic = sst.load()
tab = sst.phrase_table(nodes)
rng = np.random.default_rng(0)
itr = rng.permutation(np.where(tab.split == "train")[0])[:60000]
iva = rng.permutation(np.where(tab.split == "validation")[0])[:8000]
tcnt = Counter(w.lower() for p in tab.phrase.iloc[itr] for w in p.split(" "))
shared = [w for w in words if tcnt.get(w, 0) >= 25]
ft = fit(sst.phrase_sets(tab.iloc[itr].reset_index(drop=True), dic, FE), sst.phrase_sets(tab.iloc[iva].reset_index(drop=True), dic, FE), "gate",
         seed=0, batch=512, max_epochs=40, patience=5)
Wt = sst.phrase_sets(pd.DataFrame(dict(phrase=shared, size=1, label=0.5, split="train")), dic, FE, max_size=1)
Wt.X[:, :, -2:] = 0.5
with torch.no_grad():
    mt = 1.0 + 2.0 * torch.tanh(ft.model.delta(torch.from_numpy(Wt.X)))[:, 0, 0].numpy()
mg = np.array([mfac[words.index(w)] for w in shared])
from scipy.stats import spearmanr

low_t = {shared[i] for i in np.argsort(mt)[:20]}
low_g = {shared[i] for i in np.argsort(mg)[:20]}
out["treebank_agreement"] = dict(n_shared_words=len(shared), pearson=float(np.corrcoef(mt, mg)[0, 1]), spearman=float(spearmanr(mt, mg)[0]),
                                 overlap_of_20_lowest=sorted(low_t & low_g))
print("agreement with treebank factors:", out["treebank_agreement"], flush=True)
Path("docs/results/e4b_sgate.json").write_text(json.dumps(out, indent=1))
