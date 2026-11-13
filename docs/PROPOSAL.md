# Spike proposal — Relational Concept Flow

Status: proposal, awaiting founder go-ahead. No code, downloads or jobs yet.

## 1. Corrections to the brief that shape the experiment

**1.1 "Order 1 = Deep Sets" is loose.** Deep Sets is $\rho\big(\sum_i\phi(x_i)\big)$. With a nonlinear $\rho$ it is universal once the latent dimension is large enough, so it already contains every body order. The strictly additive model is the special case of linear $\rho$. ACE is the case of polynomial $\rho$. This gives a clean ladder on one shared pooled basis $A=\frac1N\sum_i\phi_\theta(x_i)\in\mathbb R^K$:

$$z_c^{(\nu)} = w_c\cdot A \;+\; A^\top W_c A \;+\;\cdots\;(\text{degree }\nu)$$

Q1 is then the comparison $\nu=1$ against $\nu=2,3$. Deep Sets with an MLP $\rho$ and a Set Transformer are unrestricted-order references, not the order-1 baseline.

**1.2 The body-order decomposition is not unique.** ACE products contain self-interaction terms, so the degree-2 term carries one-body pieces. For "where do interactions concentrate" I will use the canonical version, Möbius inversion over subsets of the fitted model:

$$\phi^{(2)}_c(i,j)=a_c(\{i,j\})-a_c(\{i\})-a_c(\{j\})+a_c(\emptyset)$$

**1.3 Interaction depends on the link function and on extensivity.** $\sigma(\sum_i\phi_i)$ is additive in logits and all-order in probabilities. I define additivity in logit space. Affect is plausibly intensive (ten objects are not ten times as sad), so pooling is a mean plus an explicit dependence on $N$; otherwise the normalisation alone registers as a many-body effect.

**1.4 Q2 cannot measure scaling exponents.** A scene offers two or three levels (objects, groups, scene), which is no scaling regime, so $y_c$ in $\delta g_c\sim b^{y_c}$ is not estimable. Operational substitute: the information-bottleneck spectrum between a block of objects and its environment. In the linear-Gaussian case this reduces to canonical correlations (my memory of Chechik et al. 2005; unverified). The Q2 criterion becomes "a gap in that spectrum, stable under resampling". Gordon et al. (2021) link IB-relevant variables to low scaling dimension, but only for field theories; for scenes it stays an analogy.

**1.5 Headroom caveat.** If the additive model on object features already matches a whole-image model, interactions have nothing left to explain and Q1 returns a null by construction. Every result will report the headroom between $\nu=1$ and the annotator noise ceiling.

**1.6 Dataset choice.** ArtEmis paintings have no object annotations and detectors transfer poorly to paintings (my expectation, untested). The Affection dataset has human affect labels on images that already carry human object annotations in Visual Genome and COCO. It is the better fit for Q1.

## 2. Minimal first experiment

**E0 — synthetic calibration (no data, minutes on CPU).** Generate object sets with a planted pair term of strength $\lambda$ and label noise matched to a handful of annotators per image. Outputs: false-positive rate at $\lambda=0$, and the smallest $\lambda$ detectable at realistic sample size. Without this, a null on real data is uninterpretable.

**E1 — Affection ∩ Visual Genome, annotation-only features (CPU).**

- Unit: one image. Split by image; all tuning on validation only.
- Objects: Visual Genome instances. $x_i$ = text embedding of "attributes + name" plus normalised box geometry. No image pixels needed in this phase.
- Target: the per-image distribution of annotator emotion choices. Secondary: a scalar valence (positive minus negative share).
- Models: $\nu=1,2,3$ on the shared basis, capacity-matched; references: bag-of-names floor, Deep Sets with MLP $\rho$, small Set Transformer.
- Metric: held-out cross-entropy against the annotator distribution; noise ceiling from split-half resampling of annotators.
- Statistic: $\Delta_\nu=\mathcal L_1-\mathcal L_\nu$, paired per image, 95% CI from at least 5 seeds crossed with an image bootstrap.
- Proposed pass rule, fixed before touching the test split: lower CI bound of $\Delta_2$ above 0 **and** $\Delta_2\ge 10\%$ of the headroom $\mathcal L_1-\mathcal L_{\text{ceiling}}$. The second clause is needed because at tens of thousands of images a negligible effect is still statistically significant.
- Localisation: Möbius pair terms aggregated by object-category pair, by emotion and by spatial distance.

Known confounds: object text embeddings carry language-model priors (features only; labels stay human). The size of the Affection ∩ Visual Genome overlap and the exact label set are unknown until the data is in hand.

**Later phases, gated on E1:** visual crop features and a whole-image ceiling (needs images and preferably a GPU); the Q2 spectrum; per-annotator couplings if Affection releases annotator IDs (unknown).

## 3. Repo structure

```
docs/             BRIEF.md, PRIOR_ART.md, PROPOSAL.md, results/ (dated write-ups)
pyproject.toml    pinned with uv.lock
configs/          one YAML per experiment: data, model, seeds
src/rcf/
  data/           synthetic.py, affection.py, visual_genome.py -> one ObjectSet schema
  embed/          encoder registry with on-disk cache            (reusable by the venture engine)
  models/         pooled_basis.py, body_order.py, deepsets.py, set_transformer.py
  couplings/      per-user head / field term; stub until Q1 passes (reusable)
  analysis/       mobius.py, stats.py (paired bootstrap, noise ceiling), spectrum.py (Q2)
  train.py  eval.py
scripts/          run_experiment.py, make_report.py
tests/            planted interaction is recovered; null data gives a null
runs/             gitignored; config + seed + metrics JSON per run
```

## 4. Task breakdown and timebox

Calendar time is set by data access and founder review, not by coding.

| Milestone | Content | Gate |
|---|---|---|
| M0, days 0–2 after go-ahead | Scaffold, pinned environment, tests, E0 power curve | Estimator calibrated |
| M1, week 1 | Data pipeline; EDA: overlap size, set-size distribution, noise ceiling | Enough overlap and headroom to proceed |
| M2, week 2 | E1 with CIs | Preliminary Q1 verdict |
| M3, week 3 | Robustness: visual features, whole-image ceiling, localisation | Q1 verdict |
| M4, week 4 | Q1 passes: Q2 spectrum. Q1 fails: negative write-up | — |
| Hard stop, week 5 | Final write-up either way | — |

## 5. Needed from the founder

**Decisions**

1. Repo: this session is attached to `Personal_Website`, which is empty. Keep the spike here or move to a dedicated repo?
2. Confirm or amend the pass rule in section 2 and the 5-week hard stop.
3. Primary target: native categorical emotion distribution (recommended), or VAD derived by running a lexicon over explanations (a proxy of a proxy).
4. Accept language-derived object features for phase 1?
5. Accept the operational redefinition of Q2 in 1.4?
6. Licence stance: NRC VAD is non-commercial; Affection and ArtEmis terms are probably similar. Fine for the spike, but trained artefacts could not go into a product.

**Data access**

7. Affection is form-gated and the terms must be accepted by you, not by me. Please request it and tell me where the files will live.
8. Network: the environment allows package registries only. Hugging Face, the PyTorch wheel index, Visual Genome, COCO, NRC and arXiv are all refused by the proxy. On go-ahead I will propose a dedicated environment with the exact hosts for your approval.

**Compute**

9. This container: 4 vCPU, 15 GB RAM, no GPU, about 30 GB free disk, ephemeral. Enough for E0 and E1. Is a GPU available for M3, and with what budget?
10. Persistent storage for datasets and cached features, since the container is reclaimed when idle.
