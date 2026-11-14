# Relational Concept Flow — research spike

Does the meaning or feeling of a configuration emerge from combinations of objects, beyond the sum of the parts?
This repository tests that with a body-order (cluster) expansion over object sets and a linear
real-space-mutual-information coarse-graining.

- `docs/BRIEF.md` — the founder brief
- `docs/PRIOR_ART.md` — verified prior art and data licences
- `docs/PROPOSAL.md` — experiment design and pass rule
- `docs/RESULTS.md` — **results of the first pass**
- `docs/RESULTS_2.md` — **results of the second pass: operators and per-user couplings**

## Layout

```
src/rcf/        models.py   set models with controlled body order (additive, ACE-style, pair, gated, Deep Sets, Set Transformer)
                train.py compare.py stats.py   training, cross-fitting, paired bootstrap, pass rule
                synthetic.py   planted-interaction generator
                lexical.py sst.py emoset.py tags.py goemotions.py   test beds
                embed.py   encoder registry with a disk cache
                rg.py      CCA-based relevance spectrum
scripts/        e0_synthetic.py  e1_lexical.py  e1_norms_gbm.py  e1c_sst.py  e1c_operators.py  e1_where.py
                extract_emoset.py  e2_emoset.py  e3_rg.py  make_tables.py  make_figures.py
                e4_goemotions.py  e4b_sgate.py  e4_ops.py  e4_personal.py  e4c_onboarding.py
                e4d_userspace.py  e4e_joint.py   emotions, per-rater couplings, cold start, joint training
                e6_mhs.py   per-user fields on a second dataset (7,912 annotators)
                e5_operators.py  e5b_short_list.py   operator sparsity and transfer
tests/          unit tests, including recovery of a planted interaction
docs/results/   JSON outputs and figures
```

## Reproduce

```bash
uv venv .venv --python 3.13 && . .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e . pypdf
python -m pytest -q
python scripts/e0_synthetic.py                      # calibration, no data needed
```

The language and image runs need data under `data/raw/` (not committed):

| data | source | notes |
|---|---|---|
| NRC VAD Lexicon v2.1 | saifmohammad.com/WebPages/nrc-vad.html | non-commercial research only; unzip to `data/raw/nrc_vad/` |
| concreteness norms | Hugging Face `StephanAkkerman/concreteness-ratings`, `martingrzzler/concreteness_phrase_ratings` | to `data/raw/concreteness/` |
| Stanford Sentiment Treebank | Hugging Face `stanfordnlp/sst`, branch `refs/convert/parquet` | to `data/raw/sst/` |
| EmoSet-118K | Hugging Face `Woleek/EmoSet-118K` (third-party mirror) | streamed by `scripts/extract_emoset.py` |
| GoEmotions (raw, with rater ids) | Hugging Face `google-research-datasets/go_emotions`, `raw/` | Apache-2.0; to `data/raw/goemotions/raw.parquet` |
| Measuring Hate Speech | Hugging Face `ucberkeley-dlab/measuring-hate-speech` | CC-BY-4.0; to `data/raw/mhs/mhs.parquet`; only annotator ids are used |

Everything ran on 4 CPU cores. Hosts that must be reachable are listed in `docs/RESULTS.md` section 2 and in the
cloud environment `relational-concept-flow`.
