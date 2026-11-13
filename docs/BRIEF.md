# Research Spike Brief — Relational Concept Flow

This brief hands over context from a long founder–co-founder discussion. It gives the background, the scientific question and the constraints. **It is not a plan.** You (Claude Code) own the engineering breakdown: repo structure, task decomposition, experiment design, timeline.

---

## 0. How to work with the founder

- Founder is a theoretical condensed-matter physicist (RG, many-body formalism, symmetry analysis). Treat physics analogies as precise claims; push back when one is loose.
- Concise, intuition-first writing. No filler.
- Accuracy over confident approximation: explicitly flag estimates, unverified citations and uncertainty. Never present a guess as fact.
- Math: use `$...$` and `$$...$$`. Avoid `\begin{...}` environments (they don't render in the founder's viewer).
- Founder welcomes direct peer-level disagreement with reasoning; hedged diplomacy is less useful.

---

## 1. Background: the venture thread so far

**Original ideas**
1. A physical "shape display" materializing ideas (geometry, physics, architecture, abstract/philosophical concepts) as tactile 3D forms — for blind/deaf learners and as an education/research tool; AI turns text, images or voice into the shape.
2. A vision system indexing where physical objects/documents are → file-tree + Ctrl-F for real space.

**Ruled out (and why)**
- Hardware shape display as a solo venture: decade-long, well-funded graveyard (cost, resolution, refresh); a hardware company in disguise.
- Physical-space Ctrl-F: crowded (trackers, smart-home vision), privacy friction. The movement/trajectory-optimization variant is a separate, also crowded, product.
- "Store mother principles instead of children" for data-centre storage: doesn't work. Lossless compression already exploits redundancy; reconstructing specifics from abstractions is lossy.
- Stock-market / sociology "label a moment with abstract ideas": a different company competing on signal, where the sensory layer adds nothing.

**Current venture direction: personalized sensory embedding engine**
- Embed ideas with off-the-shelf models (no bespoke training data) → structure the space (clustering, UMAP, graphs) → render structure through a perceptual grammar (hue, texture, form…).
- Decided: the grammar is **personalized per user**, not authored by the founder (sensory associations are partly universal, e.g. bouba/kiki, partly cultural, partly personal).
- Goals: an exit within a few years, plus lasting contribution to ontology. The tension between these is acknowledged.
- **Unresolved:** the first user — a non-visual learner grasping a mathematical structure vs. a researcher mapping a field.
- Standing principle: keep one real user in the loop from day one. Don't mature the engine in pure research mode and plan to "narrow later" — it optimizes for interestingness, not utility.

---

## 2. This thread: the founder's new hypothesis (paraphrased)

Motivated by the critique (LeCun and others) that current AI is bottlenecked by being LLM-based:

- Build a node space = **ontology** (sensory qualities: colour, odour, texture, shape…) **+ abstract, human-like nodes** (feelings, thoughts, associations with past experience; physical, mathematical, philosophical, psychological concepts). Use ML to refine and expand the node set, before or after initial definition.
- Learn weights from an encountered object to node activations.
- Meaning emerges from **combinations**: one object, two, three… (formally tensor products of object spaces; restricted to natural/relevant combinations, e.g. those LLMs already surface). Activations depend on the context — proximity/similarity of co-present objects and the environment. The founder calls this an **RG flow**; psychological fallacies and delusions are seen as products of it.
- Founder's conjecture: this is a route to genuine intelligence.

---

## 3. Sharpened formulation (working hypotheses from the discussion)

### 3.1 Adding objects = body-order (cluster) expansion, not RG

For a set $S$ of objects with embeddings $x_i$ in environment $E$, activation of concept node $c$:

$$a_c(S)=\sum_i \phi^{(1)}_c(x_i)+\sum_{i<j}\phi^{(2)}_c(x_i,x_j\,|\,E)+\sum_{i<j<k}\phi^{(3)}_c(x_i,x_j,x_k\,|\,E)+\cdots$$

- The exponential blow-up is avoidable: the atomic cluster expansion (ACE/MACE) writes $n$-body terms as products of one-body projections, so cost is linear in the number of objects at fixed basis size.
- Order 1 alone = Deep Sets (additive baseline). Self-attention / Set Transformer = learned interactions, with depth acting as a flow parameter.
- Hence novelty lies in the **target space** (affective/abstract nodes) and in **measuring interaction structure**, not in the mechanism.

### 3.2 RG proper = coarse-graining in scale

- RG is about scale, not count. Coarse-grain a percept: objects → groups → scene → situation.
- Conjecture: **abstract concepts are the relevant directions of this flow; sensory detail is irrelevant.** Near a fixed point, couplings scale as $\delta g_c \sim b^{\,y_c}$; relevant iff $y_c>0$.
- Concrete tool: RSMI (learn the coarse-graining by maximizing mutual information between a block and its environment) — matches the founder's "feelings only flow within an environment".

### 3.3 Personalization = per-user couplings

Per-user fine-tuned $\phi$ (or a user-specific field term) around a shared model. This is the bridge back to the venture engine.

### 3.4 Explicit non-claims and risks

- **Not** a claim of intelligence or consciousness. This is a representation of what a configuration means/feels like; intelligence would need prediction and planning (a world model) on top. Consciousness is not a success criterion.
- LeCun's own fix points toward *learned* latent world models trained by prediction (JEPA), not hand-defined nodes. Hand-specified concept nodes risk Cyc / semantic-network brittleness → prefer learned/discovered nodes anchored to a small seed of interpretable axes (e.g. valence–arousal–dominance; Gärdenfors quality dimensions).
- No ground truth for "feelings": all labels are proxies. VLM/LLM-generated labels import LLM priors — the very thing the project wants to escape — so track that as a confound.

---

## 4. The falsifiable question

- **Q1 (interaction):** Do body orders $\ge 2$ predict held-out affective/abstract labels better than the order-1 additive model, by more than seed noise? Where do interaction effects concentrate?
- **Q2 (RG):** Does a learned coarse-graining yield a small, stable set of relevant concept directions (a gap in the spectrum)? Are they interpretable?

Suggested kill criterion (founder to confirm): no significant gain from order $\ge 2$ over order 1 on held-out data within the timebox → stop and write up the negative result. Suggested timebox: 4–6 weeks (founder to confirm).

---

## 5. Prior art (listed from memory, unverified — verify each before relying on it)

- LeCun (2022), *A Path Towards Autonomous Machine Intelligence*; I-JEPA, V-JEPA, V-JEPA 2.
- Mehta & Schwab (2014), mapping between variational RG and deep learning.
- Koch-Janusz & Ringel (2018, Nature Physics), RSMI.
- Drautz (2019, PRB), Atomic Cluster Expansion; Batatia et al. (2022), MACE.
- Zaheer et al. (2017), Deep Sets; Lee et al. (2019), Set Transformer.
- Geshkovski, Letrouit, Polyanskiy, Rigollet (2023), transformers as interacting-particle systems.
- Gärdenfors (2000), *Conceptual Spaces*.
- Barsalou, grounded cognition.
- Russell (1980), circumplex model of affect.
- Cyc, ConceptNet — cautionary lineage for hand-built ontologies.

## 6. Candidate data (verify access and licences)

- ArtEmis — affective responses + explanations for artworks.
- EmoSet — emotion-labelled images with attributes.
- NRC VAD Lexicon — valence/arousal/dominance norms for words.
- Visual Genome — objects, attributes, relationships (scene graphs).
- COCO + an object detector for object sets.
- VLM-generated labels — cheap, biased; secondary only.

---

## 7. Engineering expectations

- Start small (CPU / single-GPU). Ask the founder about available compute. Ask before large downloads or long jobs.
- Reproducible: pinned environment, seeds, config files, experiment logging.
- Report every result with uncertainty (multiple seeds, confidence intervals).
- Keep components that would also serve the venture engine (embedding pipeline, per-user couplings) modular.

## 8. First deliverables (yours to define and refine)

1. A short verified prior-art note (flag anything unconfirmed).
2. Proposed repo structure, minimal first experiment, task breakdown and timebox.
3. A list of decisions/inputs needed from the founder.
