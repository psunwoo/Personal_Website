# Prior-art verification note

Checked 2026-11-13 by web search. Status reflects what the search results showed, not a full read of each paper.

- **OK** — citation confirmed against the publisher, proceedings or arXiv listing.
- **CAVEAT** — exists, but the brief's citation or the use we make of it needs a correction.
- **UNCONFIRMED** — could not confirm; do not rely on it yet.

## 1. Citations in the brief

| Item | Status | Finding |
|---|---|---|
| LeCun (2022), *A Path Towards Autonomous Machine Intelligence* | OK | Version 0.9.2, dated 27 Jun 2022, posted on OpenReview. A position paper by its own description, not peer-reviewed. I did not open the OpenReview page itself. |
| I-JEPA | OK | Assran et al., CVPR 2023, pp. 15619–15629, arXiv 2301.08243. |
| V-JEPA | CAVEAT | arXiv 2404.08471 (Feb 2024, Meta) confirmed. The title "Revisiting Feature Prediction for Learning Visual Representations from Video" is from my memory; the search did not show it. |
| V-JEPA 2 | OK | arXiv 2506.09985, 11 Jun 2025. |
| Mehta & Schwab (2014) | CAVEAT | arXiv 1410.3831 confirmed. The "exact mapping" is specifically RBM stacks ↔ Kadanoff's variational RG. It was disputed by Lin & Tegmark (arXiv 1608.08225); Schwab & Mehta replied (arXiv 1609.03541) and conceded one statement was not a biconditional. I read only the reply's account of the dispute. Do not cite this as evidence that deep nets in general perform RG. |
| Koch-Janusz & Ringel (2018), RSMI | OK | Nature Physics 14, 578–582, arXiv 1704.06279. Demonstrated on 2D Ising and dimer models. |
| Drautz (2019), ACE | OK | Phys. Rev. B 99, 014104. Confirmed through DOI in reference lists; the paper page itself was not retrieved. |
| Batatia et al. (2022), MACE | OK | NeurIPS 2022, arXiv 2206.07697. (arXiv 2205.06643 is a different paper by overlapping authors.) |
| Zaheer et al. (2017), Deep Sets | OK | arXiv 1703.06114; NIPS 2017 per the arXiv comment field. |
| Lee et al. (2019), Set Transformer | OK | ICML 2019, PMLR 97:3744–3753, arXiv 1810.00825. |
| Geshkovski, Letrouit, Polyanskiy, Rigollet (2023) | CAVEAT | Ambiguous: two 2023 papers. "The emergence of clusters in self-attention dynamics" (NeurIPS 2023, arXiv 2305.05465) and "A mathematical perspective on Transformers" (arXiv 2312.10794). Both are theory for idealised attention dynamics with fixed weights. |
| Gärdenfors (2000), *Conceptual Spaces* | OK | MIT Press (Bradford), ISBN 0262071991. |
| Barsalou, grounded cognition | OK | "Grounded Cognition", Annu. Rev. Psychol. 59:617–645 (2008). |
| Russell (1980), circumplex | CAVEAT | J. Pers. Soc. Psychol. 39(6):1161–1178 confirmed. The circumplex is two-dimensional (pleasure, arousal). Dominance is the third axis of Mehrabian & Russell's PAD model (1974 book). A "VAD seed" should cite PAD, not the circumplex. |
| ConceptNet | OK | Speer, Chin, Havasi, AAAI 2017, arXiv 1612.03975. |
| Cyc | UNCONFIRMED | No specific citation confirmed. ConceptNet papers cite Lenat & Guha (1989). "Brittleness" is a common assessment, not a result I verified. |

## 2. Prior art missing from the brief that changes the design

| Item | Status | Why it matters |
|---|---|---|
| Murphy et al., Janossy pooling (ICLR 2019, arXiv 1811.01900) | OK | The ML name for a $k$-body truncated permutation-invariant function. Section 3.1 of the brief is this construction, so the mechanism is not new in ML either. |
| Wagstaff et al., limits of sum-decomposition | CAVEAT | Result confirmed: Deep Sets is universal only when the latent dimension is large enough. Exact venue and year not confirmed. |
| Lenggenhager et al., PRX 10, 011037 (2020) | OK | An ideal RSMI coarse-graining does not lengthen the range of the renormalized interactions. |
| Gordon, Banerjee, Koch-Janusz, Ringel, PRL 126 (2021), arXiv 2012.01447 | OK | Variables kept by information-bottleneck compression correspond to the operators with lowest scaling dimension. Proven for systems described by a field theory; carrying it over to scenes is an analogy. |
| Achlioptas et al., Affection (CVPR 2023, arXiv 2210.01946) | OK | Human emotional reactions and explanations for 85,007 real-world images drawn from MS-COCO, Visual Genome, Flickr30k Entities and others. These images already carry human object annotations, which ArtEmis paintings do not. |

## 2b. Prior art for the second pass (operators, personal couplings), checked 2026-11-14

| Item | Status | Why it matters |
|---|---|---|
| Socher et al., "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", EMNLP 2013 (aclanthology D13-1170) | OK | Source of the treebank (215,154 phrases, 11,855 sentences). Its model composes children with a tensor, i.e. a second-order term, and its headline claim is handling negation and its scope. That composition matters on this data is therefore known; the body-order framing and the gated additive model are what this spike adds. |
| Zhu, Guo, Mohammad, Kiritchenko, "An Empirical Study on the Effect of Negation Words on Sentiment", ACL 2014, pp. 304–313 (P14-1029) | OK | Tests on the same treebank whether negators reverse the sign of sentiment or shift it by a constant, and finds both heuristics capture little. Our finding that negators act as multiplicative damping toward the baseline ($m \approx 0$, never negative) belongs to this line; that negation is not a sign flip is not new. I did not read their proposed model. |
| Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", ACL 2020, pp. 4040–4054 (arXiv 2005.00547) | OK | 58k Reddit comments, 27 emotions plus neutral. Rater details were not in the search results; the counts in RESULTS_2.md come from the released raw file itself. |
| Davani, Díaz, Prabhakaran, "Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations", TACL 10:92–110 (2022) | OK | Trains one prediction head per annotator on a shared representation and finds it matches or beats majority-vote training. Per-annotator modelling on a shared model is therefore established; our per-rater field is a minimal instance of it, and the cold-start measurement is the part aimed at the venture. |

Data added in the second pass: GoEmotions raw annotations (Apache-2.0 per its Hugging Face card) and Measuring Hate Speech (CC-BY-4.0 per its Hugging Face card; the card points to arXiv 2009.10277, which I have not read).

## 3. Candidate data

| Dataset | Status | Size and content | Access and licence |
|---|---|---|---|
| ArtEmis | OK | 454,684 annotations on about 80K WikiArt artworks (CVPR 2021). | Form-gated. Dataset licence not confirmed (one aggregator says CC BY-NC-SA 4.0). No object annotations. |
| Affection | OK | 85,007 images, 6,283 annotators, 526,749 responses. | Form-gated under "specific terms of use"; I could not read the terms. |
| EmoSet | OK | 3.3M images, 118,102 human-labelled; 8 emotion categories; six attributes including object class (ICCV 2023). | GitHub repo lists two releases; download links and licence not confirmed. |
| NRC VAD Lexicon | CAVEAT | v1: about 20K words (my memory). v2 (2025): over 55K terms including about 10K multiword expressions. | Free for non-commercial research and education only; no redistribution; commercial use needs a licence from NRC. |
| Visual Genome | OK | 108,077 images, 3.8M objects, 2.8M attributes, 2.3M relationships. | CC BY 4.0. Images about 23 GB in two zips; annotation JSON is tens to hundreds of MB per file (from a mirror listing). |
| COCO | CAVEAT | — | Annotations CC BY 4.0; images under Flickr terms. Official terms page and annotation zip size not confirmed. |

None of the data hosts is reachable from the current cloud environment (see PROPOSAL.md, section 5).

## 4. Sources

- Socher et al. 2013: https://aclanthology.org/D13-1170.pdf
- Zhu et al. 2014: https://aclanthology.org/P14-1029/
- GoEmotions: https://arxiv.org/pdf/2005.00547
- Davani et al. 2022: https://aclanthology.org/2022.tacl-1.6

- LeCun position paper: https://www.mafr.fr/en/article/a-path-towards-autonomous-machine-intelligence/
- I-JEPA: https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html
- V-JEPA: https://www.arxiv.org/pdf/2404.08471
- V-JEPA 2: https://arxiv.deeppaper.ai/papers/2506.09985v1
- Mehta & Schwab: https://ar5iv.arxiv.org/html/1410.3831 ; reply to criticism: https://www.arxiv.org/abs/1609.03541
- RSMI: https://arxiv.org/abs/1704.06279
- Lenggenhager et al.: https://arxiv.org/pdf/1809.09632.pdf
- Gordon et al.: https://arxiv.org/pdf/2012.01447
- MACE: https://ar5iv.labs.arxiv.org/html/2206.07697 ; https://mlanthology.org/neurips/2022/batatia2022neurips-mace
- Deep Sets: https://arxiv.org/abs/1703.06114v3
- Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Janossy pooling: https://mlanthology.org/iclr/2019/murphy2019iclr-janossy
- Wagstaff et al.: https://embodiedintelligence.web.ox.ac.uk/node/2991081
- Self-attention clustering: https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract.html
- Mathematical perspective on Transformers: https://arxiv.org/abs/2312.10794v4
- Russell (1980): https://pdodds.w3.uvm.edu/research/papers/others/1980/russell1980a.pdf
- Barsalou (2008): https://vsegaux.github.io/vault_M2_CNA_UGA/Sciences-du-langage/Articles/Barsalou_AnnuRevPsycho2008.pdf
- Gärdenfors (2000): https://cse.iitk.ac.in/users/amit/books/gardenfors-2000-conceptual-spaces-geometry.html
- ConceptNet 5.5: https://ojs.aaai.org/index.php/AAAI/article/view/11164
- ArtEmis: https://github.com/optas/artemis ; https://openaccess.thecvf.com/content/CVPR2021/html/Achlioptas_ArtEmis_Affective_Language_for_Visual_Art_CVPR_2021_paper.html
- Affection: https://arxiv.org/abs/2210.01946v1 ; https://openaccess.thecvf.com/content/CVPR2023/html/Achlioptas_Affection_Learning_Affective_Explanations_for_Real-World_Visual_Data_CVPR_2023_paper.html
- EmoSet: https://github.com/JingyuanYY/EmoSet
- NRC VAD: https://saifmohammad.com/WebPages/nrc-vad.html ; https://arxiv.org/abs/2503.23547v1
- Visual Genome: https://homes.cs.washington.edu/~ranjay/visualgenome/ ; https://docs.rcac.purdue.edu/datasets/ai/visualgenome/
- COCO licence split: https://huggingface.co/datasets/shunk031/MSCOCO/raw/main/README.md
