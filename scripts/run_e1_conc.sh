#!/usr/bin/env bash
# Concreteness run, trimmed for CPU: 3 folds, 3 seeds, no explicit-pair model.
export OMP_NUM_THREADS=1
python scripts/e1_lexical.py --target conc --features minilm --folds 3 --seeds 3 --batch 512 --kinds add,poly2,deepsets --whole minilm
