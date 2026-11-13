#!/usr/bin/env bash
# All E1 (language) runs, sequential.  Results: docs/results/e1_*.json
set -u
py=scripts/e1_lexical.py
python $py --target vad  --features minilm --whole minilm,mpnet
python $py --target vad  --features glove
python $py --target vad  --features norms --require-norms
python $py --target vad  --features glove,norms --require-norms
python $py --target conc --features minilm --seeds 3 --batch 512 --whole minilm
python $py --target conc --features glove,norms --require-norms --seeds 3 --batch 512
