#!/usr/bin/env bash
# Operator/particle follow-up on the treebank; waits for a free core.
while pgrep -f "e2_emoset.py --rep tiles3" > /dev/null; do sleep 20; done
export OMP_NUM_THREADS=1
python scripts/e1c_sst.py --part b --features minilm,wordsent --kinds add,gate,poly2 --tag gate
