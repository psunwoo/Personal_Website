#!/usr/bin/env bash
# Treebank set-model runs; waits for the other language jobs so the box is not oversubscribed.
while pgrep -f "e0_synthetic.py|run_e1_all.sh|run_e1_conc.sh" > /dev/null; do sleep 20; done
export OMP_NUM_THREADS=1
python scripts/e1c_sst.py --part b --features minilm,wordsent
python scripts/e1c_sst.py --part b --features minilm
