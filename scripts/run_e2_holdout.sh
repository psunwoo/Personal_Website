#!/usr/bin/env bash
# Confirmatory image runs on the official split: train = all extracted train shards, test = full test split.
export OMP_NUM_THREADS=1
py=scripts/e2_emoset.py
K=add,poly2,gate,deepsets
python $py --rep tiles2 --holdout --pca 0 --kinds $K --refs whole_d1 &
python $py --rep tiles3 --holdout --pca 0 --kinds $K --refs "" &
python $py --rep tiles4 --holdout --pca 128 --kinds $K --refs "" &
python $py --rep tags --holdout --kinds $K --refs "" &
wait
