#!/usr/bin/env bash
# Image body-order runs on the splits given as $1 (default: test), cross-fitted.
export OMP_NUM_THREADS=1
S=${1:-test}
py=scripts/e2_emoset.py
python $py --rep tiles2 --splits $S --pca 0
python $py --rep tags --splits $S --refs ""
python $py --rep tags_norms --splits $S --refs ""
python $py --rep tiles3 --splits $S --pca 0 --refs ""
python $py --rep attrs --splits $S --refs "" --kinds add,poly2,poly3,pair,deepsets
python $py --rep tiles4 --splits $S --pca 128 --refs "" --kinds add,poly2,poly3,deepsets,settrans
