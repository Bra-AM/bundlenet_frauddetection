#!/usr/bin/env bash
# Reproduces every number, table and figure of the article, in order.
#   pip install -r requirements.txt
#   put creditcard.csv in data/ (see data/README.md)
# Default: the CPU configuration used in the article (about 9 hours on a 10-core laptop CPU).
# Reference-size run on a GPU: export BN_WIDTH=512 BN_DEPTH=5 BN_EPOCHS=1000 BN_LR=2e-4
set -e
cd "$(dirname "$0")/src"

python data.py                            # chronological split, out-of-fold XGBoost score (checks the data checksum)

python train_all.py score 0 1 2           # BundleNet, CGAN, CGAN-local, WGAN      (model score as base space)
MODELS=BundleNet,CGAN,CGAN-local python train_all.py class 0 1 2   # (class label as base space)

python eval_generation.py score 0 1 2     # Tables 1-2
python eval_generation.py class 0 1 2     # Table 3
python eval_augmentation.py               # Table 4 and the significance table of the appendix
python eval_classification.py             # Table 5
python fiber_analysis.py                  # Figures 4-7: base-point error, fiber profiles, counterfactuals
python explanatory_figures.py             # Figures 1 and 3: fibers on real data, high-risk transactions
python make_tables.py                     # writes results/tables/*.tex
