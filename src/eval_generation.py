"""Experiment 1: generative quality, global and fiberwise (protocol of Courts & Kvinge 2022, Sec. 5.1),
evaluated against the chronologically held-out test set.

Global:    4000 test points vs 4000 generated points (base points drawn independently from the test set).
Fiberwise: 15 base points y; 200 generated points over y vs the K=50 test points whose base value is closest to y.
           'score' task: two strata -- y drawn from all test points, and y drawn from high-risk test points (p > 0.5).
           'class' task: the two fibers (legitimate, fraud).
Each regime is repeated 10 times per trained seed; we report mean and bootstrapped 95% CI.
Also reports *fiber consistency*: re-score generated points with the XGBoost model and measure how far
they land from the requested base point (only meaningful because the base space is the model's own score).

Usage: python eval_generation.py score 0 1 2
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get('THREADS', 10)))
from data import prepare, load_scorer, prob_to_base
from metrics import all_metrics, mean_ci, METRICS
from models import NAMES, load, ROOT

task = sys.argv[1]
MERGE = len(sys.argv) > 2 and sys.argv[2] == 'merge'
seeds = [] if MERGE else ([int(s) for s in sys.argv[2:]] or [0])
TAG = os.environ.get('TAG')  # run seeds in parallel processes: TAG=s0 ... then `python eval_generation.py score merge`
names = os.environ.get('MODELS', ','.join(NAMES)).split(',')
ITER, NG, NF, NFS, K = int(os.environ.get('ITER', 10)), 4000, 15, 200, 50

d = prepare()
scorer = load_scorer()
X_te, b_te, y_te, p_te = d['X_te'], d['b_te'], d['y_te'], d['p_te']
# The dataset contains exact duplicate transactions; a reference fiber made of copies of one row has no non-zero
# nearest-neighbour distance and breaks the k-NN KL estimator. Evaluate against unique test transactions only.
_u = np.sort(np.unique(X_te, axis=0, return_index=True)[1])
print(f'dropping {len(X_te) - len(_u)} duplicate test rows')
X_te, b_te, y_te, p_te = X_te[_u], b_te[_u], y_te[_u], p_te[_u]
base_te = b_te if task == 'score' else y_te.astype(np.float32)
high = np.where(p_te > 0.5)[0]
print(f'test points: {len(X_te)}, high-risk (p>0.5): {len(high)}, frauds: {y_te.sum()}')


def rescore(X):
    return prob_to_base(scorer.predict_proba(X)[:, 1])


def fiber_reference(y):
    if task == 'score':
        return np.argsort(np.abs(b_te - y))[:K]
    return np.where(y_te == int(y))[0]


rows = []
for seed in ([] if MERGE else seeds):
    for name in names:
        if name == 'WGAN' and task == 'class' and seed > 0:
            continue
        try:
            m = load(name, task, d, seed)
        except FileNotFoundError:
            print(f'missing {task} {name} seed {seed}'); continue
        rng = np.random.default_rng(1000 + seed)
        for it in range(ITER):
            # ---- global
            i, j = rng.choice(len(X_te), NG, replace=False), rng.choice(len(X_te), NG, replace=False)
            gen = m.sample_from_fiber(base_te[j], rng=rng)
            for k, v in all_metrics(X_te[i], gen).items():
                rows.append((name, seed, it, 'global', k, v))
            if name == 'WGAN':
                continue
            # ---- fiberwise
            strata = {'fiber-all': rng.choice(len(X_te), NF), 'fiber-high': rng.choice(high, NF)} if task == 'score' \
                else {'fiber-legit': [np.where(y_te == 0)[0][0]], 'fiber-fraud': [np.where(y_te == 1)[0][0]]}
            for regime, picks in strata.items():
                acc = {k: [] for k in METRICS + ['consistency', 'ref-width']}
                for p in picks:
                    y = base_te[p]
                    ref = fiber_reference(y)
                    if task == 'class' and len(ref) > NFS:
                        ref = rng.choice(ref, NFS, replace=False)
                    g = m.sample_from_fiber(np.array([y]), n_per_point=NFS, rng=rng)
                    for k, v in all_metrics(X_te[ref], g).items():
                        acc[k].append(v)
                    if task == 'score':
                        acc['consistency'].append(np.abs(rescore(g) - y).mean())
                        acc['ref-width'].append(np.abs(b_te[ref] - y).mean())
                    else:  # share of generated points the classifier puts on the requested side
                        acc['consistency'].append(((scorer.predict_proba(g)[:, 1] > 0.5) == bool(y)).mean())
                        acc['ref-width'].append(((p_te[ref] > 0.5) == bool(y)).mean())
                for k, v in acc.items():
                    rows.append((name, seed, it, regime, k, float(np.mean(v))))
        print(f'{task} {name} seed {seed} done', flush=True)

df = pd.DataFrame(rows, columns=['model', 'seed', 'iter', 'regime', 'metric', 'value'])
if TAG:
    df.to_csv(os.path.join(ROOT, 'results', f'generation_{task}_part_{TAG}.csv'), index=False)
    sys.exit(0)
if MERGE:
    import glob
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, 'results', f'generation_{task}_part_*.csv')))])
df.to_csv(os.path.join(ROOT, 'results', f'generation_{task}.csv'), index=False)

summary = []
for (model, regime, metric), g in df.groupby(['model', 'regime', 'metric']):
    mu, ci = mean_ci(g['value'].values)
    summary.append((regime, metric, model, mu, ci))
s = pd.DataFrame(summary, columns=['regime', 'metric', 'model', 'mean', 'ci95'])
s.to_csv(os.path.join(ROOT, 'results', f'generation_{task}_summary.csv'), index=False)
pd.set_option('display.width', 250)
for regime in s.regime.unique():
    t = s[s.regime == regime]
    print(f'\n### {task} / {regime}')
    print(t.assign(v=t['mean'].map('{:.4f}'.format) + ' ± ' + t['ci95'].map('{:.4f}'.format))
           .pivot(index='metric', columns='model', values='v'))
