"""Experiment 2: does synthetic fraud help a downstream detector?

XGBoost is trained on the chronological training set plus N_AUG synthetic fraud samples from each generator,
and evaluated on the (later) test set. No model ever sees test labels. Repeated over XGBoost seeds and
generator seeds.

Usage: python eval_augmentation.py
"""
import os
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get('THREADS', 10)))
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from data import prepare, fit_scorer
from metrics import mean_ci
from models import load, path, ROOT

N_AUG = int(os.environ.get('N_AUG', 2000))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
d = prepare()
X_tr, y_tr, X_te, y_te, b_tr = d['X_tr'], d['y_tr'], d['X_te'], d['y_te'], d['b_tr']
fraud_b_high = b_tr[(y_tr == 1) & (d['p_tr'] > 0.5)]


def recall_at_fpr(y, p, fpr_max=0.001):
    fpr, tpr, _ = roc_curve(y, p)
    return tpr[fpr <= fpr_max].max()


def evaluate(Xa, ya, seed):
    clf = fit_scorer(Xa, ya, seed=seed)
    p = clf.predict_proba(X_te)[:, 1]
    PREDS.setdefault(CURRENT[0], []).append(p)
    return {'AUPRC': average_precision_score(y_te, p), 'AUROC': roc_auc_score(y_te, p),
            'F1@0.5': f1_score(y_te, p > 0.5), 'Recall@FPR=0.1%': recall_at_fpr(y_te, p)}


def synth(kind, seed, rng):
    """Return N_AUG synthetic fraud feature vectors (or None for the no-augmentation baseline)."""
    if kind == 'None':
        return None
    if kind == 'SMOTE':
        n_f = (y_tr == 1).sum()
        Xs, ys = SMOTE(sampling_strategy={1: n_f + N_AUG}, random_state=seed).fit_resample(X_tr, y_tr)
        return Xs[len(X_tr):]
    model, task = kind.split(' | ')
    if not os.path.exists(path(model, task, seed)) and model not in ('Gaussian-local',):
        return 'missing'
    m = load(model, task, d, seed)
    if task == 'class':
        return m.sample_from_fiber(np.array([1.0]), n_per_point=N_AUG, rng=rng)
    return m.sample_from_fiber(rng.choice(fraud_b_high, N_AUG), rng=rng)  # high-risk fibers of the score model


KINDS = ['None', 'SMOTE',
         'BundleNet | class', 'CGAN | class', 'CGAN-local | class', 'Gaussian-local | class',
         'BundleNet | score', 'CGAN-local | score', 'Gaussian-local | score']

PREDS, CURRENT = {}, ['']
rows = []
for kind in KINDS:
    CURRENT[0] = kind
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        S = synth(kind, seed, rng)
        if isinstance(S, str):
            print('missing', kind, seed); continue
        if S is None:
            Xa, ya = X_tr, y_tr
        else:
            Xa, ya = np.vstack([X_tr, S]), np.concatenate([y_tr, np.ones(len(S), dtype=y_tr.dtype)])
        r = evaluate(Xa, ya, seed)
        rows += [(kind, seed, k, v) for k, v in r.items()]
        print(kind, seed, {k: round(v, 4) for k, v in r.items()}, flush=True)

    # Common practice for comparison: SMOTE to a 1:1 class balance
    if kind == 'SMOTE':
        CURRENT[0] = 'SMOTE (1:1)'
        for seed in SEEDS:
            Xs, ys = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
            r = evaluate(Xs, ys, seed)
            rows += [('SMOTE (1:1)', seed, k, v) for k, v in r.items()]

df = pd.DataFrame(rows, columns=['augmentation', 'seed', 'metric', 'value'])
df.to_csv(os.path.join(ROOT, 'results', 'augmentation.csv'), index=False)
summ = [(a, m, *mean_ci(g['value'].values)) for (a, m), g in df.groupby(['augmentation', 'metric'], sort=False)]
s = pd.DataFrame(summ, columns=['augmentation', 'metric', 'mean', 'ci95'])
s.to_csv(os.path.join(ROOT, 'results', 'augmentation_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(s.assign(v=s['mean'].map('{:.4f}'.format) + ' ± ' + s['ci95'].map('{:.4f}'.format))
       .pivot(index='augmentation', columns='metric', values='v'))


# Paired bootstrap over test transactions: AUPRC(augmented) - AUPRC(no augmentation), seed-averaged scores
np.savez_compressed(os.path.join(ROOT, 'results', 'augmentation_predictions.npz'),
                    **{k.replace(' | ', '__'): np.array(v) for k, v in PREDS.items()}, y_te=y_te)
rng = np.random.default_rng(0)
B = 2000
idx = [rng.choice(len(y_te), len(y_te)) for _ in range(B)]
idx = [i for i in idx if y_te[i].sum() > 0]
base = np.mean(PREDS['None'], 0)
sig = []
for k, v in PREDS.items():
    if k == 'None':
        continue
    p = np.mean(v, 0)
    diffs = np.array([average_precision_score(y_te[i], p[i]) - average_precision_score(y_te[i], base[i]) for i in idx])
    sig.append((k, average_precision_score(y_te, p) - average_precision_score(y_te, base),
                np.quantile(diffs, 0.025), np.quantile(diffs, 0.975), (diffs <= 0).mean()))
sig = pd.DataFrame(sig, columns=['augmentation', 'delta_AUPRC', 'ci_low', 'ci_high', 'p_one_sided'])
sig.to_csv(os.path.join(ROOT, 'results', 'augmentation_significance.csv'), index=False)
print(sig.round(4).to_string())
