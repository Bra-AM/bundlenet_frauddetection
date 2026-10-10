"""Classification with a trained BundleNet (Table 5).

Courts & Kvinge leave classification as future work because the neighborhood r_i must be chosen without the
label. With the class label as base space and an invertible Phi_i, both neighborhoods can be scored and a
transaction classified by the likelihood ratio  s(x) = log p(x | fraud) - log p(x | legit) + log prior ratio.
No label is used at test time. Tabular baselines are trained on the same chronological split.
"""
import os
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get('THREADS', 10)))
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from data import prepare, fit_scorer
from metrics import mean_ci
from models import load, path, ROOT

SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
d = prepare()
X_tr, y_tr, X_te, y_te = d['X_tr'], d['y_tr'], d['X_te'], d['y_te']
prior = np.log(y_tr.mean() / (1 - y_tr.mean()))


def scores(y, s, thr):
    return {'AUPRC': average_precision_score(y, s), 'AUROC': roc_auc_score(y, s), 'F1': f1_score(y, s > thr)}


rows = []
for seed in SEEDS:
    if os.path.exists(path('BundleNet', 'class', seed)):
        m = load('BundleNet', 'class', d, seed)
        # nbhd index of class c: centers are sorted unique values (0 -> legit, 1 -> fraud)
        llr = m.log_likelihood(X_te, 1) - m.log_likelihood(X_te, 0) + prior
        llr = np.nan_to_num(llr, nan=-1e9, posinf=1e9, neginf=-1e9)
        rows += [('BundleNet (likelihood ratio)', seed, k, v) for k, v in scores(y_te, llr, 0.0).items()]

    for name, clf in [('XGBoost', lambda s: fit_scorer(X_tr, y_tr, seed=s)),
                      ('LightGBM', lambda s: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1, random_state=s, verbose=-1).fit(X_tr, y_tr)),
                      ('Random Forest', lambda s: RandomForestClassifier(300, n_jobs=-1, random_state=s).fit(X_tr, y_tr)),
                      ('MLP', lambda s: MLPClassifier((128, 64), random_state=s, max_iter=200).fit(X_tr, y_tr)),
                      ('QDA (Gaussian likelihood ratio)', lambda s: QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(X_tr, y_tr))]:
        p = clf(seed).predict_proba(X_te)[:, 1]
        rows += [(name, seed, k, v) for k, v in scores(y_te, p, 0.5).items()]
    print('seed', seed, 'done', flush=True)

df = pd.DataFrame(rows, columns=['model', 'seed', 'metric', 'value'])
df.to_csv(os.path.join(ROOT, 'results', 'classification.csv'), index=False)
s = pd.DataFrame([(a, m, *mean_ci(g['value'].values)) for (a, m), g in df.groupby(['model', 'metric'], sort=False)],
                 columns=['model', 'metric', 'mean', 'ci95'])
s.to_csv(os.path.join(ROOT, 'results', 'classification_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(s.assign(v=s['mean'].map('{:.4f}'.format) + ' ± ' + s['ci95'].map('{:.4f}'.format))
       .pivot(index='model', columns='metric', values='v'))
