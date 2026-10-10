"""Data preparation for the fraud-fiber experiments.

- Features: V1..V28 and log1p(Amount) (29 dims). `Time` is only used to split.
- Chronological split: earliest 80% of transactions -> train, latest 20% -> test.
- Standardization fitted on train only.
- Base space ("fraud score"): logit of an XGBoost fraud probability.
  Train points get out-of-fold (5-fold) scores so no point is scored by a model that saw it;
  test points are scored by a model fitted on the full training set.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
CSV = os.path.join(ROOT, 'data', 'creditcard.csv')
CACHE = os.path.join(ROOT, 'cache', 'prepared.npz')
# SHA-256 of the Kaggle file (mlg-ulb/creditcardfraud, creditcard.csv) used for the article
CSV_SHA256 = '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'

LOGIT_LO, LOGIT_HI = -18.0, 10.0   # covers the full range of XGBoost logits observed on this data
BASE_SCALE = 4.0                   # base coordinate = clipped logit / 4  ->  [-4.5, 2.5]
FEATURES = [f'V{i}' for i in range(1, 29)] + ['logAmount']

XGB_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric='aucpr', n_jobs=-1, tree_method='hist')


def prob_to_base(p):
    p = np.clip(p.astype(np.float64), 1e-30, 1 - 1e-16)
    return np.clip(np.log(p / (1 - p)), LOGIT_LO, LOGIT_HI) / BASE_SCALE


def base_to_prob(b):
    return 1 / (1 + np.exp(-b * BASE_SCALE))


def fit_scorer(X, y, seed=0):
    return xgb.XGBClassifier(random_state=seed, **XGB_PARAMS).fit(X, y)


def prepare(force=False):
    if os.path.exists(CACHE) and not force:
        return dict(np.load(CACHE, allow_pickle=True))
    import hashlib
    with open(CSV, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != CSV_SHA256:
        print(f'WARNING: creditcard.csv differs from the file used in the article (sha256 {digest})')
    df = pd.read_csv(CSV).sort_values('Time', kind='stable').reset_index(drop=True)
    df['logAmount'] = np.log1p(df['Amount'])
    n_train = int(0.8 * len(df))
    tr, te = df.iloc[:n_train], df.iloc[n_train:]

    scaler = StandardScaler().fit(tr[FEATURES])
    X_tr = scaler.transform(tr[FEATURES]).astype(np.float32)
    X_te = scaler.transform(te[FEATURES]).astype(np.float32)
    y_tr, y_te = tr['Class'].values, te['Class'].values

    oof = np.zeros(len(X_tr))
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X_tr, y_tr)):
        oof[b] = fit_scorer(X_tr[a], y_tr[a], seed=f).predict_proba(X_tr[b])[:, 1]
    scorer = fit_scorer(X_tr, y_tr)
    p_te = scorer.predict_proba(X_te)[:, 1]
    scorer.save_model(os.path.join(ROOT, 'cache', 'scorer.json'))

    out = dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
               p_tr=oof, p_te=p_te, b_tr=prob_to_base(oof).astype(np.float32),
               b_te=prob_to_base(p_te).astype(np.float32),
               time_tr=tr['Time'].values, time_te=te['Time'].values,
               amount_tr=tr['Amount'].values, amount_te=te['Amount'].values,
               scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, features=np.array(FEATURES))
    np.savez_compressed(CACHE, **out)
    return out


def load_scorer():
    s = xgb.XGBClassifier()
    s.load_model(os.path.join(ROOT, 'cache', 'scorer.json'))
    return s


if __name__ == '__main__':
    os.makedirs(os.path.join(ROOT, 'cache'), exist_ok=True)
    d = prepare(force=True)
    from sklearn.metrics import roc_auc_score, average_precision_score
    print('train', d['X_tr'].shape, 'frauds', d['y_tr'].sum(), '| test', d['X_te'].shape, 'frauds', d['y_te'].sum())
    print(f"OOF  AUROC {roc_auc_score(d['y_tr'], d['p_tr']):.4f}  AUPRC {average_precision_score(d['y_tr'], d['p_tr']):.4f}")
    print(f"test AUROC {roc_auc_score(d['y_te'], d['p_te']):.4f}  AUPRC {average_precision_score(d['y_te'], d['p_te']):.4f}")
    print('base quantiles train', np.quantile(d['b_tr'], [0, .01, .25, .5, .75, .99, .999, 1]).round(2))
    print('base of frauds (train) quantiles', np.quantile(d['b_tr'][d['y_tr'] == 1], [0, .1, .25, .5, .75, 1]).round(2))
    print('# train points with p>0.5:', (d['p_tr'] > .5).sum(), ' p>0.1:', (d['p_tr'] > .1).sum())
