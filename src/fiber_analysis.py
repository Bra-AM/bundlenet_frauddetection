"""Experiment 4: what do the fibers of a fraud model look like?  (+ all paper figures)

(a) Fiber profiles: BundleNet samples over risk levels p in {0.001, ..., 0.99} vs real transactions
    whose score is closest to each level; distributions of the most important features.
(b) Fiber consistency: requested score vs. XGBoost re-score of generated points (small multiples per model).
(c) Fiber-transport counterfactuals: encode a detected fraud x under its neighborhood, replace the base
    coordinate by a low-risk target and decode under the target neighborhood. Compared with the
    'nearest unlike neighbor' counterfactual (closest real low-risk training transaction).
(d) Training curves.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(int(os.environ.get('THREADS', 10)))
from data import prepare, load_scorer, prob_to_base, base_to_prob, FEATURES
from models import load, path, ROOT

FIG = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG, exist_ok=True)
COLORS = {'BundleNet': '#2a78d6', 'CGAN': '#eb6834', 'CGAN-local': '#1baf7a', 'Gaussian-local': '#eda100',
          'Fiber-SMOTE': '#e87ba4', 'WGAN': '#4a3aa7', 'Real': '#52514e'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False})

d = prepare()
scorer = load_scorer()
X_tr, X_te, b_tr, b_te, p_te, y_te = d['X_tr'], d['X_te'], d['b_tr'], d['b_te'], d['p_te'], d['y_te']
X_all, b_all = np.vstack([X_tr, X_te]), np.concatenate([b_tr, b_te])
fi = {f: i for i, f in enumerate(FEATURES)}
rng = np.random.default_rng(0)
torch.manual_seed(0)
bn = load('BundleNet', 'score', d, 0)

# ------------------------------------------------------------------ (a) fiber profiles
LEVELS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]
SHOW = ['V14', 'V4', 'V10', 'V12', 'V17', 'logAmount']
prof_rows = []
fig, axes = plt.subplots(len(SHOW), 1, figsize=(7.2, 9.5), sharex=True)
for li, p in enumerate(LEVELS):
    y = prob_to_base(np.array([p]))[0]
    gen = bn.sample_from_fiber(np.array([y]), n_per_point=1000, rng=rng)
    ref = X_all[np.argsort(np.abs(b_all - y))[:200]]
    for ax, f in zip(axes, SHOW):
        j = fi[f]
        for k, (vals, name, off) in enumerate([(ref[:, j], 'Real', -0.17), (gen[:, j], 'BundleNet', 0.17)]):
            vp = ax.violinplot([vals], positions=[li + off], widths=0.32, showextrema=False, showmedians=True)
            for b in vp['bodies']:
                b.set_facecolor(COLORS[name]); b.set_edgecolor('white'); b.set_linewidth(1); b.set_alpha(0.85)
            vp['cmedians'].set_color(INK); vp['cmedians'].set_linewidth(1)
        prof_rows.append((p, f, np.median(ref[:, j]), np.median(gen[:, j]), np.std(ref[:, j]), np.std(gen[:, j])))
for ax, f in zip(axes, SHOW):
    ax.set_ylabel(f + ' (std.)', fontsize=8)
    lo, hi = np.percentile(X_all[(b_all > -2.5), fi[f]], [0.2, 99.8])
    ax.set_ylim(lo - 0.5, hi + 0.5)
axes[-1].set_xticks(range(len(LEVELS)), [f'$s(x) = {p:g}$' for p in LEVELS])
axes[-1].set_xlabel('Base point (XGBoost score)')
fig.legend([plt.Rectangle((0, 0), 1, 1, fc=COLORS['Real']), plt.Rectangle((0, 0), 1, 1, fc=COLORS['BundleNet'])],
           ['Real transactions (200 closest in score)', 'BundleNet samples (1000)'], loc='upper left',
           bbox_to_anchor=(0.08, 1.0), ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(os.path.join(FIG, 'fiber_profiles.png')); fig.savefig(os.path.join(FIG, 'fiber_profiles.pdf'))
pd.DataFrame(prof_rows, columns=['p', 'feature', 'median_real', 'median_gen', 'std_real', 'std_gen']).to_csv(
    os.path.join(ROOT, 'results', 'fiber_profiles.csv'), index=False)
plt.close(fig)

# ------------------------------------------------------------------ (b) fiber consistency (small multiples)
models = [m for m in ['BundleNet', 'CGAN', 'CGAN-local', 'Gaussian-local', 'Fiber-SMOTE']
          if m in ('Gaussian-local', 'Fiber-SMOTE') or os.path.exists(path(m, 'score', 0))]
req = np.concatenate([rng.choice(b_te, 1500), rng.choice(b_te[p_te > 0.5], 500)])
fig, axes = plt.subplots(1, len(models), figsize=(2.1 * len(models), 2.4), sharex=True, sharey=True)
cons_rows = []
for ax, name in zip(np.atleast_1d(axes), models):
    m = bn if name == 'BundleNet' else load(name, 'score', d, 0)
    g = m.sample_from_fiber(req, rng=rng)
    got = prob_to_base(scorer.predict_proba(g)[:, 1])
    ax.plot([-18, 10], [-18, 10], color=MUTED, lw=0.8, ls='--')
    ax.scatter(req * 4, got * 4, s=4, color=COLORS[name], alpha=0.35, linewidths=0)
    err = np.abs(got - req)
    shown = {'Fiber-SMOTE': 'SMOTE-local'}.get(name, name)
    ax.set_title(f'{shown}\nbase-point error = {4 * err.mean():.2f}', fontsize=8.5, color=INK)
    cons_rows.append((name, 4 * err.mean(), 4 * err[1500:].mean(), (scorer.predict_proba(g[1500:])[:, 1] > 0.5).mean()))
np.atleast_1d(axes)[0].set_ylabel('XGBoost logit of generated point')
ax.set_xlim(-18, 10); ax.set_ylim(-18, 10)
fig.supxlabel('Requested base point (XGBoost logit)', fontsize=9, y=-0.06)
fig.savefig(os.path.join(FIG, 'fiber_consistency.png')); fig.savefig(os.path.join(FIG, 'fiber_consistency.pdf'))
plt.close(fig)
pd.DataFrame(cons_rows, columns=['model', 'mean_abs_logit_error_all', 'mean_abs_logit_error_high',
                                 'share_high_requests_scored_fraud']).to_csv(
    os.path.join(ROOT, 'results', 'fiber_consistency.csv'), index=False)
print(pd.DataFrame(cons_rows, columns=['model', 'err_all', 'err_high', 'high_share>0.5']))

# ------------------------------------------------------------------ (c) counterfactuals
frauds = np.where((y_te == 1) & (p_te > 0.5))[0]
x0 = X_te[frauds]
xt = torch.as_tensor(x0)
src = bn.nearest_nbhd(b_te[frauds])


def transport(target_p):
    T = float(prob_to_base(np.array([target_p]))[0])
    tgt = bn.nearest_nbhd(np.array([T]))[0]
    out = np.zeros_like(x0)
    with torch.no_grad():
        for i in np.unique(src):
            r = src == i
            enc, _ = bn(xt[r], i)
            enc[:, 0] = T
            out[r] = bn(enc, tgt, rev=True)[0].numpy()
    return out


def summarize(C):
    pc = scorer.predict_proba(C)[:, 1]
    return (pc < 0.5).mean(), np.median(pc), np.linalg.norm(C - x0, axis=1).mean()


low_pool = X_tr[d['p_tr'] < 0.01]
nun = np.array([low_pool[np.linalg.norm(low_pool - xi, axis=1).argmin()] for xi in x0])
low_mean = low_pool.mean(0)
curves = {'BundleNet (change of base point)': [], 'Straight line to nearest unlike neighbor': [],
          'Straight line to low-risk mean': []}
TARGETS = [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
cf_at = {}
for tp in TARGETS:
    C = transport(tp)
    cf_at[tp] = C
    curves['BundleNet (change of base point)'].append((tp, *summarize(C)))
for t in np.linspace(0.05, 1.0, 20):
    curves['Straight line to nearest unlike neighbor'].append((t, *summarize(x0 + t * (nun - x0))))
    curves['Straight line to low-risk mean'].append((t, *summarize(x0 + t * (low_mean - x0))))
rows = [(k, *r) for k, v in curves.items() for r in v]
cfd = pd.DataFrame(rows, columns=['method', 'target_p_or_t', 'validity', 'median_p', 'mean_L2'])
cfd.to_csv(os.path.join(ROOT, 'results', 'counterfactuals.csv'), index=False)
pd.set_option('display.width', 250)
print(f'{len(frauds)} detected test frauds'); print(cfd.round(3).to_string())

fig, ax = plt.subplots(figsize=(4.6, 3.0))
for (name, v), col in zip(curves.items(), [COLORS['BundleNet'], COLORS['CGAN'], COLORS['CGAN-local']]):
    a = np.array([r[1:] for r in v])
    ax.plot(a[:, 2], a[:, 0], color=col, lw=2, marker='o', ms=4, label=name)
ax.set_xlabel('Mean L2 distance to the original transaction (std. units)')
ax.set_ylabel(r'Share with $s(x^\star) < 0.5$')
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
fig.savefig(os.path.join(FIG, 'counterfactual_tradeoff.png')); fig.savefig(os.path.join(FIG, 'counterfactual_tradeoff.pdf'))
plt.close(fig)

# per-feature change at the 1e-4 target
CF_P = 1e-4
fig, ax = plt.subplots(figsize=(7.2, 2.4))
dm = (cf_at[CF_P] - x0).mean(0)
dn = (nun - x0).mean(0)
order = np.argsort(-np.abs(dm))[:12]
w = 0.38
ax.bar(np.arange(12) - w / 2, dm[order], color=COLORS['BundleNet'], width=w, label=r'BundleNet, change of base point (target $s = 10^{-4}$)')
ax.bar(np.arange(12) + w / 2, dn[order], color=COLORS['CGAN'], width=w, label='Nearest unlike neighbor')
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(12), [FEATURES[o] for o in order], fontsize=8)
ax.set_ylabel('Mean change (std. units)')
ax.legend(fontsize=7.5, ncol=2, loc='upper right')
fig.savefig(os.path.join(FIG, 'counterfactual_shift.png')); fig.savefig(os.path.join(FIG, 'counterfactual_shift.pdf'))
plt.close(fig)
pd.DataFrame({'feature': FEATURES, 'transport_mean_change': dm, 'nun_mean_change': dn}).to_csv(
    os.path.join(ROOT, 'results', 'counterfactual_feature_shift.csv'), index=False)

# ------------------------------------------------------------------ (d) training curves
hp = os.path.join(ROOT, 'results', 'history_score_BundleNet_seed0.npy')
if os.path.exists(hp):
    h = np.load(hp)
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.2))
    for ax, k, t in zip(axes, range(4), ['Base MSE', 'KL-bwd', 'KL-fwd', 'Min. distance']):
        ax.plot(h[:, k], color=COLORS['BundleNet'], lw=1.5); ax.set_title(t, fontsize=9, color=INK); ax.set_xlabel('epoch')
    fig.savefig(os.path.join(FIG, 'training_curves.png')); plt.close(fig)
print('figures written to', FIG)
