"""Explanatory figures for the article:
  fibers_illustration  -- the fibers of the fraud model on real data, and the neighborhoods of the base space
  high_risk_gallery    -- real vs generated high-risk transactions for every generator (V14 vs V17)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(int(os.environ.get('THREADS', 6)))
from data import prepare, prob_to_base, FEATURES
from models import load, ROOT

FIG = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG, exist_ok=True)
COLORS = {'BundleNet': '#2a78d6', 'CGAN': '#eb6834', 'CGAN-local': '#1baf7a', 'Gaussian-local': '#eda100',
          'Fiber-SMOTE': '#e87ba4', 'Real': '#52514e'}
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#c9c8c3'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False})

d = prepare()
X_all = np.vstack([d['X_tr'], d['X_te']])
b_all = np.concatenate([d['b_tr'], d['b_te']])
logit_all = 4 * b_all
fi = {f: i for i, f in enumerate(FEATURES)}
A, B = fi['V14'], fi['V17']
rng = np.random.default_rng(0)
torch.manual_seed(0)

# ------------------------------------------------------------------ fibers illustration
LEVELS = [(0.001, '#2a78d6'), (0.5, '#eb6834'), (0.99, '#1baf7a')]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1.15, 1]})
bg = rng.choice(len(X_all), 20000, replace=False)
ax.scatter(X_all[bg, A], X_all[bg, B], s=2, color=BG, linewidths=0, rasterized=True, label='All transactions')
for p, col in LEVELS:
    t = 4 * prob_to_base(np.array([p]))[0]
    idx = np.where(np.abs(logit_all - t) < 0.35)[0]
    idx = rng.choice(idx, min(250, len(idx)), replace=False)
    ax.scatter(X_all[idx, A], X_all[idx, B], s=7, color=col, linewidths=0.3, edgecolors='white',
               label=rf'Fiber over $s(x) \approx {p:g}$')
ax.set_xlim(-20, 6); ax.set_ylim(-26, 8)
ax.set_xlabel('V14 (standardized)'); ax.set_ylabel('V17 (standardized)')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.22), ncol=2, fontsize=7, markerscale=2, handletextpad=0.2, columnspacing=0.8)
ax.set_title('(a) Transaction space $X$ (two of 29 coordinates)', loc='left', fontsize=8.5, color=INK)

# base space: histogram of the score logit with the k-means neighborhoods
bn = load('BundleNet', 'score', d, 0)
centers = np.sort(4 * bn.centers[:, 0].numpy())
edges = np.concatenate([[-18], (centers[1:] + centers[:-1]) / 2, [10]])
ax2.hist(logit_all, bins=np.linspace(-18, 10, 113), color=MUTED, alpha=0.85, log=True)
for k in range(len(edges) - 1):
    if k % 2 == 0:
        ax2.axvspan(edges[k], edges[k + 1], color='#cde2fb', alpha=0.6, lw=0, zorder=0)
for p, col in LEVELS:
    ax2.axvline(4 * prob_to_base(np.array([p]))[0], color=col, lw=2)
ax2.set_xlabel('Base space $Y$: XGBoost logit of $s(x)$')
ax2.set_ylabel('Transactions (log scale)')
ax2.set_title('(b) Base space and its 25 neighborhoods $U_i$', loc='left', fontsize=8.5, color=INK)
ax2.grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fibers_illustration.pdf')); fig.savefig(os.path.join(FIG, 'fibers_illustration.png'))
plt.close(fig)

# ------------------------------------------------------------------ high-risk gallery
real_hi = np.where(b_all > 0)[0]                      # real transactions with s(x) > 0.5 (train + test)
req = rng.choice(d['b_te'][d['p_te'] > 0.5], 600)     # requested high-risk base points (from the test set)
models = ['BundleNet', 'CGAN', 'CGAN-local', 'Gaussian-local', 'Fiber-SMOTE']
fig, axes = plt.subplots(1, len(models) + 1, figsize=(7.2, 1.75), sharex=True, sharey=True)
panels = [('Real', X_all[real_hi])] + [(m, load(m, 'score', d, 0).sample_from_fiber(req, rng=rng)) for m in models]
for ax, (name, P) in zip(axes, panels):
    ax.scatter(X_all[bg[:6000], A], X_all[bg[:6000], B], s=1.2, color=BG, linewidths=0, rasterized=True)
    ax.scatter(P[:, A], P[:, B], s=3, color=COLORS[name], alpha=0.6, linewidths=0, rasterized=True)
    ax.set_title({'Fiber-SMOTE': 'SMOTE-local'}.get(name, name) if name != 'Real' else 'Real ($s(x)>0.5$)', fontsize=8, color=INK)
    ax.set_xlim(-20, 6); ax.set_ylim(-26, 8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel('V17')
fig.supxlabel('V14', fontsize=8.5, y=-0.08)
fig.savefig(os.path.join(FIG, 'high_risk_gallery.pdf')); fig.savefig(os.path.join(FIG, 'high_risk_gallery.png'))
plt.close(fig)
print('written', FIG)
