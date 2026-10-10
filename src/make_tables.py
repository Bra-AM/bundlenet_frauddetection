"""Write every results table of the article as LaTeX (results/tables/*.tex) from results/*.csv.
No number in the article's tables is typed by hand."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
R = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'results', 'tables')
os.makedirs(OUT, exist_ok=True)

GEN = ['BundleNet', 'CGAN', 'CGAN-local', 'WGAN', 'Gaussian-local', 'Fiber-SMOTE']
SHOW = {'Fiber-SMOTE': 'SMOTE-local', 'Real-train': r'\textit{Real (train)}'}
METRIC_TEX = {'W1': '$W_1$', 'W2': '$W_2$', 'MSMD': 'MSMD', 'KL-fwd': 'KL-fwd', 'KL-bwd': 'KL-bwd',
              'MMD': 'MMD ($\\times 10^{3}$)', 'consistency': 'Base-point error'}
REGIME = {'global': 'Global', 'fiber-all': 'Fiberwise (all)', 'fiber-high': 'Fiberwise (high risk)',
          'fiber-legit': 'Fiberwise (legitimate)', 'fiber-fraud': 'Fiberwise (fraud)'}


def fmt(m, c, digits=3):
    """Mean to `digits` significant figures; CI with the same number of decimals."""
    mag = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
    dec = max(0, digits - 1 - mag)
    return f'{m:.{dec}f} {{\\scriptsize $\\pm$ {c:.{dec}f}}}'


def pretty(name):
    name = 'No augmentation' if name == 'None' else name
    name = name.replace('Fiber-SMOTE', 'SMOTE-local')
    return name.replace(' | class', ' (class label)').replace(' | score', ' (model score)')


def gen_table(task, regimes, metrics, caption, label, models=GEN, higher_better=(), no_best=(), labels=None):
    s = pd.read_csv(os.path.join(R, f'generation_{task}_summary.csv'))
    s.loc[s.metric == 'MMD', ['mean', 'ci95']] *= 1e3
    if task == 'score':  # base-point error in logit units (base coordinate = logit / 4)
        s.loc[s.metric == 'consistency', ['mean', 'ci95']] *= 4
    models = [m for m in models if m in set(s.model)]
    cols = models + (['Real-train'] if 'Real-train' in set(s.model) else [])
    lines = [r'\begin{table*}[t]', r'\caption{' + caption + '}', r'\label{' + label + '}', r'\small',
             r'\begin{tabular}{ll' + 'c' * len(cols) + '}', r'\toprule',
             ' & & ' + ' & '.join(SHOW.get(c, c) for c in cols) + r' \\', r'\midrule']
    for ri, regime in enumerate(regimes):
        rows = []
        for metric in metrics:
            t = s[(s.regime == regime) & (s.metric == metric)].set_index('model')
            if t.empty:
                continue
            vals = t.loc[[m for m in models if m in t.index], 'mean']
            best = vals.idxmax() if metric in higher_better else vals.idxmin()
            # SMOTE-local interpolates between training transactions; also mark the best *learned* model
            learned = vals.drop([m for m in ('Fiber-SMOTE',) if m in vals.index])
            best_learned = learned.idxmax() if metric in higher_better else learned.idxmin()
            if metric in no_best:
                best = best_learned = None
            cells = []
            for m in cols:
                if m not in t.index:
                    cells.append('--'); continue
                c = fmt(t.loc[m, 'mean'], t.loc[m, 'ci95'])
                if m == best:
                    c = r'\best{' + c + '}'
                elif m == best_learned:
                    c = r'\underline{' + c + '}'
                if m == 'Real-train':
                    c = r'\textcolor{gray}{' + c + '}'
                cells.append(c)
            rows.append(((labels or {}).get(metric, METRIC_TEX.get(metric, metric)), cells))
        for k, (mt, cells) in enumerate(rows):
            head = (r'\multirow{' + str(len(rows)) + '}{*}{' + REGIME[regime] + '}') if k == 0 else ''
            lines.append(f'{head} & {mt} & ' + ' & '.join(cells) + r' \\')
        if ri < len(regimes) - 1:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    open(os.path.join(OUT, label.replace('tab:', '') + '.tex'), 'w').write('\n'.join(lines))


MAIN = ['W1', 'W2', 'MSMD', 'KL-fwd', 'KL-bwd']
CI = r'Mean {\scriptsize $\pm$ 95\% CI} over 3 trained models $\times$ 10 repetitions; lower is better. '
gen_table('score', ['global'], MAIN,
          r'Global data generation with the model score as base space: 4{,}000 generated transactions compared with '
          r'4{,}000 held-out test transactions. ' + CI +
          r'The best method is in bold; when it is SMOTE-local, which interpolates between training transactions, '
          r'the best learned model is underlined. \textit{Real (train)} uses real training transactions in place of '
          r'generated ones and serves as a reference.', 'tab:global')
gen_table('score', ['fiber-all', 'fiber-high'], MAIN + ['consistency'],
          r'Fiberwise data generation with the model score as base space: 200 transactions generated over a base point '
          r'are compared with the 50 test transactions whose score is closest to it, for 15 base points drawn from all '
          r'test transactions (\emph{all}) or from those with $s(x) > 0.5$ (\emph{high risk}). The base-point error is '
          r'the mean absolute difference between the requested base point and the XGBoost logit of the generated '
          r'transactions. ' + CI + r'Marking as in Table~\ref{tab:global}.', 'tab:fiber',
          models=['BundleNet', 'CGAN', 'CGAN-local', 'Gaussian-local', 'Fiber-SMOTE'])
gen_table('class', ['global', 'fiber-fraud'], MAIN + ['consistency'],
          r'Data generation with the class label as base space. In the fiber over the fraud class, generated transactions '
          r'are compared with the 74 test frauds. The last row is the share of generated fraud that the XGBoost model '
          r'scores above $0.5$; the same share among the real test frauds is $0.70$ (\textit{Real (train)} reaches $1.00$ '
          r'because the XGBoost model was fitted on these transactions). ' + CI, 'tab:class',
          models=['BundleNet', 'CGAN', 'CGAN-local', 'Gaussian-local'], no_best=('consistency',),
          labels={'consistency': r'Share with $s(x) > 0.5$'})


def simple_table(csv, row, caption, label, metrics, exclude=(), env='table'):
    s = pd.read_csv(os.path.join(R, csv), keep_default_na=False)  # the row label 'None' must stay a string
    s = s[~s[row].isin(exclude)]
    rows = list(dict.fromkeys(s[row]))
    lines = [r'\begin{' + env + '}[t]', r'\caption{' + caption + '}', r'\label{' + label + '}', r'\small',
             r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{l' + 'c' * len(metrics) + '}', r'\toprule',
             ' & ' + ' & '.join(m.replace('%', r'\%') for m in metrics) + r' \\', r'\midrule']
    best = {m: s[s.metric == m]['mean'].max() for m in metrics}
    for r_ in rows:
        t = s[s[row] == r_].set_index('metric')
        cells = []
        for m in metrics:
            c = fmt(t.loc[m, 'mean'], t.loc[m, 'ci95'])
            if np.isclose(t.loc[m, 'mean'], best[m]):
                c = r'\best{' + c + '}'
            cells.append(c)
        lines.append(pretty(r_) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{' + env + '}']
    open(os.path.join(OUT, label.replace('tab:', '') + '.tex'), 'w').write('\n'.join(lines))


simple_table('augmentation_summary.csv', 'augmentation',
             r'XGBoost trained on the training set plus 2{,}000 generated fraudulent transactions and evaluated on the '
             r'test set (75 frauds). The base space used by each generator is given in parentheses. Mean '
             r'{\scriptsize $\pm$ 95\% CI} over 3 seeds; higher is better.',
             'tab:augment', ['AUPRC', 'AUROC', 'Recall@FPR=0.1%', 'F1@0.5'])
simple_table('classification_summary.csv', 'model',
             r'Classification on the test set. BundleNet scores each transaction by the log-likelihood ratio of the '
             r'two class fibers (Section~\ref{sec:llr}); no label is used at test time. Mean {\scriptsize $\pm$ 95\% CI} '
             r'over 3 seeds; higher is better.',
             'tab:classify', ['AUPRC', 'AUROC', 'F1'])

# training times (appendix)
t = pd.read_csv(os.path.join(R, 'train_times.csv')).groupby(['task', 'model'])['seconds'].mean().reset_index()
lines = [r'\begin{table}[H]', r'\caption{Wall-clock training time per model on a laptop CPU (Intel Core i7-1265U, '
         r'no GPU), in minutes, averaged over the three seeds. Two training jobs shared the CPU (6 and 4 threads).}',
         r'\label{tab:times}', r'\small', r'\begin{tabular}{llc}', r'\toprule', r'Base space & Model & Minutes \\', r'\midrule']
for _, r_ in t.iterrows():
    base = 'Class label' if r_['task'] == 'class' else 'Model score'
    lines.append(f"{base} & {r_['model']} & {r_['seconds'] / 60:.1f} \\\\")
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(os.path.join(OUT, 'times.tex'), 'w').write('\n'.join(lines))

# paired bootstrap significance of the augmentation results (appendix)
t = pd.read_csv(os.path.join(R, 'augmentation_significance.csv'))
lines = [r'\begin{table}[H]', r'\caption{Paired bootstrap (2{,}000 resamples of the test set) of the change in AUPRC '
         r'relative to no augmentation, using detector scores averaged over the three seeds. $p$ is the one-sided '
         r'bootstrap probability of no improvement, without correction for multiple comparisons.}', r'\label{tab:sig}',
         r'\small', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lccc}', r'\toprule',
         r'Augmentation & $\Delta$AUPRC & 95\% interval & $p$ \\', r'\midrule']
for _, r_ in t.iterrows():
    lines.append(f"{pretty(r_['augmentation'])} & {r_['delta_AUPRC']:+.3f} & [{r_['ci_low']:+.3f}, {r_['ci_high']:+.3f}] & {r_['p_one_sided']:.3f} \\\\")
lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
open(os.path.join(OUT, 'sig.tex'), 'w').write('\n'.join(lines))
print('tables written to', os.path.abspath(OUT))
