"""Two-sample metrics, following Courts & Kvinge (2022), Appendix A.2 and their `bundlenet/util.py`.

MMD (gaussian, blur 0.5), W1/W2 (debiased Sinkhorn, blur 0.01) use geomloss exactly as in the original.
KL (k-NN estimator of Wang et al. 2009, k=1) and MSMD are vectorized but numerically identical
to the original loop implementations.
"""
import numpy as np
import torch
import geomloss
from scipy.stats import bootstrap

_mmd = geomloss.SamplesLoss('gaussian', blur=0.5)
_w1 = geomloss.SamplesLoss('sinkhorn', p=1, blur=0.01, debias=True)
_w2 = geomloss.SamplesLoss('sinkhorn', p=2, blur=0.01, debias=True)


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def cdist(a, b):
    # exact pairwise distances: the default matrix-multiply mode returns small non-zero self-distances,
    # which would be mistaken for nearest neighbours by the k-NN estimators below
    return torch.cdist(a, b, compute_mode='donot_use_mm_for_euclid_dist')


def kl_divergence(s1, s2, k=1):
    """k-NN estimate of D_KL(P1 || P2) from samples s1 ~ P1, s2 ~ P2 (differentiable)."""
    n, m, d = len(s1), len(s2), s1.shape[1]
    d12 = cdist(s1, s2)
    d11 = cdist(s1, s1)
    big = torch.finfo(d11.dtype).max
    # exclude exact zeros (self-distances and duplicates), as in the original
    nu = torch.where(d12 == 0, torch.full_like(d12, big), d12).kthvalue(k, dim=1).values
    rho = torch.where(d11 == 0, torch.full_like(d11, big), d11).kthvalue(k, dim=1).values
    return np.log(m / (n - 1)) + (d / n) * torch.log(nu / rho).sum()


def msmd(sample, true_pts):
    return cdist(sample, true_pts).min(dim=1).values.pow(2).mean()


METRICS = ['MMD', 'MSMD', 'KL-fwd', 'KL-bwd', 'W1', 'W2']


def all_metrics(real, gen):
    """real: points from the true distribution; gen: generated points."""
    real, gen = _t(real), _t(gen)
    with torch.no_grad():
        return {
            'MMD': _mmd(real, gen).item(),
            'MSMD': msmd(real, gen).item(),  # argument order as in the original evaluation.py
            'KL-fwd': kl_divergence(real, gen).item(),
            'KL-bwd': kl_divergence(gen, real).item(),
            'W1': _w1(real, gen).item(),
            'W2': _w2(real, gen).item(),
        }


def mean_ci(values, seed=0):
    """Sample mean with bootstrapped 95% CI half-width (as in the original evaluation script)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or np.allclose(v, v[0]):
        return v.mean(), 0.0
    ci = bootstrap((v,), np.mean, confidence_level=0.95, random_state=seed).confidence_interval
    return v.mean(), (ci.high - ci.low) / 2
