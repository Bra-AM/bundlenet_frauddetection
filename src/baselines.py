"""Baseline generators. GAN architectures follow the BundleNet reference repository
(bundlenet/models/benchmarking_models: wgan.py = WGAN-div, cgan.py = CGAN, cgan_nbhd.py = CGAN-local).

Every model exposes  sample_from_fiber(base_pts, n_per_point=1, rng=None) -> (len*n, cover_dim) array.

Changes vs. the reference: minibatches are drawn neighborhood-balanced (a neighborhood uniformly at random,
then a point inside it), exactly as BundleNet sees the data -- without this the conditional GANs would almost
never see the rare high-risk region; generators are sampled in eval mode (BatchNorm running statistics).
"""
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from sklearn.cluster import KMeans


def make_nbhds(base, num_nbhds, discrete, seed):
    base = np.asarray(base, dtype=np.float32).reshape(len(base), -1)
    if discrete:
        centers = np.unique(base, axis=0)
        labels = np.array([np.where((centers == b).all(1))[0][0] for b in base])
    else:
        km = KMeans(num_nbhds, n_init=10, random_state=seed).fit(base)
        labels, centers = km.labels_, km.cluster_centers_.astype(np.float32)
    return centers, [np.where(labels == i)[0] for i in range(len(centers))]


def balanced_batch(nbhd_idx, n, rng):
    which = rng.integers(len(nbhd_idx), size=n)
    return np.array([nbhd_idx[w][rng.integers(len(nbhd_idx[w]))] for w in which])


def _gen_net(in_dim, out_dim):
    def block(i, o, norm=True):
        return [nn.Linear(i, o)] + ([nn.BatchNorm1d(o, 0.8)] if norm else []) + [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*block(in_dim, 128, False), *block(128, 256), *block(256, 512), *block(512, 1024),
                         nn.Linear(1024, out_dim))


class _CondGenerator(nn.Module):
    def __init__(self, latent, data, cond):
        super().__init__()
        self.emb = nn.Linear(cond, cond)
        self.net = _gen_net(latent + cond, data)

    def forward(self, z, c):
        return self.net(torch.cat([self.emb(c), z], -1))


class _CondDiscriminator(nn.Module):
    def __init__(self, data, cond):
        super().__init__()
        self.emb = nn.Linear(cond, cond)
        self.net = nn.Sequential(nn.Linear(cond + data, 512), nn.LeakyReLU(0.2, True),
                                 nn.Linear(512, 512), nn.Dropout(0.4), nn.LeakyReLU(0.2, True),
                                 nn.Linear(512, 512), nn.Dropout(0.4), nn.LeakyReLU(0.2, True),
                                 nn.Linear(512, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, self.emb(c)], -1))


class CGAN:
    """Conditional (LS)GAN. local=False: condition on the base point y (CGAN).
    local=True: condition on [neighborhood centre, y] (CGAN-local of Courts & Kvinge)."""

    def __init__(self, base, cover, local=False, num_nbhds=25, discrete=False, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.base = np.asarray(base, dtype=np.float32).reshape(len(base), -1)
        self.cover = np.asarray(cover, dtype=np.float32)
        self.local = local
        self.centers, self.nbhd_idx = make_nbhds(self.base, num_nbhds, discrete, seed)
        self.latent = self.cover.shape[1]
        cdim = self.base.shape[1] * (2 if local else 1)
        self.G = _CondGenerator(self.latent, self.cover.shape[1], cdim)
        self.D = _CondDiscriminator(self.cover.shape[1], cdim)

    def _cond(self, b):
        b = np.asarray(b, dtype=np.float32).reshape(len(b), -1)
        if self.local:
            nb = np.linalg.norm(b[:, None, :] - self.centers[None], axis=-1).argmin(1)
            b = np.concatenate([self.centers[nb], b], 1)
        return torch.from_numpy(b)

    def train_net(self, steps=20000, batch_size=256, lr=1e-4, log_every=2000, log=print, device=None):
        dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.G.to(dev); self.D.to(dev)
        oG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        oD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        mse = nn.MSELoss()
        ones, zeros = torch.ones(batch_size, 1, device=dev), torch.zeros(batch_size, 1, device=dev)
        for s in range(steps):
            self.G.train(); self.D.train()
            idx = balanced_batch(self.nbhd_idx, batch_size, self.rng)
            real, c = torch.from_numpy(self.cover[idx]).to(dev), self._cond(self.base[idx]).to(dev)
            c_fake = self._cond(self.base[balanced_batch(self.nbhd_idx, batch_size, self.rng)]).to(dev)
            oG.zero_grad()
            fake = self.G(torch.randn(batch_size, self.latent, device=dev), c_fake)
            g = mse(self.D(fake, c_fake), ones)
            g.backward(); oG.step()
            oD.zero_grad()
            d = (mse(self.D(real, c), ones) + mse(self.D(fake.detach(), c_fake), zeros)) / 2
            d.backward(); oD.step()
            if log and (s % log_every == 0 or s == steps - 1):
                log(f'step {s:6d} | G {g.item():.4f} | D {d.item():.4f}')
        self.G.to('cpu'); self.D.to('cpu')

    @torch.no_grad()
    def sample_from_fiber(self, base_pts, n_per_point=1, rng=None):
        self.G.eval()
        b = np.repeat(np.asarray(base_pts, dtype=np.float32).reshape(len(base_pts), -1), n_per_point, 0)
        return self.G(torch.randn(len(b), self.latent), self._cond(b)).numpy()


class WGANdiv:
    """Unconditional WGAN-div (Wu et al. 2018); global generation only (ignores the base point)."""

    def __init__(self, cover, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cover = np.asarray(cover, dtype=np.float32)
        self.latent = self.cover.shape[1]
        self.G = _gen_net(self.latent, self.cover.shape[1])
        self.D = nn.Sequential(nn.Linear(self.cover.shape[1], 512), nn.LeakyReLU(0.2, True),
                               nn.Linear(512, 256), nn.LeakyReLU(0.2, True), nn.Linear(256, 1))

    def train_net(self, steps=20000, batch_size=256, lr=1e-4, n_critic=5, k=2, p=6, log_every=2000, log=print,
                  device=None):
        dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.G.to(dev); self.D.to(dev)
        oG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        oD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        for s in range(steps):
            self.G.train()
            real = torch.from_numpy(self.cover[self.rng.integers(len(self.cover), size=batch_size)]).to(dev).requires_grad_(True)
            z = torch.randn(batch_size, self.latent, device=dev)
            fake = self.G(z)
            rv, fv = self.D(real), self.D(fake)
            rg = autograd.grad(rv, real, torch.ones_like(rv), create_graph=True)[0]
            fg = autograd.grad(fv, fake, torch.ones_like(fv), create_graph=True)[0]
            gp = (rg.pow(2).sum(1) ** (p / 2) + fg.pow(2).sum(1) ** (p / 2)).mean() * k / 2
            d = -rv.mean() + fv.mean() + gp
            oD.zero_grad(); d.backward(); oD.step()
            if s % n_critic == 0:
                oG.zero_grad()
                g = -self.D(self.G(z)).mean()
                g.backward(); oG.step()
            if log and (s % log_every == 0 or s == steps - 1):
                log(f'step {s:6d} | G {g.item():.4f} | D {d.item():.4f}')
        self.G.to('cpu'); self.D.to('cpu')

    @torch.no_grad()
    def sample_from_fiber(self, base_pts, n_per_point=1, rng=None):
        self.G.eval()
        return self.G(torch.randn(len(base_pts) * n_per_point, self.latent)).numpy()


class GaussianLocal:
    """Per-neighborhood joint Gaussian of (x, y); samples x | y from the exact conditional Gaussian.
    A strong, transparent 'classical statistics' baseline for local trivialization."""

    def __init__(self, base, cover, num_nbhds=25, discrete=False, seed=0, ridge=1e-4):
        self.base = np.asarray(base, dtype=np.float64).reshape(len(base), -1)
        self.cover = np.asarray(cover, dtype=np.float64)
        self.centers, self.nbhd_idx = make_nbhds(self.base, num_nbhds, discrete, seed)
        self.params = []
        bd = self.base.shape[1]
        for idx in self.nbhd_idx:
            X, Y = self.cover[idx], self.base[idx]
            mx, my = X.mean(0), Y.mean(0)
            Sxx = np.cov(X, rowvar=False) + ridge * np.eye(X.shape[1])
            if discrete or len(idx) < 3 or Y.std() < 1e-8:
                A, S = np.zeros((X.shape[1], bd)), Sxx
            else:
                J = np.cov(np.hstack([X, Y]), rowvar=False)
                Sxy, Syy = J[:-bd, -bd:], J[-bd:, -bd:] + ridge * np.eye(bd)
                A = Sxy @ np.linalg.inv(Syy)
                S = Sxx - A @ Sxy.T
            self.params.append((mx, my, A, np.linalg.cholesky(S + 1e-6 * np.eye(len(S)))))

    def train_net(self, **kw):
        pass

    def sample_from_fiber(self, base_pts, n_per_point=1, rng=None):
        rng = rng or np.random.default_rng()
        b = np.repeat(np.asarray(base_pts, dtype=np.float64).reshape(len(base_pts), -1), n_per_point, 0)
        nb = np.linalg.norm(b[:, None, :] - self.centers[None], axis=-1).argmin(1)
        out = np.zeros((len(b), self.cover.shape[1]))
        for i in np.unique(nb):
            r = nb == i
            mx, my, A, L = self.params[i]
            out[r] = mx + (b[r] - my) @ A.T + rng.standard_normal((r.sum(), len(mx))) @ L.T
        return out.astype(np.float32)


class FiberSMOTE:
    """SMOTE restricted to a fiber: interpolate between random pairs among the k training points whose
    base value is closest to y. The standard oversampling heuristic in fraud detection, made fiber-aware."""

    def __init__(self, base, cover, k=10, **kw):
        self.base = np.asarray(base, dtype=np.float32).reshape(len(base), -1)
        self.cover = np.asarray(cover, dtype=np.float32)
        self.k = k
        self.order = np.argsort(self.base[:, 0]) if self.base.shape[1] == 1 else None

    def train_net(self, **kw):
        pass

    def _knn(self, y):
        if self.order is not None:  # 1-D base: binary search
            sb = self.base[self.order, 0]
            pos = np.searchsorted(sb, y[0])
            lo, hi = max(0, pos - self.k), min(len(sb), pos + self.k)
            cand = self.order[lo:hi]
            return cand[np.argsort(np.abs(self.base[cand, 0] - y[0]))[:self.k]]
        return np.argsort(np.linalg.norm(self.base - y, axis=1))[:self.k]

    def sample_from_fiber(self, base_pts, n_per_point=1, rng=None):
        rng = rng or np.random.default_rng()
        base_pts = np.asarray(base_pts, dtype=np.float32).reshape(len(base_pts), -1)
        out = []
        for y in base_pts:
            nn_ = self._knn(y)
            a, b = rng.choice(nn_, n_per_point), rng.choice(nn_, n_per_point)
            lam = rng.random((n_per_point, 1))
            out.append(self.cover[a] + lam * (self.cover[b] - self.cover[a]))
        return np.concatenate(out).astype(np.float32)


class RealTrain:
    """Reference, not a generator: returns *real training* transactions from the requested fiber
    (random draws, without replacement, among the max(k, n) training points closest in base value).
    Its scores are the floor any model trained on past data can be expected to reach on future data."""

    def __init__(self, base, cover, k=50, **kw):
        self.base = np.asarray(base, dtype=np.float32).reshape(len(base), -1)[:, 0]
        self.cover = np.asarray(cover, dtype=np.float32)
        u = np.sort(np.unique(self.cover, axis=0, return_index=True)[1])  # unique transactions, as in the evaluation
        self.base, self.cover = self.base[u], self.cover[u]
        self.k = k
        self.order = np.argsort(self.base)
        self.sorted = self.base[self.order]

    def train_net(self, **kw):
        pass

    def _nearest(self, y, m):
        pos = np.searchsorted(self.sorted, y)
        lo, hi = max(0, pos - m), min(len(self.sorted), pos + m)
        cand = self.order[lo:hi]
        cand = cand[np.argsort(np.abs(self.base[cand] - y))[:m]]
        if np.abs(self.base[cand] - y).max() == 0:  # finite base space: the whole (tied) fiber
            cand = np.where(self.base == y)[0]
        return cand

    def sample_from_fiber(self, base_pts, n_per_point=1, rng=None):
        rng = rng or np.random.default_rng()
        base_pts = np.asarray(base_pts, dtype=np.float32).reshape(len(base_pts), -1)[:, 0]
        out = [self.cover[rng.choice(self._nearest(y, max(self.k, n_per_point)), n_per_point, replace=False)]
               for y in base_pts]
        return np.concatenate(out)
