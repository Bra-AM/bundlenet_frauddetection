"""Bundle Network (Courts & Kvinge, ICLR 2022), re-implemented for tabular fraud data.

Faithful to the reference implementation (github.com/NicoCourts/bundle-networks, bundlenet/models/bundlenet.py):
  * neighborhoods U_i of the base space from k-means (or one per value for a finite base space);
  * an INN Phi(-, r_i): one ConditionalAffineTransform conditioned on the neighborhood centre r_i,
    followed by `num_inv_blocks` x (RNVP coupling block, random permutation);
  * forward loss  L_fwd = ||p_Y(Phi_i(x)) - y||;
  * backward loss L_bwd = KL-fwd + KL-bwd + mean minimum distance between Phi_i^{-1}(y, z), z ~ D_Z,
    and real points of pi^{-1}(U_i);  total = 10 L_fwd + L_bwd  (weights [10, 1, 1, 1]);
  * fiber prior D_Z: per-neighborhood Gaussian (and optional circles) reparameterized from the
    sample statistics of the encoded neighborhood.
Changes, all for speed / scale (documented in the paper):
  * the O(n^2) Python loops for KL and distance losses are replaced by torch.cdist (same values);
  * each neighborhood is subsampled to `sample_size` points per step (as the reference script does
    with sample_size=400), and one latent sample is drawn per base point instead of five;
  * gradient-norm clipping for stability with heavy-tailed financial features.
"""
import numpy as np
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from sklearn.cluster import KMeans

from metrics import kl_divergence, cdist


class BundleNet(nn.Module):
    def __init__(self, base, cover, num_nbhds=25, discrete=False, width=512, num_inv_blocks=5,
                 nn_depth=5, num_circles=0, total_dim=None, seed=0):
        super().__init__()
        base = np.asarray(base, dtype=np.float32).reshape(len(base), -1)
        cover = np.asarray(cover, dtype=np.float32).reshape(len(cover), -1)
        self.base_dim, self.cover_dim = base.shape[1], cover.shape[1]
        self.total_dim = total_dim or max(self.base_dim, self.cover_dim)
        self.num_circles = num_circles
        if 2 * num_circles > self.total_dim - self.base_dim:
            raise ValueError('Too many circles for the fiber dimension.')
        self.width, self.nn_depth = width, nn_depth
        self.seed = seed

        self.base = torch.zeros(len(base), self.total_dim)
        self.base[:, :self.base_dim] = torch.from_numpy(base)
        self.cover = torch.zeros(len(cover), self.total_dim)
        self.cover[:, :self.cover_dim] = torch.from_numpy(cover)

        # Neighborhoods of the base space
        if discrete:
            values = np.unique(base, axis=0)
            labels = np.array([np.where((values == b).all(1))[0][0] for b in base])
            centers = values
        else:
            km = KMeans(num_nbhds, n_init=10, random_state=seed).fit(base)
            labels, centers = km.labels_, km.cluster_centers_
        self.num_nbhds = len(centers)
        c = torch.zeros(self.num_nbhds, self.total_dim)
        c[:, :self.base_dim] = torch.as_tensor(centers, dtype=torch.float32)
        self.register_buffer('centers', c)
        self.nbhd_idx = [np.where(labels == i)[0] for i in range(self.num_nbhds)]

        torch.manual_seed(seed)
        self.model = self._construct_net(num_inv_blocks)

    # ---------------------------------------------------------------- architecture
    def _fc(self, c_in, c_out):
        layers = [nn.Linear(c_in, self.width), nn.ReLU()]
        for _ in range(self.nn_depth):
            layers += [nn.Linear(self.width, self.width), nn.ReLU()]
        layers.append(nn.Linear(self.width, c_out))
        return nn.Sequential(*layers)

    def _construct_net(self, num_inv_blocks):
        inp = Ff.InputNode(self.total_dim, name='input')
        cond = Ff.ConditionNode(self.total_dim, name='condition')
        nodes = [inp, cond, Ff.Node(inp, Fm.ConditionalAffineTransform, {'subnet_constructor': self._fc},
                                    conditions=cond, name='local_trivialization')]
        for i in range(num_inv_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.RNVPCouplingBlock, {'subnet_constructor': self._fc}, name=f'rnvp_{i}'))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': self.seed * 100 + i}, name=f'perm_{i}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        return Ff.GraphINN(nodes)

    def forward(self, x, nbhd, rev=False):
        c = self.centers[nbhd].expand(len(x), -1)
        return self.model(x, c=[c], rev=rev)

    def nearest_nbhd(self, base_pts):
        b = torch.as_tensor(np.asarray(base_pts, dtype=np.float32)).reshape(len(base_pts), -1)
        return torch.cdist(b, self.centers[:, :self.base_dim]).argmin(1).numpy()

    # ---------------------------------------------------------------- fiber prior D_Z
    def _sample_fiber_params(self, enc, n):
        """Sample n latent fiber vectors using the statistics of encoded points `enc` (as in the original)."""
        fib = enc[:, self.base_dim:]
        means, stds = fib.mean(0), fib.std(0, unbiased=True)
        params = means + stds * torch.randn(n, fib.shape[1], device=enc.device)
        if self.num_circles:
            circ = enc[:, self.base_dim:self.base_dim + 2 * self.num_circles].reshape(len(enc), self.num_circles, 2).norm(dim=-1)
            radii = circ.mean(0) + circ.std(0, unbiased=True) * torch.randn(n, self.num_circles, device=enc.device)
            ang = 2 * np.pi * torch.rand(n, self.num_circles, device=enc.device)
            params[:, :2 * self.num_circles] = torch.stack([radii * ang.cos(), radii * ang.sin()], -1).reshape(n, -1)
        return params

    # ---------------------------------------------------------------- training
    def train_net(self, epochs=400, nbhds_per_step=5, sample_size=256, lr=1e-4, weights=(10., 1., 1., 1.),
                  lr_halvings=6, clip=10.0, log_every=20, log=print, device=None):
        # optional GPU training; the model is moved back to the CPU at the end so that all inference is CPU-only
        dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        steps_per_epoch = int(np.ceil(self.num_nbhds / nbhds_per_step))
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs * steps_per_epoch // (lr_halvings + 1)), gamma=0.5)
        rng = np.random.default_rng(self.seed)
        history = []
        for epoch in range(epochs):
            agg = np.zeros(4)
            for chunk in np.array_split(rng.permutation(self.num_nbhds), steps_per_epoch):
                self.model.train()
                opt.zero_grad()
                terms = torch.zeros(4, device=dev)
                for i in chunk:
                    idx = self.nbhd_idx[i]
                    if len(idx) > sample_size:
                        idx = rng.choice(idx, sample_size, replace=False)
                    x, b = self.cover[idx].to(dev), self.base[idx].to(dev)
                    enc, _ = self(x, i)
                    mse = (enc[:, :self.base_dim] - b[:, :self.base_dim]).norm(dim=1).mean()
                    samples = b.clone()
                    samples[:, self.base_dim:] = self._sample_fiber_params(enc, len(idx))
                    gen, _ = self(samples, i, rev=True)
                    dist = cdist(gen, x).min(1).values.mean()
                    kl_b = kl_divergence(gen, x)
                    kl_f = kl_divergence(x, gen)
                    terms = terms + torch.stack([mse, kl_b, kl_f, dist])
                terms = terms / len(chunk)
                loss = sum(w * t for w, t in zip(weights, terms))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}: {terms.tolist()}')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                opt.step()
                sched.step()
                agg += terms.detach().cpu().numpy() / steps_per_epoch
            history.append(agg)
            if log and (epoch % log_every == 0 or epoch == epochs - 1):
                log(f'epoch {epoch:4d} | MSE {agg[0]:.4f} | KL-bwd {agg[1]:8.3f} | KL-fwd {agg[2]:8.3f} | Dist {agg[3]:.4f}')
        self.to('cpu')
        return np.array(history)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def sample_from_fiber(self, base_pts, n_per_point=1, estimate_size=200, rng=None):
        """Generate n_per_point samples over each base point. Points are grouped by neighborhood so the
        prior statistics are computed once per neighborhood (equivalent to the original per-point call)."""
        self.model.eval()
        rng = rng or np.random.default_rng()
        base_pts = np.asarray(base_pts, dtype=np.float32).reshape(len(base_pts), -1)
        base_pts = np.repeat(base_pts, n_per_point, axis=0)
        out = np.zeros((len(base_pts), self.cover_dim), dtype=np.float32)
        nb = self.nearest_nbhd(base_pts)
        for i in np.unique(nb):
            rows = np.where(nb == i)[0]
            idx = self.nbhd_idx[i]
            idx = rng.choice(idx, min(estimate_size, len(idx)), replace=False)
            enc, _ = self(self.cover[idx], i)
            s = torch.zeros(len(rows), self.total_dim)
            s[:, :self.base_dim] = torch.from_numpy(base_pts[rows])
            s[:, self.base_dim:] = self._sample_fiber_params(enc, len(rows))
            out[rows] = self(s, i, rev=True)[0][:, :self.cover_dim].numpy()
        return out

    @torch.no_grad()
    def log_likelihood(self, x, nbhd, estimate_size=2000, rng=None):
        """log p(x | U_nbhd) under the learned local trivialization: Gaussian on the base coordinate
        (residual std of training points), the reparameterized Gaussian fiber prior, and log|det J|.
        Circular priors are treated as Gaussians here (num_circles=0 in our experiments)."""
        self.model.eval()
        rng = rng or np.random.default_rng(0)
        idx = self.nbhd_idx[nbhd]
        idx = rng.choice(idx, min(estimate_size, len(idx)), replace=False)
        enc_ref, _ = self(self.cover[idx], nbhd)
        res = enc_ref[:, :self.base_dim] - self.base[idx, :self.base_dim]
        mu = torch.cat([self.centers[nbhd, :self.base_dim], enc_ref[:, self.base_dim:].mean(0)])
        sd = torch.cat([res.std(0).clamp_min(1e-3), enc_ref[:, self.base_dim:].std(0)])
        xt = torch.zeros(len(x), self.total_dim)
        xt[:, :self.cover_dim] = torch.as_tensor(np.asarray(x, dtype=np.float32))
        out = []
        for chunk in torch.split(xt, 8192):
            enc, logdet = self(chunk, nbhd)
            lp = torch.distributions.Normal(mu, sd).log_prob(enc).sum(1) + logdet
            out.append(lp)
        return torch.cat(out).numpy()

    def save(self, path):
        torch.save({'state': self.model.state_dict(), 'centers': self.centers}, path)

    def load(self, path):
        ck = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ck['state'])
        self.centers.copy_(ck['centers'])
        return self
