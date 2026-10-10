"""Model registry: build / train / save / load every generator for both tasks.

task 'score': base space = XGBoost fraud score (continuous, 25 k-means neighborhoods)
task 'class': base space = class label {0, 1} (finite, U_i = {c_i})  -- sanity check
"""
import os
import pickle
import time
import numpy as np
import torch

from bundlenet import BundleNet
from baselines import CGAN, WGANdiv, GaussianLocal, FiberSMOTE, RealTrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
MODEL_DIR = os.path.join(ROOT, 'cache', 'models')

# CPU-scale configuration (see paper, Section "Implementation details")
# Override for the full reference-size run on a GPU, e.g. BN_WIDTH=512 BN_DEPTH=5 BN_EPOCHS=1000 BN_LR=2e-4
_env = os.environ.get
BN_CFG = dict(width=int(_env('BN_WIDTH', 256)), nn_depth=int(_env('BN_DEPTH', 3)), num_inv_blocks=5, num_circles=0)
BN_TRAIN = dict(epochs=int(_env('BN_EPOCHS', 300)), nbhds_per_step=5, sample_size=256,
                lr=float(_env('BN_LR', 5e-4)), lr_halvings=4)
BN_TRAIN_CLASS = dict(epochs=2 * int(_env('BN_EPOCHS', 300)), nbhds_per_step=2, sample_size=256,
                      lr=float(_env('BN_LR', 5e-4)), lr_halvings=4)
GAN_STEPS = int(os.environ.get('GAN_STEPS', 7500))  # 7500 x 256 = 1.92M samples = BundleNet's 300 epochs x 25 nbhds x 256

NAMES = ['BundleNet', 'CGAN', 'CGAN-local', 'WGAN', 'Gaussian-local', 'Fiber-SMOTE', 'Real-train']


def task_data(d, task):
    if task == 'score':
        return d['b_tr'], dict(num_nbhds=25, discrete=False)
    return d['y_tr'].astype(np.float32), dict(num_nbhds=2, discrete=True)


def build(name, task, d, seed):
    base, nb = task_data(d, task)
    X = d['X_tr']
    if name == 'BundleNet':
        return BundleNet(base, X, seed=seed, **nb, **BN_CFG)
    if name == 'CGAN':
        return CGAN(base, X, local=False, seed=seed, **nb)
    if name == 'CGAN-local':
        return CGAN(base, X, local=True, seed=seed, **nb)
    if name == 'WGAN':
        return WGANdiv(X, seed=seed)
    if name == 'Gaussian-local':
        return GaussianLocal(base, X, seed=seed, **nb)
    if name == 'Fiber-SMOTE':
        return FiberSMOTE(base, X)
    if name == 'Real-train':
        return RealTrain(base, X)
    raise KeyError(name)


def path(name, task, seed):
    return os.path.join(MODEL_DIR, f'{task}_{name}_seed{seed}.pt')


def train(name, task, d, seed, log=print):
    torch.manual_seed(seed)
    m = build(name, task, d, seed)
    t = time.time()
    hist = None
    if name == 'BundleNet':
        hist = m.train_net(**(BN_TRAIN if task == 'score' else BN_TRAIN_CLASS), log=log)
    elif name in ('CGAN', 'CGAN-local', 'WGAN'):
        m.train_net(steps=GAN_STEPS, log=log)
    secs = time.time() - t
    save(m, name, task, seed)
    return m, secs, hist


def save(m, name, task, seed):
    os.makedirs(MODEL_DIR, exist_ok=True)
    p = path(name, task, seed)
    if name == 'BundleNet':
        m.save(p)
    elif name in ('CGAN', 'CGAN-local', 'WGAN'):
        torch.save({'G': m.G.state_dict(), 'D': m.D.state_dict()}, p)


def load(name, task, d, seed):
    m = build(name, task, d, seed)
    p = path(name, task, seed)
    if name == 'BundleNet':
        m.load(p)
    elif name in ('CGAN', 'CGAN-local', 'WGAN'):
        ck = torch.load(p, map_location='cpu')
        m.G.load_state_dict(ck['G']); m.D.load_state_dict(ck['D'])
    return m
