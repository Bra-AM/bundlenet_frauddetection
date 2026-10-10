"""Train every model.  Usage:  python train_all.py score 0 1 2   (task, then seeds)
Optional env MODELS=BundleNet,CGAN to restrict."""
import os
import sys
import csv
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('THREADS', 10)))
from data import prepare
from models import NAMES, train, path, ROOT

task, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]] or [0]
names = os.environ.get('MODELS', ','.join(NAMES)).split(',')
d = prepare()
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
log_path = os.path.join(ROOT, 'results', 'train_times.csv')

for seed in seeds:
    for name in names:
        if os.path.exists(path(name, task, seed)) and not os.environ.get('FORCE'):
            print(f'skip {task} {name} seed {seed} (exists)'); continue
        if name in ('Gaussian-local', 'Fiber-SMOTE', 'Real-train'):
            continue  # fitted on the fly at evaluation time (closed form / non-parametric)
        print(f'=== {task} | {name} | seed {seed}', flush=True)
        m, secs, hist = train(name, task, d, seed, log=lambda s: print(s, flush=True))
        if hist is not None:
            np.save(os.path.join(ROOT, 'results', f'history_{task}_{name}_seed{seed}.npy'), hist)
        new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['task', 'model', 'seed', 'seconds'])
            w.writerow([task, name, seed, f'{secs:.1f}'])
        print(f'done in {secs/60:.1f} min', flush=True)
