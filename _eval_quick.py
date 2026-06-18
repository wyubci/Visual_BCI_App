import numpy as np, scipy.io, os, sys

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files = sorted(os.listdir(root))
files = [f for f in files if '213110' not in f]
print(f'Files: {len(files)}')

trials_list, labels_list = [], []
for f in files:
    path = os.path.join(root, f)
    mat = scipy.io.loadmat(path)
    raw = mat['data'] if 'data' in mat else mat['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    onset = int(round((0.5 + 0.14) * 250))
    length = 250
    trial = raw[:, onset:onset+length] if raw.shape[1] >= onset+length else raw[:, :length]
    trials_list.append(trial)
    for i, cmd in enumerate(['前进','后退','左转','停止','右转']):
        if cmd in f: labels_list.append(i); break

trials = np.array(trials_list); labels = np.array(labels_list)
print(f'Data: {trials.shape}, per class: {dict(zip(*np.unique(labels, return_counts=True)))}')

for i in range(trials.shape[0]):
    trials[i] = trials[i] - np.mean(trials[i], axis=1, keepdims=True)
    ch_std = np.std(trials[i], axis=1, keepdims=True)
    trials[i] = np.clip(trials[i] / (ch_std + 1e-6), -6, 6)

from models.TDCA import TDCA, MultiLagTDCA
from models.FBCCA import FBCCA
from models.CCA import CCA
freqs = [6.875, 7.5, 8.25, 10.3125, 13.75]

bs, n = 5, len(files)
n_blocks = n // bs

for name, cls, need_fit in [('TDCA', TDCA, True), ('MultiLagTDCA', MultiLagTDCA, True), ('FBCCA', FBCCA, False), ('CCA', CCA, False)]:
    cm = np.zeros((5,5), dtype=int)
    for b in range(n_blocks):
        ti = list(range(b*bs, (b+1)*bs))
        vi = [i for i in range(n) if i not in ti]
        Xtr, ytr = trials[vi], labels[vi]
        Xte, yte = trials[ti], labels[ti]
        try:
            model = cls(5, 1.0, freqs, Nh=8, sample_rate=250)
            if need_fit: model.fit(Xtr, ytr)
            for j in range(Xte.shape[0]):
                p = model.classify(Xte[j])
                p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
                cm[yte[j], p] += 1
        except Exception as e:
            print(f'  {name} fold {b}: {e}')
    acc = 100 * np.sum(np.diag(cm)) / max(np.sum(cm), 1)
    per_class = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
    print(f'{name}: {np.sum(np.diag(cm))}/{np.sum(cm)} = {acc:.1f}%  per: {per_class}')
    if name == 'MultiLagTDCA' or name == 'TDCA':
        cmds = ['前进','后退','左转','停止','右转']
        for i, cmd in enumerate(cmds):
            print(f'  {cmd}: {list(cm[i])}')
