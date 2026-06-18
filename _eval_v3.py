import numpy as np, scipy.io, os

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files = sorted(os.listdir(root))
nb = [f for f in files if int(f.split('_')[-1].replace('.mat','')) > 222800]
print(f'新 block: {len(nb)} 条')

trials, labels = [], []
for f in nb:
    path = os.path.join(root, f)
    mat = scipy.io.loadmat(path)
    raw = mat['data'] if 'data' in mat else mat['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    onset = int(round(0.64 * 250))
    t = raw[:, onset:onset+250] if raw.shape[1] >= onset+250 else raw[:, :250]
    trials.append(t)
    for i, cmd in enumerate(['qianjin','houtui','zuozh','tingzhi','youzh']):
        if cmd[:2] in f: labels.append(i); break
    else:
        if '前进' in f: labels.append(0)
        elif '后退' in f: labels.append(1)
        elif '左转' in f: labels.append(2)
        elif '停止' in f: labels.append(3)
        elif '右转' in f: labels.append(4)
        else: labels.append(-1)

trials = np.array(trials); labels = np.array(labels)
print(f'每类: {dict(zip(*np.unique(labels, return_counts=True)))}')
for i in range(trials.shape[0]):
    trials[i] = trials[i] - np.mean(trials[i], axis=1, keepdims=True)
    ch_std = np.std(trials[i], axis=1, keepdims=True)
    trials[i] = np.clip(trials[i] / (ch_std + 1e-6), -6, 6)

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

freqs = [6.0, 6.667, 7.5, 8.571, 10.0]
cmds = ['前进','后退','左转','停止','右转']

for name, cls, need_fit in [('TDCA', TDCA, True), ('FBCCA', FBCCA, False), ('CCA', CCA, False)]:
    cm = np.zeros((5,5), dtype=int)
    n = len(nb); bs = 5
    for b in range(n // bs):
        ti = list(range(b*bs, (b+1)*bs))
        vi = [i for i in range(n) if i not in ti]
        Xtr, ytr = trials[vi], labels[vi]
        Xte, yte = trials[ti], labels[ti]
        m = cls(5, 1.0, freqs, Nh=8, sample_rate=250)
        if need_fit: m.fit(Xtr, ytr)
        for j in range(len(ti)):
            p = m.classify(Xte[j])
            p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
            cm[yte[j], p] += 1
    acc = 100 * np.sum(np.diag(cm)) / np.sum(cm)
    pcls = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
    print(f'{name}: {np.sum(np.diag(cm))}/{np.sum(cm)} = {acc:.1f}%  per: {pcls}')
    for i, cmd in enumerate(cmds):
        row = list(map(int, cm[i])); ok = row[i]; tot = sum(row)
        bar = '#'*ok + '-'*(tot-ok)
        print(f'  {cmd} ({freqs[i]:.1f}Hz): {bar} {ok}/{tot}')
