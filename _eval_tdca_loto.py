import numpy as np, scipy.io, os

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files = sorted(os.listdir(root))
# 频率正确的那两 block: 224xxx + 22500x
nb = [f for f in files if any(t in f for t in ['224','22500'])]
print(f'{len(nb)} trials')

trials, labels = [], []
for f in nb:
    m = scipy.io.loadmat(os.path.join(root, f))
    raw = m['data'] if 'data' in m else m['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    onset = int(round(0.64 * 250))
    t = raw[:, onset:onset+250] if raw.shape[1] >= onset+250 else raw[:, :250]
    trials.append(t)
    if '前进' in f: labels.append(0)
    elif '后退' in f: labels.append(1)
    elif '左转' in f: labels.append(2)
    elif '停止' in f: labels.append(3)
    elif '右转' in f: labels.append(4)

trials = np.array(trials); labels = np.array(labels)
n = len(nb)
cnt = dict(zip(*np.unique(labels, return_counts=True)))
print(f'per class: {cnt}')

for i in range(n):
    trials[i] = trials[i] - np.mean(trials[i], axis=1, keepdims=True)
    ch_std = np.std(trials[i], axis=1, keepdims=True)
    trials[i] = np.clip(trials[i] / (ch_std + 1e-6), -6, 6)

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

freqs = [6.0, 6.667, 7.5, 8.571, 10.0]

# ====== LOTO (Leave-One-Trial-Out) ======
print('\n=== LOTO (每 fold 训练39条，测试1条) ===')
for name, cls, need_fit in [('FBCCA', FBCCA, False), ('CCA', CCA, False), ('TDCA', TDCA, True)]:
    cm = np.zeros((5,5), dtype=int)
    for i in range(n):
        vi = [j for j in range(n) if j != i]
        Xtr, ytr = trials[vi], labels[vi]
        Xte, yte = trials[i:i+1], labels[i:i+1]
        m = cls(5, 1.0, freqs, Nh=8, sample_rate=250)
        if need_fit:
            m.fit(Xtr, ytr)
        p = m.classify(Xte[0])
        p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
        cm[yte[0], p] += 1
    acc = 100 * np.sum(np.diag(cm)) / np.sum(cm)
    pcls = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
    print(f'{name}: {np.sum(np.diag(cm))}/{np.sum(cm)}={acc:.0f}%  per:{pcls}')
    for i, cmd in enumerate(['前','后','左','停','右']):
        r = list(map(int, cm[i])); ok = r[i]; tot = sum(r)
        print(f'  {cmd}:{r} ({ok}/{tot})')

# ====== LOBO 对比 ======
print('\n=== LOBO (bs=5, 每fold训练~15条) ===')
for name, cls, need_fit in [('FBCCA', FBCCA, False), ('CCA', CCA, False), ('TDCA', TDCA, True)]:
    cm = np.zeros((5,5), dtype=int)
    bs = 5
    for b in range(n // bs):
        ti = list(range(b*bs, (b+1)*bs))
        vi = [i for i in range(n) if i not in ti]
        Xtr, ytr = trials[vi], labels[vi]
        Xte, yte = trials[ti], labels[ti]
        m = cls(5, 1.0, freqs, Nh=8, sample_rate=250)
        if need_fit:
            m.fit(Xtr, ytr)
        for j in range(len(ti)):
            p = m.classify(Xte[j])
            p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
            cm[yte[j], p] += 1
    acc = 100 * np.sum(np.diag(cm)) / np.sum(cm)
    pcls = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
    print(f'{name}: {np.sum(np.diag(cm))}/{np.sum(cm)}={acc:.0f}%  per:{pcls}')
    for i, cmd in enumerate(['前','后','左','停','右']):
        r = list(map(int, cm[i])); ok = r[i]; tot = sum(r)
        print(f'  {cmd}:{r} ({ok}/{tot})')
