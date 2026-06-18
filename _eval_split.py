import numpy as np, scipy.io, os

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files_all = sorted(os.listdir(root))
# 跳孤立
files_all = [f for f in files_all if '213110' not in f]

# Block 1-3 (60条) 和 Block 4 (20条，21:32-21:34)
b4 = [f for f in files_all if any(t in f for t in ['213204','213216','213221','213227','213232','213237','213243','213248','213253','213259','213304','213310','213315','213321','213326','213338','213343','213349','213355','213400'])]
b123 = [f for f in files_all if f not in b4]
print(f'Block 1-3: {len(b123)} 条  Block 4: {len(b4)} 条')

def load_trials(files):
    trials, labels = [], []
    for f in files:
        path = os.path.join(root, f)
        mat = scipy.io.loadmat(path)
        raw = mat['data'] if 'data' in mat else mat['analysis_data']
        if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
        onset, length = int(round(0.64*250)), 250
        t = raw[:, onset:onset+length] if raw.shape[1] >= onset+length else raw[:, :length]
        trials.append(t)
        for i, cmd in enumerate(['前进','后退','左转','停止','右转']):
            if cmd in f: labels.append(i); break
    tr = np.array(trials); lb = np.array(labels)
    for i in range(tr.shape[0]):
        tr[i] = tr[i] - np.mean(tr[i], axis=1, keepdims=True)
        ch_std = np.std(tr[i], axis=1, keepdims=True)
        tr[i] = np.clip(tr[i] / (ch_std + 1e-6), -6, 6)
    return tr, lb

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

old_freqs = [6.875, 7.5, 8.25, 10.3125, 13.75]
new_freqs = [8.0, 9.4, 11.0, 12.8, 15.0]

def eval_set(trials, labels, freqs, tag):
    n = len(labels); bs = 5
    for name, cls, need_fit in [('TDCA', TDCA, True), ('FBCCA', FBCCA, False), ('CCA', CCA, False)]:
        cm = np.zeros((5,5), dtype=int)
        for b in range(n//bs):
            ti = list(range(b*bs, (b+1)*bs))
            vi = [i for i in range(n) if i not in ti]
            Xtr, ytr, Xte, yte = trials[vi], labels[vi], trials[ti], labels[ti]
            model = cls(5, 1.0, freqs, Nh=8, sample_rate=250)
            if need_fit: model.fit(Xtr, ytr)
            for j in range(Xte.shape[0]):
                p = model.classify(Xte[j])
                p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
                cm[yte[j], p] += 1
        acc = 100*np.sum(np.diag(cm))/max(np.sum(cm),1)
        pcls = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
        print(f'  [{tag}] {name}: {np.sum(np.diag(cm))}/{np.sum(cm)}={acc:.1f}% {pcls}')

# Block 1-3 用旧频率
b123_t, b123_l = load_trials(b123)
print(f'\n=== Block 1-3 ({len(b123)}条, 旧频率 {old_freqs}) ===')
eval_set(b123_t, b123_l, old_freqs, 'old')

# Block 4 试两套频率
b4_t, b4_l = load_trials(b4)
print(f'\n=== Block 4 ({len(b4)}条, 试旧频率 {old_freqs}) ===')
eval_set(b4_t, b4_l, old_freqs, 'old')
print(f'\n=== Block 4 ({len(b4)}条, 试新频率 {new_freqs}) ===')
eval_set(b4_t, b4_l, new_freqs, 'new')
