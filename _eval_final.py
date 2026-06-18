import numpy as np, scipy.io, os
from scipy import signal

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files_all = sorted(os.listdir(root))

# 最新 block: 22:10-22:12, 20 files
new_block = [f for f in files_all if '22' in f.split('_')[-1].replace('.mat','')[:2]]
new_block.sort()
print(f'新 block: {len(new_block)} 条')

# 先频谱验证
mat = scipy.io.loadmat(os.path.join(root, new_block[0]))
raw = mat['data'] if 'data' in mat else mat['analysis_data']
if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
ch_avg = np.mean(raw, axis=0)
f, Pxx = signal.welch(ch_avg, fs=250, nperseg=512)
mask = (f >= 5.5) & (f <= 16)
peaks = sorted(zip(Pxx[mask], f[mask]), reverse=True)[:6]
print(f'频谱峰值 ({new_block[0]}):')
for p, hz in peaks:
    print(f'  {hz:.2f} Hz  power={p:.0f}')

# 加载全部
trials, labels = [], []
for f in new_block:
    path = os.path.join(root, f)
    mat = scipy.io.loadmat(path)
    raw = mat['data'] if 'data' in mat else mat['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    onset = int(round(0.64 * 250))
    t = raw[:, onset:onset+250] if raw.shape[1] >= onset+250 else raw[:, :250]
    trials.append(t)
    for i, cmd in enumerate(['前进','后退','左转','停止','右转']):
        if cmd in f: labels.append(i); break

trials = np.array(trials); labels = np.array(labels)
for i in range(trials.shape[0]):
    trials[i] = trials[i] - np.mean(trials[i], axis=1, keepdims=True)
    ch_std = np.std(trials[i], axis=1, keepdims=True)
    trials[i] = np.clip(trials[i] / (ch_std + 1e-6), -6, 6)

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

# 新旧两套都试
old_f = [6.875, 7.5, 8.25, 10.3125, 13.75]
new_f = [6.667, 7.5, 8.571, 10.0, 12.0]

for tag, freqs in [('旧频', old_f), ('新频', new_f)]:
    n = len(new_block); bs = 5
    for name, cls, fit in [('TDCA', TDCA, True), ('FBCCA', FBCCA, False), ('CCA', CCA, False)]:
        cm = np.zeros((5,5), dtype=int)
        for b in range(n//bs):
            ti = list(range(b*bs, (b+1)*bs))
            vi = [i for i in range(n) if i not in ti]
            m = cls(5, 1.0, freqs, Nh=8, sample_rate=250)
            if fit: m.fit(trials[vi], labels[vi])
            for j in range(len(ti)):
                p = m.classify(trials[ti[j]])
                p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
                cm[labels[ti[j]], p] += 1
        acc = 100*np.sum(np.diag(cm))/max(np.sum(cm),1)
        pcls = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
        print(f'  [{tag}] {name}: {np.sum(np.diag(cm))}/{np.sum(cm)}={acc:.1f}% {pcls}')
        if name == 'FBCCA' and tag == '新频':
            print(f'    混淆矩阵:')
            for i, cmd in enumerate(['前进','后退','左转','停止','右转']):
                print(f'      {cmd}: {list(map(int, cm[i]))}')
