import numpy as np, scipy.io, os

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files = sorted(os.listdir(root))
nb = [f for f in files if any(t in f for t in ['224','22500'])]

trials_raw, labels = [], []
for f in nb:
    m = scipy.io.loadmat(os.path.join(root, f))
    raw = m['data'] if 'data' in m else m['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    trials_raw.append(raw)
    if '前进' in f: labels.append(0)
    elif '后退' in f: labels.append(1)
    elif '左转' in f: labels.append(2)
    elif '停止' in f: labels.append(3)
    elif '右转' in f: labels.append(4)
trials_raw = trials_raw; labels = np.array(labels)
n = len(nb)

def prep(X, onset, length):
    out = []
    for t in X:
        t = np.asarray(t, dtype=np.float64)
        w = t[:, onset:onset+length] if t.shape[1] >= onset+length else t[:, :length]
        w = w - np.mean(w, axis=1, keepdims=True)
        ch_std = np.std(w, axis=1, keepdims=True)
        w = np.clip(w / (ch_std + 1e-6), -6, 6)
        out.append(np.asarray(w, dtype=np.float64))
    return np.array(out)

from models.TDCA import TDCA
from models.FBCCA import FBCCA
freqs = [6.0, 6.667, 7.5, 8.571, 10.0]

print('=== 各自在对应时长上训练/测试, LOBO bs=5 ===')
for stim_sec in [1.0, 1.5, 2.0, 3.0, 4.0]:
    delay_samp = int(0.14 * 250)
    total_samp = int(stim_sec * 250)
    eff_samp = total_samp - delay_samp
    if eff_samp < 50: continue

    X = prep(trials_raw, delay_samp, eff_samp)

    fbcca_cm = np.zeros((5,5), dtype=int)
    tdca_cm = np.zeros((5,5), dtype=int)
    bs = 5
    for b in range(n // bs):
        ti = list(range(b*bs, (b+1)*bs))
        vi = [i for i in range(n) if i not in ti]

        fbcca = FBCCA(5, stim_sec, freqs, Nh=8, sample_rate=250, delay_sec=0.14)
        for j in ti:
            p = fbcca.classify(X[j])
            p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
            fbcca_cm[labels[j], p] += 1

        tdca = TDCA(5, stim_sec, freqs, Nh=8, sample_rate=250, delay_sec=0.14)
        tdca.fit(X[vi], labels[vi])
        for j in ti:
            p = tdca.classify(X[j])
            p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
            tdca_cm[labels[j], p] += 1

    fc = 100 * np.sum(np.diag(fbcca_cm)) / np.sum(fbcca_cm)
    tc = 100 * np.sum(np.diag(tdca_cm)) / np.sum(tdca_cm)
    arrow = 'TDCA>' if tc > fc else ('FBCCA>' if fc > tc else '=')
    print(f'  {stim_sec}s: FBCCA={fc:.0f}%  TDCA={tc:.0f}%  {arrow}')
