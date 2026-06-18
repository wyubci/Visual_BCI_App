import numpy as np, scipy.io, os

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files = sorted(os.listdir(root))
nb = [f for f in files if any(t in f for t in ['224','22500'])]
print(f'{len(nb)} trials')

trials, labels = [], []
for f in nb:
    m = scipy.io.loadmat(os.path.join(root, f))
    raw = m['data'] if 'data' in m else m['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    trials.append(raw)
    if '前进' in f: labels.append(0)
    elif '后退' in f: labels.append(1)
    elif '左转' in f: labels.append(2)
    elif '停止' in f: labels.append(3)
    elif '右转' in f: labels.append(4)
trials = np.array(trials); labels = np.array(labels)

from models.FBCCA import FBCCA
freqs = [6.0, 6.667, 7.5, 8.571, 10.0]

# 不同窗口的真实在线场景
# 在线 trial: cue(0.5s) → stim(t秒) — 用户只有 t 秒看闪烁
# 分析从 visual_delay(0.14s) 后开始取 t-0.14 秒有效数据

print('\n真实在线准确率 (不同刺激时长, 全 41 条, LOBO bs=5):')
for stim_sec in [1.0, 1.5, 2.0, 3.0, 4.0]:
    delay = int(0.14 * 250)  # 35 samples visual delay
    win = int(stim_sec * 250)  # total window including delay
    eff_win = win - delay  # effective SSVEP window

    X, y = [], []
    for i in range(len(nb)):
        t = trials[i][:, delay:win] if trials[i].shape[1] >= win else trials[i][:, :]
        # 只做最简单的标准化
        t = t - np.mean(t, axis=1, keepdims=True)
        ch_std = np.std(t, axis=1, keepdims=True)
        t = np.clip(t / (ch_std + 1e-6), -6, 6)
        X.append(t)
        y.append(labels[i])
    X = np.array(X); y = np.array(y)

    cm = np.zeros((5,5), dtype=int)
    n = len(nb); bs = 5
    for b in range(n // bs):
        ti = list(range(b*bs, (b+1)*bs))
        vi = [i for i in range(n) if i not in ti]
        m = FBCCA(5, stim_sec, freqs, Nh=8, sample_rate=250, delay_sec=0.14)
        for j in range(len(ti)):
            p = m.classify(X[ti[j]])
            p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
            cm[y[ti[j]], p] += 1
    acc = 100 * np.sum(np.diag(cm)) / np.sum(cm)
    trial_time = stim_sec + 0.5 + 0.5  # stim + cue + rest
    vote3_time = trial_time * 3
    print(f'  {stim_sec}s刺激 (有效{eff_win/250:.2f}s): FBCCA={acc:.0f}%  '
          f'单trial={trial_time:.1f}s  3票={vote3_time:.0f}s')
