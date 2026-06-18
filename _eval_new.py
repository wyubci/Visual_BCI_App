import numpy as np, scipy.io, os

root = 'saveCarData/hc33/train/2026-06-02/4.00s'
files_all = sorted(os.listdir(root))
# 取 21:32 之后的 3 个 block (21:32-21:43, 60 条)
new_blocks = [f for f in files_all if any(
    ts in f for ts in [
        '213204','213216','213221','213227','213232','213237','213243','213248','213253','213259',
        '213304','213310','213315','213321','213326','213338','213343','213349','213355','213400',
        '213924','213929','213935','213941','213946','213952','213958','214004','214009','214015',
        '214020','214026','214031','214037','214042','214048','214100','214106','214112','214118',
        '214135','214141','214151','214157','214203','214210','214222','214228','214234','214240',
        '214246','214257','214303','214315','214322','214328','214334','214339','214345','214351',
    ]
)]
print(f'新频率 blocks: {len(new_blocks)} 条')

trials, labels = [], []
for f in new_blocks:
    path = os.path.join(root, f)
    mat = scipy.io.loadmat(path)
    raw = mat['data'] if 'data' in mat else mat['analysis_data']
    if raw.ndim == 3: raw = raw.reshape(raw.shape[0], raw.shape[1])
    onset, length = int(round(0.64*250)), 250
    t = raw[:, onset:onset+length] if raw.shape[1] >= onset+length else raw[:, :length]
    trials.append(t)
    for i, cmd in enumerate(['前进','后退','左转','停止','右转']):
        if cmd in f: labels.append(i); break

trials = np.array(trials); labels = np.array(labels)
print(f'数据: {trials.shape}, 每类: {dict(zip(*np.unique(labels, return_counts=True)))}')

for i in range(trials.shape[0]):
    trials[i] = trials[i] - np.mean(trials[i], axis=1, keepdims=True)
    ch_std = np.std(trials[i], axis=1, keepdims=True)
    trials[i] = np.clip(trials[i] / (ch_std + 1e-6), -6, 6)

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

old_freqs = [6.875, 7.5, 8.25, 10.3125, 13.75]

ns, bs = len(new_blocks), 5
for name, cls, need_fit in [('TDCA', TDCA, True), ('FBCCA', FBCCA, False), ('CCA', CCA, False)]:
    cm = np.zeros((5,5), dtype=int)
    for b in range(ns // bs):
        ti = list(range(b*bs, (b+1)*bs))
        vi = [i for i in range(ns) if i not in ti]
        model = cls(5, 1.0, old_freqs, Nh=8, sample_rate=250)
        if need_fit: model.fit(trials[vi], labels[vi])
        for j in range(len(ti)):
            p = model.classify(trials[ti[j]])
            p = int(p) if not isinstance(p, (list, np.ndarray)) else int(p[0])
            cm[labels[ti[j]], p] += 1
    acc = 100*np.sum(np.diag(cm))/max(np.sum(cm),1)
    pcls = [f'{100*cm[i,i]/max(cm[i].sum(),1):.0f}%' for i in range(5)]
    print(f'{name}: {np.sum(np.diag(cm))}/{np.sum(cm)}={acc:.1f}%  per: {pcls}')
    cmds = ['前进','后退','左转','停止','右转']
    for i, cmd in enumerate(cmds):
        print(f'  {cmd}: {list(map(int, cm[i]))}')
