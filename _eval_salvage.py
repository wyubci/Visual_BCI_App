# -*- coding: utf-8 -*-
"""Re-evaluate saved car data with matching comb notch (from deviceControl_window)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.io import loadmat
from scipy import signal
from glob import glob
from collections import Counter, defaultdict
import time
from sklearn.model_selection import StratifiedKFold

from models.FBCCA import FBCCA
from models.CCA import CCA
from models.TDCA import TDCA, MultiLagTDCA

SAMPLE_RATE = 250
WINDOWS = [0.5, 1.0, 2.0, 4.0]
N_FOLDS = 5

def p(msg): print(msg, flush=True)

# Exact same filters as deviceControl_window.py
FS = 250
bpB, bpA = signal.iircomb(50, 35, ftype='notch', fs=FS)       # 50Hz comb notch
bpB2, bpA2 = signal.butter(5, [4, 90], 'bandpass', fs=FS)      # 4-90Hz bandpass

def preprocess(data):
    """Apply exactly the same filters as the acquisition pipeline."""
    d = np.asarray(data, dtype=float)
    d = signal.filtfilt(bpB, bpA, d, axis=-1)    # notch
    d = signal.filtfilt(bpB2, bpA2, d, axis=-1)  # bandpass
    return d

def load_data():
    dirs = ["saveCarData/hc33/train/2026-05-29/4.00s",
            "saveCarData/hc33/train/2026-05-30/4.00s"]
    groups = defaultdict(list)
    for d in dirs:
        for fp in sorted(glob(os.path.join(d, "*.mat"))):
            try:
                m = loadmat(fp)
                li = int(np.array(m["label_idx"]).reshape(-1)[0])
                if li < 0 or li >= 5: continue
                ad = m.get("analysis_data")
                if ad is None or np.array(ad).size == 0: continue
                ad = preprocess(np.array(ad, dtype=float))
                freqs = tuple(round(float(x), 3) for x in np.array(m["stim_freqs_hz"]).flatten())
                groups[freqs].append({"data": ad, "label": li})
            except: pass
    return sorted(groups.items(), key=lambda x: -len(x[1]))

def extract_window(data, dl_sec):
    T = int(dl_sec * SAMPLE_RATE)
    return data[:, -T:] if T <= data.shape[1] else None

def eval_ref(cls, items, dl_sec, freqs):
    tt = dl_sec + 0.14
    m = cls(5, tt, list(freqs), sample_rate=SAMPLE_RATE, delay_sec=0.14)
    c = t = 0
    for it in items:
        trial = extract_window(it["data"], dl_sec)
        if trial is None: continue
        try: p = m.classify(trial)
        except: p = -1
        if p == it["label"]: c += 1
        t += 1
    return 100*c/t if t else 0.0

def eval_tdca_cv(cls, items, dl_sec, freqs, n_comp=1, **kw):
    tt = dl_sec + 0.14
    tmp = cls(5, tt, list(freqs), sample_rate=SAMPLE_RATE, delay_sec=0.14, n_components=n_comp, **kw)
    lag = tmp.lagging_len
    X, Y = [], []
    for it in items:
        T = int(dl_sec * SAMPLE_RATE)
        s = it["data"].shape[1] - T - lag
        if s < 0: continue
        X.append(it["data"][:, s:]); Y.append(it["label"])
    if len(X) < 10: return 0.0
    X, Y = np.stack(X), np.array(Y, dtype=int)
    mc = min(Counter(Y).values())
    ns = min(N_FOLDS, mc)
    if ns < 2: return 0.0
    c = t = 0
    for ti, vi in StratifiedKFold(ns, shuffle=True, random_state=42).split(X, Y):
        m = cls(5, tt, list(freqs), sample_rate=SAMPLE_RATE, delay_sec=0.14, n_components=n_comp, **kw)
        try: m.fit(X[ti], Y[ti])
        except: continue
        for i in vi:
            try: pred = m.classify(X[i])
            except: pred = -1
            if pred == Y[i]: c += 1
            t += 1
    return 100*c/t if t else 0.0

def main():
    groups = load_data()
    total = sum(len(v) for _, v in groups)
    p(f"数据: {total} 样本 (comb notch 50Hz + bandpass 4-90Hz)")
    p(f"滤波器: iircomb(50,35,notch) + butter(5,[4,90],bandpass)")
    p(f"= 与 deviceControl_window.py 完全一致的滤波链\n")

    for gidx, (freqs, items) in enumerate(groups, 1):
        n = len(items)
        if n < 20: continue
        dist = Counter(it["label"] for it in items)
        lb = lambda l: ['前进','后退','左转','停止','右转'][l]
        p(f"{'='*70}")
        p(f"组 #{gidx}: {n} 样本 | {[round(x,1) for x in freqs]}")
        p(f"分布: {', '.join(f'{lb(l)}:{dist[l]}' for l in range(5))}")
        p(f"{'='*70}")

        for dl in WINDOWS:
            t0 = time.time()
            valid = sum(1 for it in items if extract_window(it["data"], dl) is not None)
            if valid < n * 0.8: continue

            a_cca   = eval_ref(CCA, items, dl, freqs)
            a_fbcca = eval_ref(FBCCA, items, dl, freqs)
            a_online = eval_tdca_cv(TDCA, items, dl, freqs, n_comp=1)
            a_opt    = eval_tdca_cv(TDCA, items, dl, freqs, n_comp=3)
            a_multi  = eval_tdca_cv(MultiLagTDCA, items, dl, freqs, n_comp=3)

            delta_opt = a_opt - a_online
            p(f"  {dl:.1f}s: CCA={a_cca:5.1f}% | FBCCA={a_fbcca:5.1f}% | "
              f"TDCA_on={a_online:5.1f}% | TDCA_opt={a_opt:5.1f}%({delta_opt:+.0f}) | "
              f"MultiLag={a_multi:5.1f}% [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
