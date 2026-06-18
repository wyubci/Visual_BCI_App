# -*- coding: utf-8 -*-
"""Evaluate collected data WITH 50Hz notch filter.
Reveals the true accuracy after fixing the power line noise issue.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.io import loadmat
from scipy import signal as sg
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

def notch_filter(data, fs=250, freq=50.0, q=30):
    """Apply notch filter to remove power line noise."""
    b, a = sg.iirnotch(freq, q, fs)
    return sg.filtfilt(b, a, data, axis=-1)

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
                ad = np.array(ad, dtype=float)
                if ad.ndim != 2 or ad.shape[0] < 3 or ad.shape[1] < 100: continue
                # Apply notch filter
                ad = notch_filter(ad)
                freqs = tuple(round(float(x), 3) for x in np.array(m["stim_freqs_hz"]).flatten())
                groups[freqs].append({"data": ad, "label": li})
            except: pass
    return sorted(groups.items(), key=lambda x: -len(x[1]))

def extract_window(analysis_data, dl_sec):
    T = int(dl_sec * SAMPLE_RATE)
    if T > analysis_data.shape[1]: return None
    return analysis_data[:, -T:]

def eval_ref(model_cls, items, dl_sec, freqs):
    total_times = dl_sec + 0.14
    model = model_cls(5, total_times, list(freqs), sample_rate=SAMPLE_RATE, delay_sec=0.14)
    correct = total = 0
    for it in items:
        trial = extract_window(it["data"], dl_sec)
        if trial is None: continue
        try: pred = model.classify(trial)
        except: pred = -1
        if pred == it["label"]: correct += 1
        total += 1
    return 100*correct/total if total else 0.0

def eval_tdca_cv(tdca_cls, items, dl_sec, freqs, n_comp=1, **extra):
    total_times = dl_sec + 0.14
    temp = tdca_cls(5, total_times, list(freqs), sample_rate=SAMPLE_RATE,
                    delay_sec=0.14, n_components=n_comp, **extra)
    lag = temp.lagging_len
    X_all, y_all = [], []
    for it in items:
        full_data = it["data"]
        T = int(dl_sec * SAMPLE_RATE)
        start = full_data.shape[1] - T - lag
        if start < 0: continue
        X_all.append(full_data[:, start:])
        y_all.append(it["label"])
    if len(X_all) < 10: return 0.0
    X_all = np.stack(X_all); y_all = np.array(y_all, dtype=int)
    min_class = min(Counter(y_all).values())
    n_splits = min(N_FOLDS, min_class)
    if n_splits < 2: return 0.0
    correct = total = 0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X_all, y_all):
        model = tdca_cls(5, total_times, list(freqs), sample_rate=SAMPLE_RATE,
                         delay_sec=0.14, n_components=n_comp, **extra)
        try: model.fit(X_all[train_idx], y_all[train_idx])
        except: continue
        for i in test_idx:
            try: pred = model.classify(X_all[i])
            except: pred = -1
            if pred == y_all[i]: correct += 1
            total += 1
    return 100*correct/total if total else 0.0

def main():
    groups = load_data()
    total = sum(len(v) for _, v in groups)
    p(f"数据: {total} 样本 (已加 50Hz 陷波滤波器)")
    p(f"频率组: {len(groups)} 个\n")

    for gidx, (freqs, items) in enumerate(groups, 1):
        n = len(items)
        if n < 20:
            p(f"组 #{gidx}: {n} 样本 — 跳过(太少)")
            continue
        dist = Counter(it["label"] for it in items)
        lb = lambda l: ['前进','后退','左转','停止','右转'][l]
        p(f"\n{'='*70}")
        p(f"组 #{gidx}: {n} 样本 | {list(freqs)}")
        p(f"分布: {', '.join(f'{lb(l)}:{dist[l]}' for l in range(5))}")
        p(f"{'='*70}")

        for dl in WINDOWS:
            t0 = time.time()
            valid = sum(1 for it in items if extract_window(it["data"], dl) is not None)
            if valid < n * 0.8:
                p(f"  {dl:.1f}s: 仅{valid}/{n} 跳过")
                continue

            a_cca = eval_ref(CCA, items, dl, freqs)
            a_fbcca = eval_ref(FBCCA, items, dl, freqs)
            t1 = time.time()
            a_online = eval_tdca_cv(TDCA, items, dl, freqs, n_comp=1)
            t2 = time.time()
            a_opt = eval_tdca_cv(TDCA, items, dl, freqs, n_comp=3)
            t3 = time.time()
            a_multi = eval_tdca_cv(MultiLagTDCA, items, dl, freqs, n_comp=3)
            t4 = time.time()

            p(f"  {dl:.1f}s: CCA={a_cca:5.1f}% | FBCCA={a_fbcca:5.1f}% | "
              f"TDCA_online={a_online:5.1f}% | TDCA_opt={a_opt:5.1f}% | "
              f"MultiLag={a_multi:5.1f}% [{time.time()-t0:.0f}s]")

    p(f"\n{'='*70}")
    p("结论: 50Hz 陷波后真实准确率如上。需在采集管线加入 50Hz 陷波。")

if __name__ == "__main__":
    main()
