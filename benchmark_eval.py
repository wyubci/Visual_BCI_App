# -*- coding: utf-8 -*-
"""Benchmark evaluation: TDCA leave-one-block-out cross-validation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.io import loadmat
import time

from models.TDCA import TDCA

BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14

# Correct Tsinghua benchmark order: 5 groups of 8 freqs
TARGET_FREQS = []
for _offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for _base in range(8, 16):
        TARGET_FREQS.append(_base + _offset)
NUM_TARGETS = 40
NUM_HARMONICS = 3

DATA_LENGTH_SEC = 0.7
N_SUBJECTS_START = 6   # 从 S6 开始
N_SUBJECTS_END = 35    # 到 S35


def load_subject(subj_id):
    fpath = os.path.join(BENCHMARK_DIR, f"S{subj_id}.mat", f"S{subj_id}.mat")
    return np.array(loadmat(fpath)["data"], dtype=float)


def run_lobo_cv(data, data_length_sec):
    _, _, n_targets, n_blocks = data.shape
    total_times = data_length_sec + VISUAL_DELAY
    T = int(SAMPLE_RATE * data_length_sec)
    lag = max(3, int(round(SAMPLE_RATE * 0.02)))
    use_points = T + lag

    correct, total = 0, 0

    for test_block in range(n_blocks):
        train_mask = [b for b in range(n_blocks) if b != test_block]

        train_trials, train_labels = [], []
        for tgt in range(n_targets):
            for b in train_mask:
                train_trials.append(data[:, -use_points:, tgt, b])
                train_labels.append(tgt)

        test_trials, test_labels = [], []
        for tgt in range(n_targets):
            test_trials.append(data[:, -use_points:, tgt, test_block])
            test_labels.append(tgt)

        train_x = np.stack(train_trials, axis=0)
        train_y = np.array(train_labels, dtype=int)
        test_x = np.stack(test_trials, axis=0)
        test_y = np.array(test_labels, dtype=int)

        model = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE)
        try:
            model.fit(train_x, train_y)
        except Exception as e:
            print(f"    [ERROR] fit failed, block={test_block}: {e}", flush=True)
            continue

        for i in range(test_x.shape[0]):
            try:
                pred = model.classify(test_x[i])
            except Exception:
                pred = -1
            if pred == test_y[i]:
                correct += 1
            total += 1

    return correct / total * 100 if total > 0 else 0.0


def main():
    print("=" * 60, flush=True)
    print(f"TDCA Benchmark | S{N_SUBJECTS_START}-S{N_SUBJECTS_END} | {DATA_LENGTH_SEC}s data", flush=True)
    print(f"Targets: {NUM_TARGETS} ({TARGET_FREQS[0]}~{TARGET_FREQS[-1]} Hz)", flush=True)
    print("=" * 60, flush=True)

    accs = []
    t_start = time.time()

    for subj_id in range(N_SUBJECTS_START, N_SUBJECTS_END + 1):
        t0 = time.time()
        try:
            data = load_subject(subj_id)
        except Exception as e:
            print(f"S{subj_id:02d}: LOAD ERROR - {e}", flush=True)
            continue

        acc = run_lobo_cv(data, DATA_LENGTH_SEC)
        accs.append(acc)
        elapsed = time.time() - t0
        print(f"S{subj_id:02d}: {acc:.2f}%  ({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_start
    arr = np.array(accs)

    print("\n" + "=" * 60, flush=True)
    print(f"Results (N={len(accs)}, {DATA_LENGTH_SEC}s data):", flush=True)
    print(f"  Mean accuracy: {arr.mean():.2f}%", flush=True)
    print(f"  Std:           {arr.std():.2f}%", flush=True)
    print(f"  Max:           {arr.max():.2f}% (S{np.argmax(arr)+N_SUBJECTS_START})", flush=True)
    print(f"  Min:           {arr.min():.2f}% (S{np.argmin(arr)+N_SUBJECTS_START})", flush=True)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
