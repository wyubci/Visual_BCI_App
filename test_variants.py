# -*- coding: utf-8 -*-
"""Quick test of TDCA variants (Shrinkage, MultiLag) on S1-S3."""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import loadmat
from models.TDCA import TDCA, TDCA_SHRINK, MultiLagTDCA
from benchmark_worker import OCCIPITAL_CHANNELS, extract_trial

BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14
NUM_HARMONICS = 5
DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]

TARGET_FREQS = []
for offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for base in range(8, 16):
        TARGET_FREQS.append(base + offset)


def eval_tdca_variant(sid, model_class, model_kwargs, channels, name):
    fpath = os.path.join(BENCHMARK_DIR, f"S{sid}.mat", f"S{sid}.mat")
    data = np.asarray(loadmat(fpath)["data"], dtype=float)
    _, _, n_targets, n_blocks = data.shape
    results = {}

    for dl_sec in DATA_LENGTHS:
        total_times = dl_sec + VISUAL_DELAY
        temp = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                    sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY,
                    lagging_len=model_kwargs.get("lagging_len", 8))
        lag_len = temp.lagging_len
        req_pts = temp.required_points

        correct, total = 0, 0
        for test_block in range(n_blocks):
            train_blocks = [b for b in range(n_blocks) if b != test_block]
            train_x = np.stack([
                extract_trial(data, tgt, b, dl_sec, lag_len, channels)
                for tgt in range(n_targets) for b in train_blocks
            ], axis=0)
            train_y = np.array([tgt for tgt in range(n_targets) for b in train_blocks], dtype=int)

            if model_class == MultiLagTDCA:
                model = MultiLagTDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                                     n_components=model_kwargs.get("n_components", 3),
                                     sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
            else:
                model = model_class(NUM_HARMONICS, total_times, TARGET_FREQS,
                                    sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY,
                                    **model_kwargs)
            try:
                model.fit(train_x, train_y)
            except Exception as e:
                print(f"    {name} fit error: {e}", flush=True)
                continue

            for tgt in range(n_targets):
                trial = extract_trial(data, tgt, test_block, dl_sec, lag_len, channels)
                try:
                    pred = model.classify(trial)
                except Exception:
                    pred = -1
                if pred == tgt:
                    correct += 1
                total += 1

        results[dl_sec] = correct / total * 100.0 if total > 0 else 0.0
    return results


MODELS = [
    ("TDCA 基线(n_comp=3)", TDCA, {"lagging_len": 8, "n_components": 3}),
    ("TDCA_SHRINK", TDCA_SHRINK, {"lagging_len": 8, "n_components": 3}),
    ("MultiLagTDCA", MultiLagTDCA, {"n_components": 3}),
]

SUBJECTS = [1, 2, 3]
CHANNELS = OCCIPITAL_CHANNELS

print("=" * 70)
print(f"TDCA 变体快速测试 (S1-S3, {len(MODELS)} 变体)")
print("=" * 70)

all_means = {}
for name, model_cls, kwargs in MODELS:
    print(f"\n--- {name} ---")
    cfg_accs = {dl: [] for dl in DATA_LENGTHS}
    t0 = time.time()

    for sid in SUBJECTS:
        res = eval_tdca_variant(sid, model_cls, kwargs, CHANNELS, name)
        for dl, acc in res.items():
            cfg_accs[dl].append(acc)
        parts = " | ".join(f"{dl:.1f}s={res[dl]:.2f}%" for dl in DATA_LENGTHS)
        print(f"  S{sid:02d}: {parts}  [{time.time()-t0:.0f}s]")

    means = {dl: float(np.mean(cfg_accs[dl])) for dl in DATA_LENGTHS}
    all_means[name] = means
    print(f"  均值: " + " | ".join(f"{dl:.1f}s={means[dl]:.2f}%" for dl in DATA_LENGTHS))

# Comparison
print(f"\n{'='*70}")
print(f"变体对比 (S1-S3 均值):")
print(f"  {'变体':<25} {'0.3s':>10} {'0.5s':>10} {'0.7s':>10} {'1.0s':>10}")
print(f"  {'-'*65}")
for name, means in all_means.items():
    print(f"  {name:<25} {means[0.3]:>8.2f}% {means[0.5]:>8.2f}% {means[0.7]:>8.2f}% {means[1.0]:>8.2f}%")

print(f"\n完成!")
