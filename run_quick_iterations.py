# -*- coding: utf-8 -*-
"""Quick TDCA iterations on S1-S5, then best on all subjects."""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from benchmark_worker import OCCIPITAL_CHANNELS, extract_trial
from scipy.io import loadmat
from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14
NUM_HARMONICS = 5
DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
TARGET_FREQS = []
for offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for base in range(8, 16):
        TARGET_FREQS.append(base + offset)

def eval_subject_tdca(sid, config, channels):
    """Evaluate TDCA only on one subject with given config."""
    fpath = os.path.join(BENCHMARK_DIR, f"S{sid}.mat", f"S{sid}.mat")
    data = np.asarray(loadmat(fpath)["data"], dtype=float)
    _, _, n_targets, n_blocks = data.shape
    results = {}

    for dl_sec in DATA_LENGTHS:
        total_times = dl_sec + VISUAL_DELAY
        temp = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                    sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY,
                    lagging_len=config.get("lag", 8),
                    n_components=config.get("n_comp", 1))
        lag_len, req_pts = temp.lagging_len, temp.required_points
        correct, total = 0, 0

        for test_block in range(n_blocks):
            train_blocks = [b for b in range(n_blocks) if b != test_block]
            train_x = np.stack([
                extract_trial(data, tgt, b, dl_sec, lag_len, channels)
                for tgt in range(n_targets) for b in train_blocks
            ], axis=0)
            train_y = np.array([tgt for tgt in range(n_targets) for b in train_blocks], dtype=int)

            model = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                         sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY,
                         lagging_len=config.get("lag", 8),
                         n_components=config.get("n_comp", 1))
            try:
                model.fit(train_x, train_y)
            except Exception:
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


CONFIGS = [
    {"name": "基线 lag=8 n_comp=1", "lag": 8, "n_comp": 1},
    {"name": "n_comp=2", "lag": 8, "n_comp": 2},
    {"name": "n_comp=3", "lag": 8, "n_comp": 3},
    {"name": "lag=4 (16ms)", "lag": 4, "n_comp": 1},
    {"name": "lag=12 (48ms)", "lag": 12, "n_comp": 1},
    {"name": "lag=4 n_comp=2", "lag": 4, "n_comp": 2},
    {"name": "lag=12 n_comp=2", "lag": 12, "n_comp": 2},
]

def main():
    test_subjects = [1, 2, 3, 4, 5]  # Quick test on S1-S5
    channels = OCCIPITAL_CHANNELS

    print("=" * 70)
    print(f"TDCA 迭代快速测试 (S1-S5, {len(CONFIGS)} 配置)")
    print("=" * 70)

    all_results = {}
    for cfg in CONFIGS:
        print(f"\n--- {cfg['name']} ---")
        cfg_accs = {dl: [] for dl in DATA_LENGTHS}
        t0 = time.time()

        for sid in test_subjects:
            res = eval_subject_tdca(sid, cfg, channels)
            for dl, acc in res.items():
                cfg_accs[dl].append(acc)
            parts = " | ".join(f"{dl:.1f}s={res[dl]:.2f}%" for dl in DATA_LENGTHS)
            print(f"  S{sid:02d}: {parts}  [{time.time()-t0:.0f}s]")

        means = {dl: np.mean(cfg_accs[dl]) for dl in DATA_LENGTHS}
        all_results[cfg["name"]] = {"means": means, "per_subject": cfg_accs}

        print(f"  均值: " + " | ".join(f"{dl:.1f}s={means[dl]:.2f}%" for dl in DATA_LENGTHS) +
              f"  [总耗时 {time.time()-t0:.0f}s]")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"配置对比 (S1-S5 均值):")
    print(f"  {'配置':<25} {'0.3s':>10} {'0.5s':>10} {'0.7s':>10} {'1.0s':>10}")
    print(f"  {'-'*65}")
    for cfg in CONFIGS:
        m = all_results[cfg["name"]]["means"]
        print(f"  {cfg['name']:<25} {m[0.3]:>8.2f}% {m[0.5]:>8.2f}% {m[0.7]:>8.2f}% {m[1.0]:>8.2f}%")

    # Find best at 0.5s
    best_05 = max(CONFIGS, key=lambda c: all_results[c["name"]]["means"][0.5])
    best_03 = max(CONFIGS, key=lambda c: all_results[c["name"]]["means"][0.3])
    best_10 = max(CONFIGS, key=lambda c: all_results[c["name"]]["means"][1.0])

    print(f"\n最佳 0.3s: {best_03['name']} ({all_results[best_03['name']]['means'][0.3]:.2f}%)")
    print(f"最佳 0.5s: {best_05['name']} ({all_results[best_05['name']]['means'][0.5]:.2f}%)")
    print(f"最佳 1.0s: {best_10['name']} ({all_results[best_10['name']]['means'][1.0]:.2f}%)")

    # Save
    out = {}
    for cfg in CONFIGS:
        name = cfg["name"].replace(" ", "_")
        out[name] = {str(dl): all_results[cfg["name"]]["means"][dl] for dl in DATA_LENGTHS}
    with open("benchmark_report/quick_iterations_s1s5.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
