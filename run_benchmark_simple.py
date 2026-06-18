# -*- coding: utf-8 -*-
"""Sequential benchmark runner — avoids Windows spawn issues."""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from collections import OrderedDict
from benchmark_worker import evaluate_subject, OCCIPITAL_CHANNELS

DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
MODEL_TYPES = ["TDCA", "FBCCA", "CCA"]
N_SUBJECTS = 35
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")
os.makedirs(REPORT_DIR, exist_ok=True)

def compute_itr(acc, n_targets=40, data_length_sec=1.0, gap_sec=0.5):
    N = n_targets
    P = max(min(acc / 100.0, 0.999), 1.0 / N)
    T = data_length_sec + gap_sec
    if P >= 0.999:
        return N * np.log2(N) * 60.0 / T
    if P <= 1.0 / N:
        return 0.0
    return max(0.0, (np.log2(N) + P * np.log2(P) +
                     (1 - P) * np.log2((1 - P) / (N - 1))) * 60.0 / T)

def main():
    subjects = list(range(1, N_SUBJECTS + 1))

    print("=" * 80, flush=True)
    print("BENCHMARK: Sequential (S1-S35, 9 occipital channels)", flush=True)
    print(f"Data: {DATA_LENGTHS}s | Targets: 40 | Models: {MODEL_TYPES}", flush=True)
    print("=" * 80, flush=True)

    all_accs = OrderedDict()
    for mt in MODEL_TYPES:
        for dl in DATA_LENGTHS:
            all_accs[(mt, dl)] = []

    t_start = time.time()

    for idx, sid in enumerate(subjects):
        t0 = time.time()
        result = evaluate_subject((sid, DATA_LENGTHS, MODEL_TYPES, OCCIPITAL_CHANNELS))

        if result["error"]:
            print(f"[{idx+1:2d}/35] S{sid:02d}: {result['error']}", flush=True)
        else:
            parts = []
            for (mt, dl), acc in sorted(result["results"].items()):
                all_accs[(mt, dl)].append(acc)
                parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
            et = time.time() - t_start
            print(f"[{idx+1:2d}/35] S{sid:02d}: " + " | ".join(parts) +
                  f"  [{time.time()-t0:.0f}s] (total {et:.0f}s)", flush=True)

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: Mean accuracy across S1-S35", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<8} {'0.3s':>12} {'0.5s':>12} {'0.7s':>12} {'1.0s':>12}", flush=True)
    print("-" * 56, flush=True)

    summary = {}
    for mt in MODEL_TYPES:
        row = f"{mt:<8}"
        for dl in DATA_LENGTHS:
            accs = all_accs[(mt, dl)]
            if accs:
                m = float(np.mean(accs))
                s = float(np.std(accs))
                itr = compute_itr(m, data_length_sec=dl)
                summary[(mt, dl)] = {"mean": m, "std": s, "n": len(accs),
                                     "min": float(min(accs)), "max": float(max(accs)),
                                     "itr": itr, "accs": [float(a) for a in accs]}
                row += f" {m:8.2f}%±{s:.1f}"
            else:
                summary[(mt, dl)] = {"mean": 0, "std": 0, "n": 0, "min": 0, "max": 0, "itr": 0, "accs": []}
                row += f" {'--':>12}"
        print(row, flush=True)

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Detailed stats
    print(f"\n{'='*80}", flush=True)
    for mt in MODEL_TYPES:
        for dl in DATA_LENGTHS:
            info = summary[(mt, dl)]
            if info["n"] > 0:
                print(f"{mt} @ {dl:.1f}s: mean={info['mean']:.2f}% ±{info['std']:.2f}%  "
                      f"min={info['min']:.2f}%  max={info['max']:.2f}%  "
                      f"ITR={info['itr']:.1f} bits/min  n={info['n']}", flush=True)

    # Save
    out = {}
    for (mt, dl), info in summary.items():
        out[f"{mt}_{dl}s"] = {k: v for k, v in info.items() if k != "accs"}
    with open(os.path.join(REPORT_DIR, "baseline_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to: {os.path.join(REPORT_DIR, 'baseline_summary.json')}", flush=True)
    return summary


if __name__ == "__main__":
    main()
