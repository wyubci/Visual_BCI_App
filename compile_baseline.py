# -*- coding: utf-8 -*-
"""Compile baseline results from all collected data."""
import json, os, numpy as np

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")

# All per-subject TDCA results collected from benchmark output
# Format: SXX: [0.3s, 0.5s, 0.7s, 1.0s]
TDCA_DATA = {
    "S01": [66.25, 85.00, 91.25, 95.83],
    "S02": [50.00, 69.17, 80.83, 89.58],
    "S03": [93.75, 96.67, 98.75, 99.17],
    "S04": [85.00, 89.17, 95.42, 96.67],
    "S05": [45.00, 76.25, 87.92, 96.67],
    "S06": [57.50, 80.00, 90.42, 97.08],
    "S07": [35.00, 56.25, 69.17, 85.42],
    "S08": [46.25, 57.92, 78.75, 92.50],
    "S09": [50.00, 58.33, 76.25, 89.58],
    "S10": [40.83, 54.58, 70.83, 85.42],
    "S11": [14.58, 20.42, 38.33, 58.75],
    "S12": [37.92, 60.83, 84.58, 91.25],
    "S13": [60.83, 82.50, 91.25, 97.50],
    "S16": [35.42, 55.00, 75.00, 87.08],
    "S17": [52.08, 77.92, 85.00, 90.83],
    "S18": [69.58, 80.00, 87.08, 92.08],
    "S19": [17.50, 30.42, 47.08, 74.58],
    "S20": [59.17, 79.17, 95.00, 99.17],
    "S21": [50.83, 76.25, 92.50, 99.17],
    "S26": [57.50, 73.33, 92.08, 95.42],
    "S27": [66.25, 82.92, 94.58, 98.75],
    "S28": [65.42, 82.08, 92.50, 97.08],
    "S29": [36.25, 52.50, 67.92, 82.08],
    "S30": [51.67, 71.67, 86.25, 94.58],
    "S31": [88.33, 95.42, 97.08, 98.33],
    "S32": [90.42, 95.83, 100.00, 100.00],
}

FBCCA_DATA = {
    "S01": [3.75, 17.50, 27.50, 49.58],
    "S02": [5.42, 9.58, 23.75, 45.00],
    "S03": [8.75, 40.00, 75.00, 95.00],
    "S04": [6.25, 29.17, 58.33, 83.75],
    "S05": [4.58, 10.83, 20.00, 51.25],
    "S06": [2.50, 12.50, 30.00, 65.42],
    "S07": [4.58, 11.67, 17.50, 40.42],
    "S08": [1.67, 6.25, 10.42, 30.83],
    "S09": [3.75, 12.08, 28.33, 53.33],
    "S10": [5.00, 10.83, 22.92, 44.58],
    "S11": [3.33, 2.50, 5.00, 6.67],
    "S12": [5.00, 8.75, 22.92, 63.33],
    "S13": [5.00, 14.17, 30.00, 55.42],
    "S16": [2.08, 4.58, 10.83, 17.08],
    "S17": [6.25, 6.25, 13.75, 22.92],
    "S18": [10.00, 22.50, 39.17, 60.00],
    "S19": [1.67, 2.08, 6.67, 22.08],
    "S20": [4.17, 12.92, 36.67, 73.33],
    "S21": [2.92, 11.25, 16.67, 45.00],
    "S26": [3.33, 14.17, 37.50, 73.33],
    "S27": [4.17, 10.42, 24.58, 57.92],
    "S28": [5.42, 13.33, 33.75, 62.92],
    "S29": [5.83, 11.25, 15.83, 34.58],
    "S30": [7.08, 7.50, 15.42, 37.50],
    "S31": [5.83, 22.50, 45.83, 74.58],
    "S32": [6.25, 35.42, 69.17, 94.58],
}

CCA_DATA = {
    "S01": [2.50, 7.92, 15.00, 22.08],
    "S02": [3.33, 4.17, 3.33, 12.08],
    "S03": [6.25, 23.75, 41.67, 78.33],
    "S04": [5.42, 22.92, 53.75, 84.17],
    "S05": [3.33, 5.00, 10.00, 20.00],
    "S06": [3.75, 10.00, 26.67, 52.92],
    "S07": [3.33, 8.33, 8.33, 17.50],
    "S08": [2.50, 5.00, 7.08, 18.33],
    "S09": [6.67, 11.25, 19.58, 37.50],
    "S10": [5.00, 7.50, 12.08, 20.42],
    "S11": [2.08, 5.00, 8.75, 10.00],
    "S12": [4.58, 11.67, 29.58, 68.75],
    "S13": [5.00, 12.50, 25.42, 49.17],
    "S16": [2.92, 2.50, 5.42, 6.67],
    "S17": [2.08, 3.75, 4.58, 11.67],
    "S18": [6.67, 17.08, 20.42, 38.33],
    "S19": [5.42, 4.17, 5.42, 16.67],
    "S20": [3.33, 6.67, 20.42, 42.08],
    "S21": [2.50, 9.17, 11.67, 27.50],
    "S26": [6.67, 22.92, 42.50, 72.08],
    "S27": [2.08, 5.42, 10.83, 33.33],
    "S28": [4.17, 6.67, 16.67, 29.17],
    "S29": [3.75, 9.17, 8.75, 14.17],
    "S30": [2.92, 4.58, 14.17, 35.83],
    "S31": [5.00, 17.08, 28.75, 45.42],
    "S32": [7.50, 20.00, 50.42, 80.42],
}


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


DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]

def summarize(data_dict, name):
    n = len(data_dict)
    print(f"\n{name} (N={n}):")
    print(f"  {'数据长度':<10} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10} {'ITR':>10}")
    print(f"  {'-'*60}")
    for i, dl in enumerate(DATA_LENGTHS):
        vals = [v[i] for v in data_dict.values()]
        m, s = np.mean(vals), np.std(vals)
        itr = compute_itr(m, data_length_sec=dl)
        print(f"  {f'{dl:.1f}s':<10} {m:>8.2f}% {s:>8.2f}% {min(vals):>8.2f}% {max(vals):>8.2f}% {itr:>8.1f}")
    return {dl: {"mean": float(np.mean([v[i] for v in data_dict.values()])),
                 "std": float(np.std([v[i] for v in data_dict.values()])),
                 "n": n} for i, dl in enumerate(DATA_LENGTHS)}

if __name__ == "__main__":
    print("=" * 70)
    print(f"基线 Benchmark 结果汇总 (N={len(TDCA_DATA)} 被试)")
    print("=" * 70)

    tdca_summary = summarize(TDCA_DATA, "TDCA")
    fbcca_summary = summarize(FBCCA_DATA, "FBCCA")
    cca_summary = summarize(CCA_DATA, "CCA")

    # Print comparison table
    print(f"\n{'='*70}")
    print("算法对比表:")
    print(f"  {'数据长度':<10} {'CCA':>10} {'FBCCA':>10} {'TDCA':>10}")
    print(f"  {'-'*42}")
    for dl in DATA_LENGTHS:
        cca_m = cca_summary[dl]["mean"]
        fbcca_m = fbcca_summary[dl]["mean"]
        tdca_m = tdca_summary[dl]["mean"]
        print(f"  {f'{dl:.1f}s':<10} {cca_m:>8.2f}% {fbcca_m:>8.2f}% {tdca_m:>8.2f}%")

    # Save
    output = {
        "n_subjects": len(TDCA_DATA),
        "subjects": sorted(TDCA_DATA.keys()),
        "TDCA": {f"{dl}s": tdca_summary[dl] for dl in DATA_LENGTHS},
        "FBCCA": {f"{dl}s": fbcca_summary[dl] for dl in DATA_LENGTHS},
        "CCA": {f"{dl}s": cca_summary[dl] for dl in DATA_LENGTHS},
        "per_subject": {
            s: {"TDCA": TDCA_DATA[s], "FBCCA": FBCCA_DATA.get(s, []), "CCA": CCA_DATA.get(s, [])}
            for s in TDCA_DATA
        }
    }
    with open(os.path.join(REPORT_DIR, "compiled_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {os.path.join(REPORT_DIR, 'compiled_baseline.json')}")
