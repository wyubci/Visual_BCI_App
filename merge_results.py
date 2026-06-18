# -*- coding: utf-8 -*-
"""合并所有批次 Benchmark 结果，生成完整汇总。"""
import sys, os, json, glob
import numpy as np

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")

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
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "results_*.json")))
    if not files:
        print("未找到结果文件!", flush=True)
        return

    DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
    MODEL_TYPES = ["TDCA", "FBCCA", "CCA"]

    # 汇总所有被试
    all_per_subject = {}
    all_accs = {f"{mt}_{dl}s": [] for mt in MODEL_TYPES for dl in DATA_LENGTHS}

    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
        for subj, results in data.items():
            if isinstance(results, dict) and "error" not in results:
                all_per_subject[subj] = results
                for key, acc in results.items():
                    all_accs[key].append(acc)

    n_subjects = len(all_per_subject)
    print(f"合并完成: {n_subjects} 个被试, 来自 {len(files)} 个文件")
    print()

    # 计算均值/标准差
    print("=" * 80)
    print(f"全量 Benchmark 结果 (S1-S{max(int(k[1:]) for k in all_per_subject)}, N={n_subjects})")
    print("=" * 80)
    print(f"{'算法':<8} {'0.3s':>14} {'0.5s':>14} {'0.7s':>14} {'1.0s':>14}  {'ITR@1.0s':>12}")
    print("-" * 76)

    final = {}
    for mt in MODEL_TYPES:
        row = f"{mt:<8}"
        for dl in DATA_LENGTHS:
            key = f"{mt}_{dl}s"
            accs = all_accs[key]
            if accs:
                m, s = np.mean(accs), np.std(accs)
                row += f" {m:8.2f}%±{s:.1f}"
                final[key] = {"mean": float(m), "std": float(s),
                              "n": len(accs), "min": float(min(accs)),
                              "max": float(max(accs))}
            else:
                row += f" {'--':>14}"
        itr = compute_itr(final.get(f"{mt}_1.0s", {}).get("mean", 0), data_length_sec=1.0)
        row += f"  {itr:10.1f}"
        print(row)

    print()
    print("=" * 80)
    for key, info in sorted(final.items()):
        print(f"{key}: {info['mean']:.2f}% ±{info['std']:.2f}% "
              f"(min={info['min']:.2f}%, max={info['max']:.2f}%, n={info['n']})")

    # 找出最佳和最差被试 (TDCA @ 0.5s)
    tdca_05 = [(subj, results.get("TDCA_0.5s", 0)) for subj, results in all_per_subject.items()]
    tdca_05.sort(key=lambda x: -x[1])
    print()
    print("TDCA @ 0.5s 排名 (Top 5 & Bottom 5):")
    for rank, (subj, acc) in enumerate(tdca_05[:5]):
        print(f"  {rank+1}. {subj}: {acc:.2f}%")
    print("  ...")
    for rank, (subj, acc) in enumerate(tdca_05[-5:]):
        print(f"  {len(tdca_05)-4+rank}. {subj}: {acc:.2f}%")

    # 保存合并结果
    summary = {
        "n_subjects": n_subjects,
        "files": [os.path.basename(f) for f in files],
        "summary": final,
        "per_subject": all_per_subject,
        "top5_tdca_05s": [(s, a) for s, a in tdca_05[:5]],
        "bottom5_tdca_05s": [(s, a) for s, a in tdca_05[-5:]],
    }
    out_path = os.path.join(REPORT_DIR, "merged_baseline_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n合并结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
