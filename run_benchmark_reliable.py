# -*- coding: utf-8 -*-
"""可靠的 Benchmark 运行器 — 写入文件日志，每被试完成后立即保存结果。

用法: python run_benchmark_reliable.py
"""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from collections import OrderedDict
from benchmark_worker import evaluate_subject, OCCIPITAL_CHANNELS

DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
MODEL_TYPES = ["TDCA", "FBCCA", "CCA"]
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")
os.makedirs(REPORT_DIR, exist_ok=True)
LOG_PATH = os.path.join(REPORT_DIR, "benchmark_progress.txt")
RESULT_PATH = os.path.join(REPORT_DIR, "baseline_results.json")


def log(msg):
    """写入日志文件并打印到控制台 (强制刷新)"""
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


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
    # 清空日志
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 80)
    log("Benchmark 全量评估: TDCA / FBCCA / CCA (S1-S35)")
    log(f"数据长度: {DATA_LENGTHS}s | 通道: 9 枕区 (10-20)")
    log(f"频率: 40类 SSVEP (5组×8频) | CV: L1BO (6块)")
    log("=" * 80)

    all_accs = OrderedDict()
    for mt in MODEL_TYPES:
        for dl in DATA_LENGTHS:
            all_accs[(mt, dl)] = []
    per_subject = {}

    t_start = time.time()

    for idx, sid in enumerate(range(1, 36)):
        t0 = time.time()

        try:
            result = evaluate_subject((sid, DATA_LENGTHS, MODEL_TYPES, OCCIPITAL_CHANNELS))
        except Exception as e:
            log(f"[{idx+1:2d}/35] S{sid:02d}: 严重错误 - {type(e).__name__}: {e}")
            continue

        if result["error"]:
            log(f"[{idx+1:2d}/35] S{sid:02d}: {result['error']}")
        else:
            parts = []
            subj_data = {}
            for (mt, dl), acc in sorted(result["results"].items()):
                all_accs[(mt, dl)].append(acc)
                parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
                subj_data[f"{mt}_{dl}s"] = acc
            per_subject[f"S{sid:02d}"] = subj_data

            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            log(f"[{idx+1:2d}/35] S{sid:02d}: " + " | ".join(parts) +
                f"  [{elapsed:.0f}s] (总计 {total_elapsed:.0f}s)")

        # 每完成一个被试，立即保存结果
        summary = {}
        for mt in MODEL_TYPES:
            for dl in DATA_LENGTHS:
                accs = all_accs[(mt, dl)]
                if accs:
                    summary[f"{mt}_{dl}s"] = {
                        "mean": float(np.mean(accs)),
                        "std": float(np.std(accs)),
                        "n": len(accs),
                        "min": float(min(accs)),
                        "max": float(max(accs)),
                    }

        save_data = {
            "config": {
                "data_lengths": DATA_LENGTHS,
                "model_types": MODEL_TYPES,
                "channels": OCCIPITAL_CHANNELS,
                "subjects_completed": idx + 1,
            },
            "summary": summary,
            "per_subject": per_subject,
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_start

    # 最终汇总
    log("\n" + "=" * 80)
    log("最终汇总: S1-S35 平均准确率")
    log("=" * 80)
    log(f"{'模型':<8} {'0.3s':>12} {'0.5s':>12} {'0.7s':>12} {'1.0s':>12}  {'ITR@1.0s':>12}")
    log("-" * 68)

    final_summary = {}
    for mt in MODEL_TYPES:
        row = f"{mt:<8}"
        for dl in DATA_LENGTHS:
            accs = all_accs[(mt, dl)]
            if accs:
                m, s = np.mean(accs), np.std(accs)
                row += f" {m:8.2f}%±{s:.1f}"
                final_summary[(mt, dl)] = {"mean": m, "std": s, "n": len(accs)}
            else:
                row += f" {'--':>12}"
        info_1s = final_summary.get((mt, 1.0), {})
        itr = compute_itr(info_1s.get("mean", 0), data_length_sec=1.0) if info_1s else 0
        row += f"  {itr:10.1f}"
        log(row)

    log(f"\n总耗时: {total_time:.0f}s ({total_time/60:.1f} 分钟)")
    log(f"结果已保存至: {RESULT_PATH}")

    return final_summary


if __name__ == "__main__":
    main()
