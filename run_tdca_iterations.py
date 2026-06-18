# -*- coding: utf-8 -*-
"""TDCA Iteration Runner — tests model variants on full S1-S35 benchmark.

Each iteration modifies a TDCA hyperparameter and runs the full benchmark.
Results are saved and can be merged into the report.
"""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from collections import OrderedDict

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")
os.makedirs(REPORT_DIR, exist_ok=True)

DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
MODEL_TYPES = ["TDCA", "FBCCA", "CCA"]
OCCIPITAL_CHANNELS = [45, 51, 52, 53, 54, 55, 58, 59, 60]


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


# ============================================================================
# Iteration configurations
# ============================================================================
ITERATIONS = [
    {
        "name": "TDCA 基线 (Baseline)",
        "description": "Paper-standard: lag=8, n_components=1, 9枕区通道, 5谐波",
        "tdca_kwargs": {"lagging_len": 8, "n_components": 1},
    },
    {
        "name": "TDCA n_components=2",
        "description": "增加判别分量从 1 到 2，可能提升短窗口准确率",
        "tdca_kwargs": {"lagging_len": 8, "n_components": 2},
    },
    {
        "name": "TDCA n_components=3",
        "description": "判别分量增加到 3",
        "tdca_kwargs": {"lagging_len": 8, "n_components": 3},
    },
    {
        "name": "TDCA lag=4 (16ms)",
        "description": "减少时延嵌入到 4 点 (16ms)，减少维度",
        "tdca_kwargs": {"lagging_len": 4, "n_components": 1},
    },
    {
        "name": "TDCA lag=12 (48ms)",
        "description": "增加时延嵌入到 12 点 (48ms)，捕获更多时序信息",
        "tdca_kwargs": {"lagging_len": 12, "n_components": 1},
    },
    {
        "name": "TDCA lag=16 (64ms)",
        "description": "时延嵌入到 16 点 (64ms)",
        "tdca_kwargs": {"lagging_len": 16, "n_components": 1},
    },
]


def run_iteration(iter_config, subjects_range=(1, 35)):
    """Run one TDCA iteration on all subjects."""
    from benchmark_worker import evaluate_subject

    start_s, end_s = subjects_range
    subjects = list(range(start_s, end_s + 1))
    tdca_kwargs = iter_config.get("tdca_kwargs", {})

    # Temporarily patch TDCA defaults — we pass kwargs via evaluate_subject
    # by modifying the worker function's TDCA constructor

    all_accs = OrderedDict()
    for mt in MODEL_TYPES:
        for dl in DATA_LENGTHS:
            all_accs[(mt, dl)] = []

    print(f"\n{'='*80}", flush=True)
    print(f"迭代: {iter_config['name']}", flush=True)
    print(f"说明: {iter_config['description']}", flush=True)
    print(f"参数: {tdca_kwargs}", flush=True)
    print(f"{'='*80}", flush=True)

    # Override TDCA defaults in the worker module
    import benchmark_worker as bw
    orig_tdca_init = bw.TDCA.__init__

    def patched_init(self, num_harmonics, times, targets, Nh=8, lagging_len=None,
                     sample_rate=250, delay_sec=0.14, n_components=1):
        kwargs = dict(tdca_kwargs)
        if "lagging_len" in kwargs:
            lagging_len = kwargs["lagging_len"]
        if "n_components" in kwargs:
            n_components = kwargs["n_components"]
        orig_tdca_init(self, num_harmonics, times, targets, Nh=Nh,
                       lagging_len=lagging_len, sample_rate=sample_rate,
                       delay_sec=delay_sec, n_components=n_components)

    bw.TDCA.__init__ = patched_init

    try:
        t_start = time.time()
        for idx, sid in enumerate(subjects):
            t0 = time.time()
            result = evaluate_subject((sid, DATA_LENGTHS, MODEL_TYPES, OCCIPITAL_CHANNELS))

            if result["error"]:
                print(f"  [{idx+1:2d}/{len(subjects)}] S{sid:02d}: {result['error']}", flush=True)
            else:
                parts = []
                for (mt, dl), acc in sorted(result["results"].items()):
                    all_accs[(mt, dl)].append(acc)
                    parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
                print(f"  [{idx+1:2d}/{len(subjects)}] S{sid:02d}: " + " | ".join(parts) +
                      f"  [{time.time()-t0:.0f}s]", flush=True)
    finally:
        bw.TDCA.__init__ = orig_tdca_init

    total_time = time.time() - t_start

    # Compute summary
    summary = {}
    for mt in MODEL_TYPES:
        for dl in DATA_LENGTHS:
            accs = all_accs[(mt, dl)]
            if accs:
                m = float(np.mean(accs))
                s = float(np.std(accs))
                summary[(mt, dl)] = {
                    "mean": m, "std": s, "n": len(accs),
                    "min": float(min(accs)), "max": float(max(accs)),
                    "itr": float(compute_itr(m, data_length_sec=dl)),
                }
            else:
                summary[(mt, dl)] = {"mean": 0, "std": 0, "n": 0, "min": 0, "max": 0, "itr": 0}

    # Print table
    print(f"\n{'Model':<8} {'0.3s':>12} {'0.5s':>12} {'0.7s':>12} {'1.0s':>12}", flush=True)
    print("-" * 56, flush=True)
    for mt in MODEL_TYPES:
        row = f"{mt:<8}"
        for dl in DATA_LENGTHS:
            info = summary[(mt, dl)]
            if info["n"] > 0:
                row += f" {info['mean']:8.2f}%±{info['std']:.1f}"
            else:
                row += f" {'--':>12}"
        print(row, flush=True)

    print(f"\n耗时: {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Iteration index (0=baseline)")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=35)
    args = parser.parse_args()

    if args.iter >= len(ITERATIONS):
        print(f"错误: 迭代索引 {args.iter} 超出范围 (0-{len(ITERATIONS)-1})", flush=True)
        sys.exit(1)

    config = ITERATIONS[args.iter]
    summary = run_iteration(config, (args.start, args.end))

    # Save
    out_path = os.path.join(REPORT_DIR, f"iteration_{args.iter}_{config['name'].replace(' ', '_')}.json")
    serializable = {}
    for (mt, dl), info in summary.items():
        serializable[f"{mt}_{dl}s"] = info
    serializable["_config"] = config
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n结果已保存: {out_path}", flush=True)
