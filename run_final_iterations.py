# -*- coding: utf-8 -*-
"""Full benchmark iterations: baseline (n_comp=1) vs optimal (n_comp=3) on all 26 subjects.
Runs TDCA, FBCCA, CCA for both configs and records results.
"""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from collections import OrderedDict
from benchmark_worker import evaluate_subject, OCCIPITAL_CHANNELS

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")
os.makedirs(REPORT_DIR, exist_ok=True)

DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
MODEL_TYPES = ["TDCA", "FBCCA", "CCA"]
SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,26,27,28,29,30,31,32]

def compute_itr(acc, n=40, dl=1.0, gap=0.5):
    N = n; P = max(min(acc/100.0, 0.999), 1.0/N); T = dl + gap
    if P >= 0.999: return N * np.log2(N) * 60.0 / T
    if P <= 1.0/N: return 0.0
    return max(0.0, (np.log2(N)+P*np.log2(P)+(1-P)*np.log2((1-P)/(N-1)))*60.0/T)

ITERATIONS = [
    {"name": "Iter1_最优_lag8_ncomp3", "desc": "最优: lag=8, n_components=3",
     "tdca_kwargs": {"lagging_len": 8, "n_components": 3}},
]

def run_one_iteration(cfg):
    """Run one full iteration on all subjects."""
    import benchmark_worker as bw
    orig_init = bw.TDCA.__init__

    def patched_init(self, num_harmonics, times, targets, Nh=8, lagging_len=None,
                     sample_rate=250, delay_sec=0.14, n_components=1):
        kwargs = cfg["tdca_kwargs"]
        orig_init(self, num_harmonics, times, targets, Nh=Nh,
                  lagging_len=kwargs.get("lagging_len", 8),
                  sample_rate=sample_rate, delay_sec=delay_sec,
                  n_components=kwargs.get("n_components", 1))

    bw.TDCA.__init__ = patched_init

    try:
        all_accs = OrderedDict()
        for mt in MODEL_TYPES:
            for dl in DATA_LENGTHS:
                all_accs[(mt, dl)] = []

        print(f"\n{'='*80}", flush=True)
        print(f"迭代: {cfg['desc']}", flush=True)
        print(f"{'='*80}", flush=True)

        t_start = time.time()
        for idx, sid in enumerate(SUBJECTS):
            t0 = time.time()
            result = evaluate_subject((sid, DATA_LENGTHS, MODEL_TYPES, OCCIPITAL_CHANNELS))
            if result["error"]:
                print(f"  [{idx+1:2d}/{len(SUBJECTS)}] S{sid:02d}: {result['error']}", flush=True)
            else:
                parts = []
                for (mt, dl), acc in sorted(result["results"].items()):
                    all_accs[(mt, dl)].append(acc)
                    parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
                print(f"  [{idx+1:2d}/{len(SUBJECTS)}] S{sid:02d}: " + " | ".join(parts) +
                      f"  [{time.time()-t0:.0f}s] (总 {time.time()-t_start:.0f}s)", flush=True)

        summary = {}
        for mt in MODEL_TYPES:
            for dl in DATA_LENGTHS:
                accs = all_accs[(mt, dl)]
                if accs:
                    m, s = float(np.mean(accs)), float(np.std(accs))
                    summary[f"{mt}_{dl}s"] = {"mean": m, "std": s, "n": len(accs),
                                              "min": float(min(accs)), "max": float(max(accs)),
                                              "itr": float(compute_itr(m, dl=dl))}

        # Print table
        print(f"\n{'模型':<8} {'0.3s':>12} {'0.5s':>12} {'0.7s':>12} {'1.0s':>12}", flush=True)
        print("-" * 56, flush=True)
        for mt in MODEL_TYPES:
            row = f"{mt:<8}"
            for dl in DATA_LENGTHS:
                info = summary.get(f"{mt}_{dl}s", {})
                if info.get("n", 0) > 0:
                    row += f" {info['mean']:8.2f}%±{info['std']:.1f}"
                else:
                    row += f" {'--':>12}"
            print(row, flush=True)

        print(f"总耗时: {time.time()-t_start:.0f}s", flush=True)
        return summary

    finally:
        bw.TDCA.__init__ = orig_init


def main():
    all_summaries = {}

    for cfg in ITERATIONS:
        summary = run_one_iteration(cfg)
        all_summaries[cfg["name"]] = summary

        # Save after each iteration
        with open(os.path.join(REPORT_DIR, f"{cfg['name']}_full.json"), "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "summary": summary, "n_subjects": len(SUBJECTS)}, f,
                      indent=2, ensure_ascii=False)

    # Final comparison with baseline (loaded from compiled JSON)
    baseline_path = os.path.join(REPORT_DIR, "compiled_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
    else:
        baseline = {}

    print(f"\n{'='*80}")
    print("最终对比: 基线 (n_comp=1) vs 最优 (n_comp=3) — 全 26 被试")
    print(f"{'='*80}")
    for dl in DATA_LENGTHS:
        bl = baseline.get("TDCA", {}).get(f"{dl}s", {}).get("mean", 0)
        opt = all_summaries["Iter1_最优_lag8_ncomp3"].get(f"TDCA_{dl}s", {}).get("mean", 0)
        fbcca = baseline.get("FBCCA", {}).get(f"{dl}s", {}).get("mean", 0)
        cca = baseline.get("CCA", {}).get(f"{dl}s", {}).get("mean", 0)
        delta = opt - bl
        print(f"  {dl:.1f}s: CCA={cca:.2f}% | FBCCA={fbcca:.2f}% | TDCA基线={bl:.2f}% → 最优={opt:.2f}% (Δ={delta:+.2f}pp)")

    # Save combined final report
    final = {
        "baseline_TDCA": baseline.get("TDCA", {}),
        "optimal_TDCA_ncomp3": all_summaries.get("Iter1_最优_lag8_ncomp3", {}),
        "FBCCA": baseline.get("FBCCA", {}),
        "CCA": baseline.get("CCA", {}),
    }
    with open(os.path.join(REPORT_DIR, "final_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n完成! 结果保存在: {REPORT_DIR}")

if __name__ == "__main__":
    main()
