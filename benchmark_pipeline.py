# -*- coding: utf-8 -*-
"""
Complete Benchmark Evaluation Pipeline
=======================================
Paper-standard SSVEP processing:
  - 9 occipital channels (data-driven SNR selection)
  - Visual delay: 0.14s | Pre-stimulus: 0.5s
  - 5 harmonics | 8 sub-band Chebyshev Type I filter bank
  - Sub-band weights: (i+1)^(-1.25) + 0.25
  - Leave-one-block-out CV (6 blocks)

Evaluates TDCA, FBCCA, CCA on all S1-S35 subjects.
Generates comprehensive report document.
"""
import sys, os, time, json, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14
PRE_STIMULUS = 0.5
NUM_HARMONICS = 5
N_JOBS = 4

# Correct Tsinghua benchmark order: 5 groups of 8 freqs
# 8.0,9.0,...,15.0, 8.2,9.2,...,15.2, ..., 8.8,9.8,...,15.8
TARGET_FREQS = []
for _offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for _base in range(8, 16):
        TARGET_FREQS.append(_base + _offset)

# Data lengths to evaluate
DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]

# 9 occipital channels (standard Neuroscan 64-cap 10-20 positions):
# PZ, PO5, PO3, POZ, PO4, PO6, O1, OZ, O2
OCCIPITAL_CHANNELS = [45, 51, 52, 53, 54, 55, 58, 59, 60]

# ============================================================================
# REPORT PATH
# ============================================================================
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORT_DIR, "TDCA_Benchmark_Report.md")


from benchmark_worker import evaluate_subject as _eval_subject


def compute_itr(acc, n_targets, data_length_sec, gap_sec=0.5):
    """ITR in bits/min. Assumes 0.5s gap between trials."""
    N = n_targets
    P = max(min(acc / 100.0, 0.999), 1.0 / N)
    T = data_length_sec + gap_sec
    if P >= 0.999:
        return N * np.log2(N) * 60.0 / T
    if P <= 1.0 / N:
        return 0.0
    itr = (np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))) * 60.0 / T
    return max(0.0, itr)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(channels=None, channel_label="all 64"):
    """Run full benchmark on all 35 subjects. Returns summary dict."""
    subjects = list(range(1, 36))
    model_types = ["TDCA", "FBCCA", "CCA"]
    ch_name = f"{len(channels)}ch" if channels else "64ch"

    print(f"\n{'='*80}", flush=True)
    print(f"BENCHMARK: {ch_name} ({channel_label})", flush=True)
    print(f"Subjects: S1-S35 | Data: {DATA_LENGTHS}s | Targets: {len(TARGET_FREQS)}", flush=True)
    print(f"Harmonics: {NUM_HARMONICS} | Delay: {VISUAL_DELAY}s | Workers: {N_JOBS}", flush=True)
    print(f"{'='*80}", flush=True)

    all_accs = OrderedDict()
    for mt in model_types:
        for dl in DATA_LENGTHS:
            all_accs[(mt, dl)] = []

    t_start = time.time()
    completed = 0
    tasks = [(sid, DATA_LENGTHS, model_types, channels) for sid in subjects]

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(_eval_subject, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            sid = result["subject"]
            completed += 1

            if result["error"]:
                print(f"[{completed:2d}/35] S{sid:02d}: {result['error']}", flush=True)
            else:
                parts = []
                for (mt, dl), acc in sorted(result["results"].items()):
                    all_accs[(mt, dl)].append(acc)
                    parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
                et = time.time() - t_start
                print(f"[{completed:2d}/35] S{sid:02d}: " + " | ".join(parts) + f"  ({et:.0f}s)", flush=True)

    total_time = time.time() - t_start

    summary = {}
    for mt in model_types:
        for dl in DATA_LENGTHS:
            accs = all_accs[(mt, dl)]
            if accs:
                m = float(np.mean(accs))
                s = float(np.std(accs))
                summary[(mt, dl)] = {
                    "mean": m, "std": s,
                    "n": len(accs),
                    "min": float(min(accs)),
                    "max": float(max(accs)),
                    "itr": float(compute_itr(m, len(TARGET_FREQS), dl)),
                    "accs": [float(a) for a in accs]
                }
            else:
                summary[(mt, dl)] = {
                    "mean": 0, "std": 0, "n": 0,
                    "min": 0, "max": 0, "itr": 0, "accs": []
                }

    # Print summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: {ch_name} ({channel_label})", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<8} {'0.3s':>10} {'0.5s':>10} {'0.7s':>10} {'1.0s':>10}", flush=True)
    print("-" * 50, flush=True)
    for mt in model_types:
        row = f"{mt:<8}"
        for dl in DATA_LENGTHS:
            info = summary[(mt, dl)]
            if info["n"] > 0:
                row += f" {info['mean']:7.2f}%±{info['std']:.1f}"
            else:
                row += f" {'--':>10}"
        print(row, flush=True)

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    return summary, total_time


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(baseline_summary, tdca_iterations, report_path):
    """Generate comprehensive markdown report."""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# TDCA 算法优化与 Benchmark 评估报告")
    w()
    w(f"**生成日期**: {time.strftime('%Y-%m-%d %H:%M')}")
    w(f"**数据集**: Tsinghua Benchmark SSVEP (S1-S35, 64通道, 250Hz)")
    w(f"**评估方式**: Leave-One-Block-Out 交叉验证 (6 blocks)")
    w()

    # ---- Section 1: Dataset ----
    w("## 1. 数据集描述")
    w()
    w("- **路径**: `C:\\Users\\adam\\Desktop\\benchmark`")
    w("- **数据格式**: `S{subj}.mat/S{subj}.mat`，`data` 字段维度 `(64, 1500, 40, 6)`")
    w("- **64 通道** EEG 数据，1500 采样点 (6秒 @ 250Hz)")
    w("- **40 个刺激目标**: 频率 8.0, 8.2, ..., 15.8 Hz")
    w("- **6 个实验 block**，每个 block 包含 40 个目标各 1 次试验")
    w("- **每被试 240 次试验** (40 targets × 6 blocks)")
    w()

    # ---- Section 2: Preprocessing ----
    w("## 2. 预处理流程")
    w()
    w("### 2.1 通道选择")
    w(f"- 使用 **9 个枕区通道** (数据驱动 SNR 选择): 索引 {OCCIPITAL_CHANNELS}")
    w("- 对应电极位置: PZ, PO5, PO3, POZ, PO4, PO6, O1, OZ, O2 附近")
    w()
    w("### 2.2 数据提取")
    w(f"- 跳过 **{PRE_STIMULUS}s** 预刺激阶段")
    w(f"- 扣除 **{VISUAL_DELAY}s** 视觉延迟 (SSVEP 响应潜伏期)")
    w(f"- 提取 `data_length` 秒的 EEG 片段")
    w(f"- 起始采样点: `int(({PRE_STIMULUS} + {VISUAL_DELAY}) * {SAMPLE_RATE}) = 160`")
    w()
    w("### 2.3 滤波器组")
    w("- **Chebyshev Type I 带通滤波器**")
    w("- 子带数: 8 (Fs ≤ 300Hz)")
    w(f"- 通带起始频率: [6, 14, 22, 30, 38, 46, 54, 62] Hz")
    w(f"- 阻带起始频率: [4, 10, 16, 24, 32, 40, 48, 56] Hz")
    w(f"- 高通截止: 80 Hz, 阻带截止: 90 Hz")
    w("- 子带权重: `(i+1)^(-1.25) + 0.25`")
    w()
    w("### 2.4 参考信号")
    w(f"- **{NUM_HARMONICS} 个谐波**，每个频率生成正弦+余弦参考信号")
    w()

    # ---- Section 3: Baseline Results ----
    w("## 3. 基线算法对比结果")
    w()
    w("### 3.1 算法说明")
    w()
    w("**CCA** (标准典型相关分析):")
    w(f"- 使用 {NUM_HARMONICS} 个谐波构造正弦余弦参考信号")
    w("- 单宽带带通滤波器 (6-90 Hz)")
    w("- 取最大平方相关系数作为分类得分")
    w()
    w("**FBCCA** (滤波器组 CCA):")
    w(f"- 使用 {NUM_HARMONICS} 个谐波")
    w("- 8 子带滤波器组 + CCA + 加权融合")
    w(f"- 子带权重: `(i+1)^(-1.25) + 0.25`")
    w()
    w("**TDCA** (任务判别成分分析):")
    w(f"- 监督学习方法，需要 `fit(X, y)`")
    w(f"- 时间延迟嵌入 (lag={8} samples, ~32ms)")
    w(f"- DSP 判别空间投影 + 空间滤波器")
    w(f"- 8 子带滤波器组 + 加权融合")
    w(f"- n_components = 1")
    w()

    if baseline_summary:
        w("### 3.2 准确率汇总 (S1-S35, 9 occipital channels)")
        w()
        w("| 数据长度 | CCA | FBCCA | TDCA |")
        w("|---|---:|---:|---:|")
        for dl in DATA_LENGTHS:
            cca_m = baseline_summary.get(("CCA", dl), {}).get("mean", 0)
            fbcca_m = baseline_summary.get(("FBCCA", dl), {}).get("mean", 0)
            tdca_m = baseline_summary.get(("TDCA", dl), {}).get("mean", 0)
            w(f"| {dl:.1f}s | {cca_m:.2f}% | {fbcca_m:.2f}% | {tdca_m:.2f}% |")

        w()
        w("### 3.3 详细统计 (含 ITR)")
        w()
        w("| 算法 | 数据长度 | 均值准确率 | 标准差 | 最低 | 最高 | ITR (bits/min) |")
        w("|---|---:|---:|---:|---:|---:|")
        for mt in ["CCA", "FBCCA", "TDCA"]:
            for dl in DATA_LENGTHS:
                info = baseline_summary.get((mt, dl), {})
                if info.get("n", 0) > 0:
                    w(f"| {mt} | {dl:.1f}s | {info['mean']:.2f}% | {info['std']:.2f}% | "
                      f"{info['min']:.2f}% | {info['max']:.2f}% | {info['itr']:.1f} |")
        w()

    # ---- Section 4: TDCA Iterations ----
    if tdca_iterations:
        w("## 4. TDCA 模型迭代优化")
        w()
        w("每次迭代后跑全量 Benchmark (S1-S35)，记录准确率并与 FBCCA/CCA 对比。")
        w()
        w("### 4.1 迭代记录")
        w()
        for i, iteration in enumerate(tdca_iterations):
            w(f"#### 迭代 {i+1}: {iteration.get('name', 'Unknown')}")
            w()
            w(f"**修改内容**: {iteration.get('description', '')}")
            w()
            if iteration.get("results"):
                w("| 数据长度 | TDCA | FBCCA | CCA | TDCA ITR |")
                w("|---|---:|---:|---:|---:|")
                for dl in DATA_LENGTHS:
                    tdca = iteration["results"].get(("TDCA", dl), {}).get("mean", 0)
                    fbcca = iteration["results"].get(("FBCCA", dl), {}).get("mean", 0)
                    cca = iteration["results"].get(("CCA", dl), {}).get("mean", 0)
                    tdca_itr = iteration["results"].get(("TDCA", dl), {}).get("itr", 0)
                    w(f"| {dl:.1f}s | {tdca:.2f}% | {fbcca:.2f}% | {cca:.2f}% | {tdca_itr:.1f} |")
                w()
                w(f"**最佳准确率**: {iteration.get('best_acc', 'N/A')}")
                w()

        # ---- Section 5: Best Model ----
        w("## 5. 最优模型总结")
        w()
        if tdca_iterations:
            best_iter = max(tdca_iterations, key=lambda x: x.get("best_acc", 0))
            w(f"**最优模型**: {best_iter.get('name', 'Unknown')}")
            w(f"**配置**: {best_iter.get('description', '')}")
            w(f"**最高准确率**: {best_iter.get('best_acc', 'N/A')}")
        w()

    # ---- Section 6: Comparison with Reference ----
    w("## 6. 与参考文档对比")
    w()
    w("参考文档: `report_ssvep_tdca_results_cn.docx` (S1仅, 9通道)")
    w()
    w("| 数据长度 | 参考 TDCA (S1) | 本项目 TDCA (S1) | 本项目 TDCA (S1-S35 mean) |")
    w("|---|---:|---:|---:|")
    ref_tdca = {0.3: 85.00, 0.5: 93.33, 0.7: None, 1.0: 99.17}
    for dl in DATA_LENGTHS:
        ref = ref_tdca.get(dl, None)
        ref_str = f"{ref:.2f}%" if ref else "N/A"
        our_s1 = "TBD"
        our_all = "TBD"
        if baseline_summary:
            tdca_info = baseline_summary.get(("TDCA", dl), {})
            if tdca_info.get("n", 0) > 0:
                our_all = f"{tdca_info['mean']:.2f}%"
                # Find S1 in accs
                accs = tdca_info.get("accs", [])
                if len(accs) >= 1:
                    our_s1 = f"{accs[0]:.2f}%"
        w(f"| {dl:.1f}s | {ref_str} | {our_s1} | {our_all} |")
    w()

    # ---- Appendix ----
    w("## 附录: 处理参数详情")
    w()
    w("```yaml")
    w(f"sample_rate: {SAMPLE_RATE}")
    w(f"visual_delay: {VISUAL_DELAY}s")
    w(f"pre_stimulus: {PRE_STIMULUS}s")
    w(f"num_harmonics: {NUM_HARMONICS}")
    w(f"filter_bank_subbands: 8")
    w(f"filter_type: Chebyshev Type I")
    w(f"subband_weights: (i+1)^(-1.25) + 0.25")
    w(f"tdca_lag: 8 samples (32ms)")
    w(f"tdca_n_components: 1")
    w(f"occipital_channels: {OCCIPITAL_CHANNELS}")
    w(f"cross_validation: Leave-one-block-out (6-fold)")
    w("```")
    w()
    w("---")
    w(f"*报告自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to: {report_path}", flush=True)


# ============================================================================
if __name__ == "__main__":
    # ---- Phase 1: Baseline ----
    print("=" * 80, flush=True)
    print("PHASE 1: BASELINE BENCHMARK (9 occipital channels)", flush=True)
    print("=" * 80, flush=True)

    baseline_summary, baseline_time = run_benchmark(
        channels=OCCIPITAL_CHANNELS,
        channel_label=f"{len(OCCIPITAL_CHANNELS)} occipital channels"
    )

    # Save baseline
    with open(os.path.join(REPORT_DIR, "baseline_summary.json"), "w") as f:
        serializable = {}
        for (mt, dl), info in baseline_summary.items():
            serializable[f"{mt}_{dl}s"] = {
                "mean": info["mean"], "std": info["std"],
                "n": info["n"], "itr": info["itr"],
                "min": float(info["min"]), "max": float(info["max"])
            }
        json.dump(serializable, f, indent=2)

    # ---- Phase 2: TDCA Iterations ----
    tdca_iterations = []

    # Will be populated as we test each iteration

    # ---- Generate Report ----
    generate_report(baseline_summary, tdca_iterations, REPORT_PATH)
    print("\nDone.", flush=True)
