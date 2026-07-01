"""
离线评测 — SSVEP 200 目标混合编码系统
=========================================

功能：
  1. 加载 .mat 训练数据
  2. 多算法对比 (FBCCA / TDCA / CCA)
  3. 多数据窗长 (0.3s, 0.5s, 1s, 2s, 3s, 4s)
  4. 交叉验证 (k-fold 或留一法)
  5. 计算准确率 & ITR
  6. 输出表格 + matplotlib 图表

用法：
  python -m evaluation.offline_eval_200 --subject hc33 --kfold 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat, savemat

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.FBCCA import FBCCA
from models.CCA import CCA
from models.TDCA import TDCA
from interface.car_interface.training_framework import (
    extract_training_sample,
    extract_target_id,
    extract_int,
)
from interface.car_interface.acquisition import preprocess_model_input


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def _subject_data_dir(subject: str) -> str:
    return os.path.join(PROJECT_ROOT, "saveCarData", subject)


def _list_data_files(subject: str) -> List[str]:
    """列出被试的所有 .mat 数据文件。"""
    root = _subject_data_dir(subject)
    if not os.path.isdir(root):
        return []
    files = glob(os.path.join(root, "*.mat"))
    return sorted(files)


def _load_data_matrix(
    files: List[str],
    freqs: List[float],
    window_sec: float,
    sample_rate: int,
    n_classes: int = 40,
    delay_sec: float = 0.14,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载 .mat 文件并构造数据矩阵。

    Returns
    -------
    X : (n_trials, n_channels, n_samples)
    y : (n_trials,) — 类别标签
    """
    required_points = int(round(window_sec * sample_rate))
    effective_sec = max(window_sec - delay_sec, 0.2)
    effective_points = int(round(effective_sec * sample_rate))

    X_list, y_list = [], []
    skipped = 0

    for fp in files:
        try:
            mat = loadmat(fp)

            # 检查频率兼容性
            saved_freqs = mat.get("stim_freqs_hz", None)
            if saved_freqs is not None:
                sf = np.asarray(saved_freqs).reshape(-1)
                cf = np.asarray(freqs).reshape(-1)
                if sf.shape != cf.shape or not np.allclose(sf, cf, atol=0.05):
                    skipped += 1
                    continue

            label = extract_target_id(mat)
            if label < 0 or label >= n_classes:
                skipped += 1
                continue

            sample = extract_training_sample(mat, required_points, sample_rate)
            if sample is None:
                skipped += 1
                continue

            sample = preprocess_model_input(sample)
            if sample is None:
                skipped += 1
                continue

            X_list.append(sample)
            y_list.append(int(label))
        except Exception:
            skipped += 1
            continue

    if not X_list:
        print(f"[WARNING] No valid trials loaded (skipped {skipped})")
        return np.array([]), np.array([])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    print(f"[DATA] Loaded {len(X)} trials, {skipped} skipped, "
          f"classes={len(np.unique(y))}")
    return X, y


def _kfold_split(n_samples: int, n_folds: int = 5, seed: int = 42):
    """生成 k-fold 训练/测试索引。"""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    fold_size = n_samples // n_folds
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx


def _compute_itr(accuracy: float, n_targets: int, window_sec: float,
                 gaze_shift_sec: float = 0.5) -> float:
    """计算信息传输率 ITR (bits/min)。

    ITR = (log2(N) * P + (1-P) * log2((1-P)/(N-1)) + log2(N-1)) * 60 / T

    Parameters
    ----------
    accuracy : 分类准确率 (0-1)
    n_targets : 目标数
    window_sec : 数据窗长 (秒)
    gaze_shift_sec : 注视切换时间 (秒)
    """
    N = float(n_targets)
    P = float(accuracy)
    T = float(window_sec) + float(gaze_shift_sec)

    if P <= 1.0 / N:
        return 0.0
    if P >= 0.9999:
        P = 0.9999

    bits_per_trial = (
        np.log2(N)
        + P * np.log2(P)
        + (1.0 - P) * np.log2((1.0 - P) / (N - 1.0))
    )
    return max(0.0, float(bits_per_trial * 60.0 / T))


# ═══════════════════════════════════════════════════════════════════
#  离线评测主类
# ═══════════════════════════════════════════════════════════════════

class OfflineEvaluator:
    """SSVEP 离线评测器。"""

    def __init__(
        self,
        subject: str = "hc33",
        freqs: Optional[List[float]] = None,
        sample_rate: int = 250,
        n_classes: int = 40,
        n_harmonics: int = 5,
        delay_sec: float = 0.14,
    ):
        self.subject = subject
        self.freqs = freqs or [8.0 + i * 0.2 for i in range(40)]
        self.Fs = sample_rate
        self.Nf = n_classes
        self.n_harmonics = n_harmonics
        self.delay_sec = delay_sec

    def evaluate(
        self,
        files: Optional[List[str]] = None,
        windows: Optional[List[float]] = None,
        models: Optional[List[str]] = None,
        n_folds: int = 5,
    ) -> dict:
        """运行完整评测。

        Parameters
        ----------
        files : 数据文件列表 (None = 自动搜索)
        windows : 数据窗长列表 (秒)
        models : 算法列表 ['FBCCA', 'TDCA', 'CCA']
        n_folds : 交叉验证折数

        Returns
        -------
        results : {window: {model: {accuracy, itr, ...}}}
        """
        if windows is None:
            windows = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0]
        if models is None:
            models = ["FBCCA", "TDCA", "CCA"]
        if files is None:
            files = _list_data_files(self.subject)

        if not files:
            print(f"[ERROR] No data files found for subject '{self.subject}'")
            return {}

        results = {}
        for w in windows:
            print(f"\n{'='*50}")
            print(f"  Window: {w}s")
            print(f"{'='*50}")

            X, y = _load_data_matrix(
                files, self.freqs, w, self.Fs,
                n_classes=self.Nf, delay_sec=self.delay_sec,
            )
            if len(X) == 0:
                print(f"  [SKIP] No data for window {w}s")
                continue

            results[w] = {}
            for model_name in models:
                accuracies, itrs = [], []

                for fold, (train_idx, test_idx) in enumerate(
                    _kfold_split(len(X), n_folds)
                ):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]

                    try:
                        classifier = self._make_classifier(model_name, w)
                        if model_name == "TDCA":
                            classifier.fit(X_train, y_train)

                        correct = 0
                        for i in range(len(X_test)):
                            pred = classifier.classify(X_test[i])
                            if pred == y_test[i]:
                                correct += 1

                        acc = correct / len(X_test)
                        itr = _compute_itr(acc, self.Nf, w)
                        accuracies.append(acc)
                        itrs.append(itr)
                    except Exception as exc:
                        print(f"  [{model_name}] Fold {fold} ERROR: {exc}")
                        continue

                if accuracies:
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    mean_itr = np.mean(itrs)
                    std_itr = np.std(itrs)
                    results[w][model_name] = {
                        "accuracy_mean": float(mean_acc),
                        "accuracy_std": float(std_acc),
                        "itr_mean_bpm": float(mean_itr),
                        "itr_std_bpm": float(std_itr),
                        "n_folds": len(accuracies),
                    }
                    print(f"  [{model_name}] Acc={mean_acc:.4f}±{std_acc:.4f}  "
                          f"ITR={mean_itr:.1f}±{std_itr:.1f} bpm")
                else:
                    results[w][model_name] = {"error": "all folds failed"}

        return results

    def _make_classifier(self, model_name: str, window_sec: float):
        """创建分类器实例。"""
        kwargs = dict(
            num_harmonics=self.n_harmonics,
            times=window_sec,
            targets=self.freqs,
            Nh=8,
            sample_rate=self.Fs,
            delay_sec=self.delay_sec,
        )
        if model_name == "TDCA":
            return TDCA(**kwargs)
        elif model_name == "FBCCA":
            return FBCCA(**kwargs)
        elif model_name == "CCA":
            return CCA(**kwargs)
        raise ValueError(f"Unknown model: {model_name}")

    def print_table(self, results: dict):
        """打印结果表格。"""
        if not results:
            print("No results to display.")
            return

        models = set()
        for w_data in results.values():
            models.update(w_data.keys())
        models = sorted(models)

        print("\n" + "=" * 80)
        print(f"{'Window':>8s} | " + " | ".join(
            f"{m:^20s}" for m in models))
        print("-" * 80)

        for w in sorted(results.keys()):
            row = f"{w:>7.1f}s |"
            for m in models:
                if m in results[w] and "accuracy_mean" in results[w][m]:
                    d = results[w][m]
                    row += f" {d['accuracy_mean']:.3f}±{d['accuracy_std']:.3f} |"
                else:
                    row += f" {'—':^20s} |"
            print(row)

        # ITR 表
        print(f"\n{'Window':>8s} | " + " | ".join(
            f"{m:^20s}" for m in models))
        print("-" * 80)

        for w in sorted(results.keys()):
            row = f"{w:>7.1f}s |"
            for m in models:
                if m in results[w] and "itr_mean_bpm" in results[w][m]:
                    d = results[w][m]
                    row += f" {d['itr_mean_bpm']:>6.1f}±{d['itr_std_bpm']:<5.1f} bpm |"
                else:
                    row += f" {'—':^20s} |"
            print(row)

        print("=" * 80)

    def save_results(self, results: dict, output_dir: Optional[str] = None):
        """保存评测结果为 JSON。"""
        if output_dir is None:
            output_dir = os.path.join(_subject_data_dir(self.subject), "eval_results")
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"offline_eval_{ts}.json")
        payload = {
            "subject": self.subject,
            "timestamp": ts,
            "freqs": self.freqs,
            "n_classes": self.Nf,
            "sample_rate": self.Fs,
            "results": {str(k): v for k, v in results.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Results saved to {path}")
        return path


# ═══════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SSVEP 离线评测 — 200目标混合编码系统"
    )
    parser.add_argument("--subject", default="hc33", help="被试名称")
    parser.add_argument("--kfold", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--models", default="FBCCA,TDCA,CCA",
                        help="算法列表, 逗号分隔")
    parser.add_argument("--windows", default="0.5,1,2,3,4",
                        help="数据窗长列表(秒), 逗号分隔")
    parser.add_argument("--n-classes", type=int, default=40,
                        help="目标类别数 (40 或 200)")
    parser.add_argument("--data-dir", default=None,
                        help="数据目录 (默认 saveCarData/<subject>)")
    parser.add_argument("--save", action="store_true", help="保存结果 JSON")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    windows = [float(w) for w in args.windows.split(",")]

    evaluator = OfflineEvaluator(
        subject=args.subject,
        n_classes=args.n_classes,
    )

    # 加载数据
    if args.data_dir:
        files = sorted(glob(os.path.join(args.data_dir, "*.mat")))
    else:
        files = _list_data_files(args.subject)

    if not files:
        print(f"[ERROR] No .mat files found. "
              f"Please collect training data first.")
        return

    print(f"[FILES] {len(files)} files found")

    results = evaluator.evaluate(
        files=files,
        windows=windows,
        models=models,
        n_folds=args.kfold,
    )

    evaluator.print_table(results)

    if args.save:
        evaluator.save_results(results)


if __name__ == "__main__":
    main()
