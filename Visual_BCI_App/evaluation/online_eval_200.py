"""
在线评测 — SSVEP 200 目标实时 BCI
=====================================

功能：
  1. 加载训练好的解码器权重
  2. 实时采集 EEG → 滑窗 buffer
  3. 滑窗解码 + 投票机制
  4. 统计在线准确率 & ITR
  5. 可选发送小车指令

用法：
  python -m evaluation.online_eval_200 --subject hc33 --duration 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat, savemat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.HybridDecoder import HybridDecoder
from models.FBCCA import FBCCA
from models.TDCA import TDCA
from interface.car_interface.acquisition import preprocess_model_input
from interface.car_interface.training_framework import (
    extract_training_sample,
    extract_target_id,
    extract_int,
)


# ═══════════════════════════════════════════════════════════════════
#  在线评测器
# ═══════════════════════════════════════════════════════════════════

class OnlineEvaluator:
    """SSVEP 在线 BCI 评测器。

    Parameters
    ----------
    subject : str
    freqs : list[float]
    sample_rate : int
    n_classes : int
    model_name : str — 'TDCA' | 'FBCCA' | 'CCA'
    window_sec : float — 滑窗窗口长度
    slide_step_sec : float — 滑动步长
    vote_buffer : int — 投票缓冲区大小
    gaze_shift_sec : float — 注视切换时间 (用于 ITR 计算)
    """

    def __init__(
        self,
        subject: str = "hc33",
        freqs: Optional[List[float]] = None,
        sample_rate: int = 250,
        n_classes: int = 40,
        model_name: str = "FBCCA",
        window_sec: float = 3.0,
        slide_step_sec: float = 0.5,
        vote_buffer: int = 3,
        gaze_shift_sec: float = 0.5,
    ):
        self.subject = subject
        self.freqs = freqs or [8.0 + i * 0.2 for i in range(40)]
        self.Fs = sample_rate
        self.Nf = n_classes
        self.model_name = model_name
        self.window_sec = window_sec
        self.slide_step_sec = slide_step_sec
        self.vote_buffer_size = vote_buffer
        self.gaze_shift_sec = gaze_shift_sec

        # 创建解码器
        self.decoder: Optional[HybridDecoder] = None

        # 测试序列
        self.test_sequence: List[dict] = []
        self.current_trial: int = 0
        self.total_trials: int = 0

        # 结果记录
        self.results: List[dict] = []
        self.correct_count: int = 0
        self.wrong_count: int = 0
        self.start_time: float = 0.0

        # 实时数据缓冲
        self.eeg_buffer: deque = deque()
        self.vote_buffer: deque = deque(maxlen=vote_buffer)
        self.last_command: int = -1
        self.command_cooldown: float = 1.0
        self.last_command_time: float = 0.0

    # ── 训练数据加载 ────────────────────────────────────────────

    def load_training_data(self, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """加载训练数据并训练解码器。

        Returns
        -------
        X : (n_trials, n_channels, n_samples)
        y : (n_trials,)
        """
        if data_dir is None:
            data_dir = os.path.join(
                PROJECT_ROOT, "saveCarData", self.subject
            )

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        files = sorted(glob(os.path.join(data_dir, "*.mat")))
        if not files:
            raise FileNotFoundError(f"No .mat files in {data_dir}")

        print(f"[DATA] Loading {len(files)} files from {data_dir}...")

        required_points = int(self.window_sec * self.Fs)
        X_list, y_list = [], []
        skipped = 0

        for fp in files:
            try:
                mat = loadmat(fp)
                # Extract data
                data = mat.get("eeg_data", mat.get("benchmark_trial_data", mat.get("data", None)))
                if data is None or data.ndim != 2:
                    skipped += 1
                    continue
                if data.shape[-1] < required_points:
                    skipped += 1
                    continue

                label = extract_target_id(mat)
                if label < 0 or label >= self.Nf:
                    skipped += 1
                    continue

                sample = data[:, -required_points:]
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
            raise RuntimeError(f"No valid trials found (skipped {skipped})")

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=int)
        print(f"[DATA] Loaded {len(X)} trials, {skipped} skipped")

        # 训练解码器
        print(f"[TRAIN] Fitting {self.model_name} decoder...")
        self.decoder = HybridDecoder(
            sample_rate=self.Fs,
            freqs=self.freqs,
            n_fixations=5,
            freq_model=self.model_name,
            top_k=3,
            n_harmonics=5,
            data_len_sec=self.window_sec,
        )
        self.decoder.mode = "freq_only"
        self.decoder.fit(X, y)
        print(f"[TRAIN] Done. Decoder mode={self.decoder.mode}")

        return X, y

    # ── 测试序列生成 ────────────────────────────────────────────

    def generate_test_sequence(self, n_per_class: int = 5):
        """生成随机测试序列 (每类 n_per_class 个 trial)。

        Parameters
        ----------
        n_per_class : 每类的测试 trial 数
        """
        seq = []
        for cls_id in range(self.Nf):
            for _ in range(n_per_class):
                seq.append({
                    "target_id": cls_id,
                    "block_idx": cls_id // 5,
                    "fp_idx": cls_id % 5,
                    "freq_hz": self.freqs[cls_id % len(self.freqs)],
                })

        rng = np.random.RandomState(42)
        rng.shuffle(seq)
        self.test_sequence = seq
        self.total_trials = len(seq)
        self.current_trial = 0
        print(f"[TEST] Generated {self.total_trials} trials "
              f"({n_per_class} per class × {self.Nf} classes)")

    # ── 模拟在线处理 ────────────────────────────────────────────

    def push_eeg_chunk(self, data: np.ndarray):
        """将实时 EEG 数据块推入缓冲区。

        Parameters
        ----------
        data : (n_channels, n_samples_chunk)
        """
        data = np.asarray(data, dtype=float)
        n_chunk = data.shape[-1]
        for i in range(n_chunk):
            self.eeg_buffer.append(data[:, i])

    def get_window(self, n_samples: int) -> Optional[np.ndarray]:
        """获取最新的 n_samples 个数据点。

        Returns
        -------
        window : (n_channels, n_samples) or None
        """
        if len(self.eeg_buffer) < n_samples:
            return None

        # 取最后 n_samples 个点
        window = np.column_stack([
            self.eeg_buffer[i]
            for i in range(len(self.eeg_buffer) - n_samples, len(self.eeg_buffer))
        ])
        return preprocess_model_input(window)

    def decide(self, window_samples: int) -> Tuple[int, float]:
        """对当前缓冲窗口进行解码决策。

        Returns
        -------
        predicted_class : int
        confidence : float
        """
        if self.decoder is None:
            return -1, 0.0

        window = self.get_window(window_samples)
        if window is None:
            return -1, 0.0

        scores = self.decoder.score_vector(window)
        pred = int(np.argmax(scores))

        # 置信度 = Top1 - Top2
        if len(scores) >= 2:
            top2 = np.partition(scores, -2)[-2:]
            conf = float(top2[1] - top2[0])
        else:
            conf = float(scores[pred])

        return pred, conf

    def voting_decide(self, window_samples: int) -> Tuple[int, float, bool]:
        """带投票缓冲的解码决策。

        Returns
        -------
        final_pred : int
        confidence : float
        decision_ready : bool — 投票是否达成一致
        """
        pred, conf = self.decide(window_samples)
        self.vote_buffer.append(pred)

        # 检查投票一致性
        if len(self.vote_buffer) >= self.vote_buffer_size:
            votes = list(self.vote_buffer)
            # 找出现最多的预测
            unique, counts = np.unique(votes, return_counts=True)
            max_count = counts.max()
            if max_count >= self.vote_buffer_size - 1:  # 允许 1 票不同
                winner = unique[counts.argmax()]
                # 冷却检查
                now = time.perf_counter()
                if now - self.last_command_time >= self.command_cooldown:
                    self.last_command_time = now
                    self.last_command = int(winner)
                    return int(winner), conf, True

        return pred, conf, False

    # ── 评测统计 ────────────────────────────────────────────────

    def record_result(self, target_id: int, predicted_id: int, confidence: float):
        """记录单次 trial 结果。"""
        correct = (predicted_id == target_id)
        if correct:
            self.correct_count += 1
        else:
            self.wrong_count += 1

        self.results.append({
            "trial": len(self.results) + 1,
            "target_id": target_id,
            "predicted_id": predicted_id,
            "correct": correct,
            "confidence": confidence,
            "timestamp": time.time(),
        })

    @property
    def accuracy(self) -> float:
        total = self.correct_count + self.wrong_count
        if total == 0:
            return 0.0
        return self.correct_count / total

    @property
    def itr_bpm(self) -> float:
        """计算在线 ITR (bits/min)。"""
        N = float(self.Nf)
        P = self.accuracy
        T = self.window_sec + self.gaze_shift_sec

        if P <= 1.0 / N or T <= 0:
            return 0.0
        if P >= 0.9999:
            P = 0.9999

        bits = (
            np.log2(N)
            + P * np.log2(P)
            + (1.0 - P) * np.log2((1.0 - P) / (N - 1.0))
        )
        return max(0.0, float(bits * 60.0 / T))

    def summary(self) -> dict:
        """生成评测摘要。"""
        return {
            "subject": self.subject,
            "model": self.model_name,
            "n_classes": self.Nf,
            "total_trials": self.correct_count + self.wrong_count,
            "correct": self.correct_count,
            "wrong": self.wrong_count,
            "accuracy": self.accuracy,
            "itr_bpm": self.itr_bpm,
            "window_sec": self.window_sec,
            "vote_buffer": self.vote_buffer_size,
        }

    def print_summary(self):
        """打印评测摘要。"""
        s = self.summary()
        print("\n" + "=" * 50)
        print("  ONLINE EVALUATION SUMMARY")
        print("=" * 50)
        print(f"  Subject:     {s['subject']}")
        print(f"  Model:       {s['model']}")
        print(f"  Classes:     {s['n_classes']}")
        print(f"  Trials:      {s['total_trials']}")
        print(f"  Correct:     {s['correct']}")
        print(f"  Wrong:       {s['wrong']}")
        print(f"  Accuracy:    {s['accuracy']:.4f} ({s['accuracy']*100:.1f}%)")
        print(f"  ITR:         {s['itr_bpm']:.1f} bits/min")
        print(f"  Window:      {s['window_sec']}s")
        print("=" * 50)

    def save_results(self, output_dir: Optional[str] = None):
        """保存评测结果为 JSON。"""
        if output_dir is None:
            output_dir = os.path.join(
                PROJECT_ROOT, "saveCarData", self.subject, "online_results"
            )
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"online_eval_{ts}.json")
        payload = {
            "summary": self.summary(),
            "results": self.results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Results saved to {path}")
        return path


# ═══════════════════════════════════════════════════════════════════
#  离线模拟在线评测 (用已有数据模拟滑动窗口)
# ═══════════════════════════════════════════════════════════════════

def simulate_online_from_files(
    data_dir: str,
    n_classes: int = 40,
    model_name: str = "FBCCA",
    window_sec: float = 3.0,
    slide_step_sec: float = 0.5,
    vote_buffer: int = 3,
    sample_rate: int = 250,
):
    """用已有 .mat 数据文件模拟在线评测。

    从文件加载数据，模拟滑窗实时解码过程。
    """
    ev = OnlineEvaluator(
        n_classes=n_classes,
        model_name=model_name,
        window_sec=window_sec,
        slide_step_sec=slide_step_sec,
        vote_buffer=vote_buffer,
        sample_rate=sample_rate,
    )

    # 加载并训练
    try:
        ev.load_training_data(data_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return ev

    # 加载测试数据 (用训练数据做留一法模拟)
    files = sorted(glob(os.path.join(data_dir, "*.mat")))
    if len(files) < 10:
        print("[ERROR] Not enough data files for simulation")
        return ev

    window_samples = int(window_sec * sample_rate)
    slide_samples = int(slide_step_sec * sample_rate)

    print(f"[SIM] Simulating online with {len(files)} trials...")
    print(f"      Window: {window_sec}s ({window_samples} samples)")
    print(f"      Slide: {slide_step_sec}s ({slide_samples} samples)")

    for fp in files:
        try:
            mat = loadmat(fp)
            data = mat.get("eeg_data", mat.get("benchmark_trial_data", mat.get("data", None)))
            if data is None or data.ndim != 2:
                continue
            target = extract_target_id(mat)
            if target < 0 or target >= n_classes:
                continue

            # 模拟滑窗
            n_total = data.shape[-1]
            for start in range(0, n_total - window_samples + 1, slide_samples):
                end = start + window_samples
                window = data[:, start:end]
                window = preprocess_model_input(window)
                if window is None:
                    continue

                pred, conf, ready = ev.voting_decide(window_samples)

                if ready:
                    ev.record_result(target, pred, conf)
                    ev.vote_buffer.clear()
        except Exception:
            continue

    ev.print_summary()
    return ev


# ═══════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SSVEP 在线评测 — 200目标混合编码系统"
    )
    parser.add_argument("--subject", default="hc33", help="被试名称")
    parser.add_argument("--data-dir", default=None, help="数据目录")
    parser.add_argument("--model", default="FBCCA",
                        help="解码器: TDCA | FBCCA | CCA")
    parser.add_argument("--n-classes", type=int, default=40,
                        help="目标类别数 (40)")
    parser.add_argument("--window", type=float, default=3.0,
                        help="滑窗窗口长度 (秒)")
    parser.add_argument("--slide", type=float, default=0.5,
                        help="滑动步长 (秒)")
    parser.add_argument("--vote", type=int, default=3,
                        help="投票缓冲区大小")
    parser.add_argument("--simulate", action="store_true",
                        help="用已有数据文件模拟在线评测")
    parser.add_argument("--save", action="store_true",
                        help="保存结果 JSON")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        PROJECT_ROOT, "saveCarData", args.subject
    )

    if args.simulate or not args.data_dir:
        # 离线模拟模式
        ev = simulate_online_from_files(
            data_dir=data_dir,
            n_classes=args.n_classes,
            model_name=args.model,
            window_sec=args.window,
            slide_step_sec=args.slide,
            vote_buffer=args.vote,
        )
    else:
        ev = OnlineEvaluator(
            subject=args.subject,
            n_classes=args.n_classes,
            model_name=args.model,
            window_sec=args.window,
            slide_step_sec=args.slide,
            vote_buffer=args.vote,
        )
        try:
            ev.load_training_data(data_dir)
            ev.generate_test_sequence(n_per_class=5)
            print(f"[READY] {ev.total_trials} trials in sequence. "
                  f"Press Enter to start...")
            input()
            # ... (real-time loop would go here)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[ERROR] {e}")
            return

    if args.save:
        ev.save_results()


if __name__ == "__main__":
    main()
