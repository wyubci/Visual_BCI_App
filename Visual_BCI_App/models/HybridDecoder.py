"""
混合解码器 — 频率 + 空间级联
==================================

两级级联架构：
  1. 频率解码器 (TDCA/FBCCA/CCA): 40 类 → 识别闪烁块
  2. 空间解码器 (SpatialDecoder): 5 类 → 识别注视点

融合输出 200 类得分向量。

对于仅 40 目标模式（无空间解码），直接使用频率解码器。

参考文献：
  Liao et al., "Hybrid frequency-phase-spatial encoding", Cyborg Bionic Syst, 2024
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from models.FBCCA import FBCCA
from models.CCA import CCA
from models.TDCA import TDCA
from models.SpatialDecoder import SpatialDecoder


# ═══════════════════════════════════════════════════════════════════
#  HybridDecoder
# ═══════════════════════════════════════════════════════════════════

class HybridDecoder:
    """
    SSVEP 频率-相位-空间混合解码器。

    两种模式：
    - mode='freq_only' (40 目标): 仅频率解码
    - mode='hybrid' (200 目标): 频率解码 → 空间解码级联

    Parameters
    ----------
    sample_rate : int
    freqs : list[float] — 40 个频率
    n_fixations : int — 每块注视点数 (默认 5)
    freq_model : str — 频率解码器类型 'TDCA' | 'FBCCA' | 'CCA'
    top_k : int — 频率解码保留的候选块数 (用于空间解码)
    n_harmonics : int
    data_len_sec : float
    """

    def __init__(
        self,
        sample_rate: int = 250,
        freqs: Optional[list] = None,
        n_fixations: int = 5,
        freq_model: str = "TDCA",
        top_k: int = 3,
        n_harmonics: int = 5,
        data_len_sec: float = 3.0,
        delay_sec: float = 0.14,
    ):
        self.Fs = int(sample_rate)
        self.n_fixations = int(n_fixations)
        self.top_k = int(top_k)

        # 默认 40 个频率
        if freqs is None:
            self.freqs = [8.0 + i * 0.2 for i in range(40)]
        else:
            self.freqs = [float(f) for f in freqs]
        self.Nf = len(self.freqs)
        self.Nt = self.Nf * self.n_fixations  # 总目标数

        self.freq_model_name = str(freq_model).upper()
        self._freq_decoder = None
        self._spatial_decoder = SpatialDecoder(
            sample_rate=sample_rate, n_fixations=n_fixations
        )

        # 频率解码器参数
        self._n_harmonics = int(n_harmonics)
        self._data_len_sec = float(data_len_sec)
        self._delay_sec = float(delay_sec)

        self._is_fitted = False
        self.mode = "freq_only"  # 'freq_only' | 'hybrid'

    # ── 工厂方法 ────────────────────────────────────────────────

    def _make_freq_decoder(self):
        """创建频率解码器实例。"""
        kwargs = dict(
            num_harmonics=self._n_harmonics,
            times=self._data_len_sec,
            targets=self.freqs,
            Nh=8,
            sample_rate=self.Fs,
            delay_sec=self._delay_sec,
        )
        if self.freq_model_name == "TDCA":
            return TDCA(**kwargs)
        elif self.freq_model_name == "FBCCA":
            return FBCCA(**kwargs)
        elif self.freq_model_name == "CCA":
            return CCA(**kwargs)
        else:
            raise ValueError(f"Unknown freq model: {self.freq_model_name}")

    # ── 训练 ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y_target: np.ndarray):
        """
        训练混合解码器。

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples)
        y_target : (n_trials,) — 全局目标 ID (0 ~ Nf*n_fixations-1)
        """
        X = np.asarray(X, dtype=float)
        y_target = np.asarray(y_target, dtype=int).reshape(-1)

        # 从目标 ID 反推块索引和注视点索引
        y_block = y_target // self.n_fixations
        y_fp = y_target % self.n_fixations

        # ── 训练频率解码器 ──
        self._freq_decoder = self._make_freq_decoder()
        if self.freq_model_name == "TDCA":
            self._freq_decoder.fit(X, y_block)
        # FBCCA 和 CCA 无需训练 (无监督)

        # ── 训练空间解码器 ──
        if self.mode == "hybrid":
            self._spatial_decoder.fit(X, y_block, y_fp)
            if self._spatial_decoder.is_fitted:
                print(f"[HybridDecoder] Spatial models trained for "
                      f"{len(self._spatial_decoder._models)} blocks")
            else:
                print("[HybridDecoder] WARNING: Spatial decoder failed to fit, "
                      "falling back to freq_only mode")
                self.mode = "freq_only"

        self._is_fitted = True
        return self

    # ── 打分 ────────────────────────────────────────────────────

    def score_vector(self, test_data: np.ndarray) -> np.ndarray:
        """
        返回 200 类（或 40 类）的得分向量。

        Parameters
        ----------
        test_data : (n_channels, n_samples)

        Returns
        -------
        scores : (Nf * n_fixations,) — 全局目标得分
        """
        test_data = np.asarray(test_data, dtype=float)

        # ── 频率解码: 40 类得分 ──
        freq_scores = self._freq_decoder.score_vector(test_data)

        if self.mode != "hybrid" or not self._spatial_decoder.is_fitted:
            # 仅频率模式: 每个块的 5 个注视点得分相同
            hybrid_scores = np.zeros(self.Nt, dtype=float)
            for blk in range(self.Nf):
                base = blk * self.n_fixations
                hybrid_scores[base:base + self.n_fixations] = freq_scores[blk]
            return hybrid_scores

        # ── 混合模式: 频率 Top-K → 空间解码 ──
        k = min(self.top_k, self.Nf)
        top_blocks = np.argsort(freq_scores)[-k:][::-1]

        hybrid_scores = np.zeros(self.Nt, dtype=float)
        for blk in range(self.Nf):
            base = blk * self.n_fixations
            if blk in top_blocks:
                # 空间解码得分
                spatial_scores = self._spatial_decoder.predict_block(
                    test_data, blk
                )
                # 融合: freq_score * spatial_score (乘积融合)
                for fp in range(self.n_fixations):
                    hybrid_scores[base + fp] = (
                        freq_scores[blk] * max(0.0, spatial_scores[fp])
                    )
            else:
                # 非 Top-K 块: 仅频率得分 (所有注视点相同)
                hybrid_scores[base:base + self.n_fixations] = freq_scores[blk] * 0.5

        return hybrid_scores

    def classify_with_scores(self, test_data: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """分类并返回 (target_id, scores, confidence)。"""
        scores = self.score_vector(test_data)
        result = int(np.argmax(scores))
        if scores.size >= 2:
            top2 = np.partition(scores, -2)[-2:]
            confidence = float(top2[1] - top2[0])
        else:
            confidence = float(scores[result])
        return result, scores, confidence

    def classify(self, test_data: np.ndarray) -> int:
        result, _, _ = self.classify_with_scores(test_data)
        return result

    def classify_block(self, test_data: np.ndarray) -> int:
        """仅频率分类，返回块索引 0-39。"""
        scores = self._freq_decoder.score_vector(test_data)
        return int(np.argmax(scores))

    # ── 属性 ────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_targets(self) -> int:
        return self.Nf if self.mode == "freq_only" else self.Nt

    def set_mode(self, mode: str):
        if mode in ("freq_only", "hybrid"):
            self.mode = mode

    def clear(self):
        self._freq_decoder = None
        self._spatial_decoder.clear()
        self._is_fitted = False

    # ── 权重管理 ────────────────────────────────────────────────

    def set_frequency_weights(self, weights: np.ndarray):
        if self._freq_decoder and hasattr(self._freq_decoder, "set_frequency_weights"):
            self._freq_decoder.set_frequency_weights(weights)

    def get_frequency_weights(self) -> np.ndarray:
        if self._freq_decoder and hasattr(self._freq_decoder, "get_frequency_weights"):
            return self._freq_decoder.get_frequency_weights()
        return np.ones(self.Nf, dtype=float)


# ═══════════════════════════════════════════════════════════════════
#  简易测试
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    # 模拟 40 块 × 5 注视点, 8 导, 250Hz, 3s 窗口
    n_channels, n_samples = 8, 750  # 3s × 250Hz
    n_blocks = 40
    n_fps = 5
    n_trials_per = 5  # 每类 5 trial
    freqs = [8.0 + i * 0.2 for i in range(n_blocks)]

    X_list, y_list = [], []
    for blk in range(n_blocks):
        for fp in range(n_fps):
            target_id = blk * n_fps + fp
            for _ in range(n_trials_per):
                # 不同频率 → 不同正弦信号
                t = np.arange(n_samples) / 250.0
                base = np.sin(2 * np.pi * freqs[blk] * t)
                # 不同注视点 → 不同通道权重
                weights = np.random.randn(n_channels) * (fp + 1) * 0.3
                data = weights[:, None] * base[None, :]
                data += np.random.randn(n_channels, n_samples) * 0.3
                X_list.append(data)
                y_list.append(target_id)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    # 频率模式测试
    decoder = HybridDecoder(
        sample_rate=250, freqs=freqs, n_fixations=5,
        freq_model="FBCCA", mode="freq_only",
        n_harmonics=3, data_len_sec=3.0,
    )
    decoder.fit(X, y)

    correct = 0
    for i in range(len(X)):
        pred = decoder.classify_block(X[i])
        true_block = y[i] // n_fps
        if pred == true_block:
            correct += 1
    print(f"Freq-only block accuracy: {correct / len(X):.4f}")

    # 混合模式测试
    decoder2 = HybridDecoder(
        sample_rate=250, freqs=freqs, n_fixations=5,
        freq_model="TDCA", mode="hybrid",
        n_harmonics=3, data_len_sec=3.0,
    )
    decoder2.fit(X, y)
    print(f"Hybrid decoder fitted: {decoder2.is_fitted}, mode={decoder2.mode}")
