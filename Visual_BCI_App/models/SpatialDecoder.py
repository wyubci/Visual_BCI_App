"""
空间解码器 — 同一闪烁块内 5 注视点分类
=============================================

当用户注视同一闪烁块内的不同注视点时，闪烁块的频率和相位完全相同，
但刺激在视网膜上的落点位置不同，诱发的 SSVEP 具有不同的空间分布特征
（幅度侧化方向和相位分布）。

本解码器基于 DSP (Discriminative Spatial Pattern) + TRCA 融合，
专用于同一频率块内的 5 类注视点分类。

适用场景：
  - 8 导系统：精度有限，建议使用 2-3 个注视点（中心 + 左右）
  - 32+ 导系统：5 类全部分类可达较高精度

参考文献：
  Liao et al., "Hybrid frequency-phase-spatial encoding", Cyborg Bionic Syst, 2024
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import eigh


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def _nearest_pd(mat: np.ndarray) -> np.ndarray:
    """强制矩阵为半正定。"""
    b = (mat + mat.T) / 2.0
    _, s, v = np.linalg.svd(b)
    h = v.T @ np.diag(s) @ v
    a2 = (b + h) / 2.0
    a3 = (a2 + a2.T) / 2.0
    try:
        np.linalg.cholesky(a3)
        return a3
    except np.linalg.LinAlgError:
        pass

    spacing = np.spacing(np.linalg.norm(mat))
    eye = np.eye(mat.shape[0])
    k = 1
    while True:
        try:
            np.linalg.cholesky(a3)
            return a3
        except np.linalg.LinAlgError:
            min_eig = np.min(np.real(np.linalg.eigvals(a3)))
            a3 += eye * (-min_eig * (k ** 2) + spacing)
            k += 1
            if k > 100:
                return a3


def _design_bandpass(fs: float, low: float = 4.0, high: float = 45.0,
                     order: int = 5) -> tuple:
    """设计 Butterworth 带通滤波器。"""
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def _filter_data(data: np.ndarray, b, a) -> np.ndarray:
    """零相位带通滤波 [channels, samples] 或 [trials, channels, samples]."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        return signal.filtfilt(b, a, data, axis=-1)
    elif data.ndim == 3:
        return np.array([signal.filtfilt(b, a, d, axis=-1) for d in data])
    return data


# ═══════════════════════════════════════════════════════════════════
#  DSP 空间滤波器 (Discriminative Spatial Pattern)
# ═══════════════════════════════════════════════════════════════════

def _dsp_train(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    训练 DSP 空间滤波器。

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples)
    y : (n_trials,) — 类标签 (0 ~ n_classes-1)

    Returns
    -------
    W : (n_channels, n_channels) — 空间滤波器矩阵
    templates : (n_classes, n_channels, n_samples) — 每类均值模板
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    classes = np.unique(y)
    n_classes = len(classes)
    n_trials, n_channels, n_samples = X.shape

    # 去均值
    X_centered = X - np.mean(X, axis=-1, keepdims=True)

    # 类内散布矩阵 Sw
    Sw = np.zeros((n_channels, n_channels), dtype=float)
    class_means = []
    for c in classes:
        Xc = X_centered[y == c]
        Mc = np.mean(Xc, axis=0)  # (n_channels, n_samples)
        class_means.append(Mc)
        for trial in Xc:
            Sw += trial @ trial.T

    # 类间散布矩阵 Sb
    grand_mean = np.mean(X_centered, axis=0)
    Sb = np.zeros((n_channels, n_channels), dtype=float)
    for c_idx, c in enumerate(classes):
        Mc = class_means[c_idx]
        n_c = np.sum(y == c)
        diff = Mc - grand_mean
        Sb += n_c * (diff @ diff.T)

    Sw = _nearest_pd(Sw)
    Sb = _nearest_pd(Sb)

    # 广义特征值分解
    eigenvalues, eigenvectors = eigh(Sb, Sw)
    # 降序排列
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx]

    # 每类模板
    templates = np.stack(class_means, axis=0)

    return W, templates


def _dsp_score(sample: np.ndarray, W: np.ndarray, templates: np.ndarray,
               n_components: int = 3) -> np.ndarray:
    """
    用 DSP 滤波器打分。

    Parameters
    ----------
    sample : (n_channels, n_samples)
    W : (n_channels, n_components)
    templates : (n_classes, n_channels, n_samples)

    Returns
    -------
    scores : (n_classes,) — 每类的相关系数
    """
    sample = np.asarray(sample, dtype=float)
    n_classes = templates.shape[0]
    n_comp = min(n_components, W.shape[1])
    W_use = W[:, :n_comp]

    # 投影
    proj_sample = W_use.T @ sample  # (n_components, n_samples)
    scores = np.zeros(n_classes, dtype=float)

    for c in range(n_classes):
        proj_template = W_use.T @ templates[c]  # (n_components, n_samples)
        # Pearson 相关系数
        a = proj_sample.ravel()
        b = proj_template.ravel()
        a -= np.mean(a)
        b -= np.mean(b)
        denom = np.sqrt(float(a @ a) * float(b @ b))
        if denom > 1e-12:
            scores[c] = float((a @ b) / denom)
        else:
            scores[c] = 0.0

    return scores


# ═══════════════════════════════════════════════════════════════════
#  SpatialDecoder 主类
# ═══════════════════════════════════════════════════════════════════

class SpatialDecoder:
    """
    同一频率闪烁块内的注视点空间解码器。

    对每个闪烁块单独训练一个 DSP 模型，
    输入该频率的 EEG 数据，输出 5 类注视点得分。

    Parameters
    ----------
    sample_rate : int
    n_fixations : int — 注视点数量 (默认 5)
    n_components : int — DSP 保留成分数
    """

    def __init__(self, sample_rate: int = 250, n_fixations: int = 5,
                 n_components: int = 3):
        self.Fs = int(sample_rate)
        self.n_fixations = int(n_fixations)
        self.n_components = int(n_components)

        # 滤波器
        self._b, self._a = _design_bandpass(self.Fs)

        # 每个块一个模型
        self._models: dict = {}  # {block_idx: (W, templates)}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray, y_block: np.ndarray, y_fp: np.ndarray):
        """
        训练所有闪烁块的空间解码器。

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples)
        y_block : (n_trials,) — 块索引 (0 ~ 39)
        y_fp : (n_trials,) — 注视点索引 (0 ~ 4)
        """
        X = np.asarray(X, dtype=float)
        y_block = np.asarray(y_block, dtype=int).reshape(-1)
        y_fp = np.asarray(y_fp, dtype=int).reshape(-1)

        # 滤波
        X_filt = _filter_data(X, self._b, self._a)

        unique_blocks = np.unique(y_block)
        self._models = {}

        for blk in unique_blocks:
            mask = y_block == blk
            X_blk = X_filt[mask]
            y_blk = y_fp[mask]

            # 至少有 2 类 × 每类 2 trial 才训练
            unique_fps = np.unique(y_blk)
            if len(unique_fps) < 2:
                continue
            class_counts = [np.sum(y_blk == fp) for fp in unique_fps]
            if min(class_counts) < 2:
                continue

            try:
                W, templates = _dsp_train(X_blk, y_blk)
                self._models[int(blk)] = (W, templates)
            except Exception:
                continue

        self._is_fitted = len(self._models) > 0
        return self

    def predict_block(self, sample: np.ndarray, block_idx: int) -> np.ndarray:
        """
        对单个样本预测指定块内的注视点得分。

        Parameters
        ----------
        sample : (n_channels, n_samples)
        block_idx : int

        Returns
        -------
        scores : (n_fixations,) — 每类得分
        """
        if block_idx not in self._models:
            return np.zeros(self.n_fixations, dtype=float)

        sample = np.asarray(sample, dtype=float)
        sample_filt = signal.filtfilt(self._b, self._a, sample, axis=-1)
        W, templates = self._models[block_idx]
        return _dsp_score(sample_filt, W, templates, self.n_components)

    def predict(self, sample: np.ndarray, block_idx: int) -> int:
        """返回预测的注视点索引。"""
        scores = self.predict_block(sample, block_idx)
        if np.all(scores == 0):
            return 0
        return int(np.argmax(scores))

    def clear(self):
        self._models = {}
        self._is_fitted = False


# ═══════════════════════════════════════════════════════════════════
#  简易测试
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 生成模拟数据: 2 个块，每块 3 类注视点，8 导，250 采样点
    np.random.seed(42)
    n_trials_per = 20
    n_channels, n_samples = 8, 250
    blocks = [0, 5]
    fps = [0, 1, 2]

    X_list, y_block_list, y_fp_list = [], [], []
    for blk in blocks:
        for fp in fps:
            # 不同注视点产生不同的空间模式（不同通道权重）
            pattern = np.random.randn(n_channels) * (fp + 1) * 0.5
            for _ in range(n_trials_per):
                noise = np.random.randn(n_channels, n_samples) * 0.5
                signal_component = pattern[:, None] * np.sin(
                    2 * np.pi * (10 + blk * 0.5) * np.arange(n_samples) / 250
                )[None, :]
                X_list.append(signal_component + noise)
                y_block_list.append(blk)
                y_fp_list.append(fp)

    X = np.stack(X_list, axis=0)
    y_block = np.array(y_block_list)
    y_fp = np.array(y_fp_list)

    decoder = SpatialDecoder(sample_rate=250, n_fixations=5)
    decoder.fit(X, y_block, y_fp)
    print(f"Fitted: {decoder.is_fitted}, models: {list(decoder._models.keys())}")

    # 测试
    acc = 0
    for i in range(len(X)):
        pred = decoder.predict(X[i], y_block[i])
        if pred == y_fp[i]:
            acc += 1
    print(f"Train accuracy: {acc / len(X):.4f}")
