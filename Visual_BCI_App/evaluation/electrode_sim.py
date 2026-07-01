"""
电极密度模拟器 — 从完整蒙太奇中选取通道子集
==================================================

模拟不同电极密度的解码性能：
  - 8ch: 默认 NeuroDance 8 导 (全部)
  - 9ch: 传统 Pz/POz/Oz/PO3-6/O1/O2 配置
  - 21ch: 64 导帽的顶枕区 21 电极 (间距 ~2.8 cm)
  - 32ch: 128 导帽的顶枕区 32 电极 (间距 ~2.0 cm)
  - 66ch: 256 导帽的顶枕区 66 电极 (间距 ~1.5 cm)

对于仅有 8 导的实际系统，本模块主要用于：
  1. 验证不同通道数的理论增益
  2. 仿真实验设计参考
  3. 未来扩展预留

参考: Liao et al. Cyborg Bionic Syst, 2024
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════
#  预设电极蒙太奇
# ═══════════════════════════════════════════════════════════════════

# 实际 NeuroDance 8 导
MONTAGE_8CH = [0, 1, 2, 3, 4, 5, 6, 7]

# 传统 9 电极 (国际 10-20 顶枕区)
# 当只有 8 导时，9ch 模拟为保留全部 8 导
MONTAGE_9CH = [0, 1, 2, 3, 4, 5, 6, 7]

# 标准蒙太奇字典
PRESET_MONTAGES: Dict[str, List[int]] = {
    "8ch": MONTAGE_8CH,
    "9ch": MONTAGE_9CH,
}


class ElectrodeSimulator:
    """电极密度模拟器。

    Parameters
    ----------
    n_total_channels : 总通道数 (如 8)
    preset : 预设蒙太奇名称 '8ch' | '9ch'
    """

    def __init__(self, n_total_channels: int = 8,
                 preset: str = "8ch"):
        self.n_total = int(n_total_channels)
        self.preset = str(preset)

        # 可用蒙太奇
        self.montages = dict(PRESET_MONTAGES)

        # 自定义蒙太奇
        self._custom: Dict[str, List[int]] = {}

    def available_montages(self) -> List[str]:
        """返回可用的蒙太奇名称列表。"""
        names = list(self.montages.keys())
        names.extend(self._custom.keys())
        return sorted(set(names))

    def get_channels(self, name: str) -> np.ndarray:
        """获取指定蒙太奇的通道索引。"""
        if name in self._custom:
            channels = self._custom[name]
        elif name in self.montages:
            channels = self.montages[name]
        else:
            raise KeyError(f"Unknown montage: {name}")

        # 确保索引不超出实际通道数
        valid = [c for c in channels if 0 <= c < self.n_total]
        if not valid:
            valid = list(range(self.n_total))
        return np.asarray(valid, dtype=int)

    def add_custom(self, name: str, channels: List[int]):
        """添加自定义蒙太奇。"""
        self._custom[name] = [int(c) for c in channels]

    def simulate(self, X: np.ndarray, montage: str) -> np.ndarray:
        """从完整数据中提取指定蒙太奇的通道子集。

        Parameters
        ----------
        X : (..., n_channels, n_samples)
        montage : 蒙太奇名称

        Returns
        -------
        X_sub : (..., n_selected_channels, n_samples)
        """
        X = np.asarray(X, dtype=float)
        channels = self.get_channels(montage)
        return X[..., channels, :]

    def benchmark_all(self, X: np.ndarray, y: np.ndarray,
                      eval_func) -> Dict[str, dict]:
        """在所有可用蒙太奇上评估。

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples)
        y : (n_trials,)
        eval_func : callable(X_sub, y) → dict of metrics

        Returns
        -------
        results : {montage_name: {metric: value}}
        """
        results = {}
        for name in self.available_montages():
            try:
                X_sub = self.simulate(X, name)
                metrics = eval_func(X_sub, y)
                results[name] = metrics
            except Exception as exc:
                results[name] = {"error": str(exc)}
        return results


# ═══════════════════════════════════════════════════════════════════
#  简易测试
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sim = ElectrodeSimulator(n_total_channels=8)
    print("Available montages:", sim.available_montages())

    # 模拟数据
    X = np.random.randn(20, 8, 250)
    y = np.random.randint(0, 40, 20)

    for name in sim.available_montages():
        X_sub = sim.simulate(X, name)
        ch = sim.get_channels(name)
        print(f"  {name}: channels={ch.tolist()}, shape={X_sub.shape}")
