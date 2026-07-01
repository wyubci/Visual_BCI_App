"""
SSVEP 混合编码刺激画布 — Qt 原生渲染
=============================================
40 个闪烁块 (5×8 矩阵)，频率-相位联合编码。
每个块内含 5 个十字形注视点，共 200 个可区分目标。

渲染方式：正弦调制 L(t) = 0.5×(1 + sin(2π·f·t + φ))
帧同步通过 QTimer(PreciseTimer) + 屏幕刷新率驱动。

设计参考：高小榕团队 Cyborg and Bionic Systems (2024)
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics,
    QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QApplication


# ═══════════════════════════════════════════════════════════════════
#  配置常量
# ═══════════════════════════════════════════════════════════════════

class GridConfig:
    """SSVEP 网格刺激的全部可配置参数。"""

    # 网格 (3×3 = 9 块: 中心停止 + 八方向移动)
    N_ROWS = 3
    N_COLS = 3
    N_BLOCKS = N_ROWS * N_COLS  # 9
    N_FIXATIONS = 1              # 每块注视点数 (9 块已足够区分)
    N_TARGETS = N_BLOCKS * N_FIXATIONS  # 9

    # 频率 (Hz) — 9 个，间隔 ~1.0 Hz 确保充分区分
    FREQ_START = 8.0
    FREQ_END = 15.8
    FREQ_STEP = 0.2  # 实际频率由 config 指定，此为参考步长

    # 相位 (× π)
    PHASE_STEP = 0.35
    N_PHASES = 3  # 3×3 网格用 3 种相位即可 (棋盘格)

    # 注视点方向 (相对于块中心的归一化方向) — 仅中心
    FIXATION_DIRECTIONS = (
        (0, 0),     # 0: 中心（唯一注视点）
    )

    # 颜色 (0-255)
    BG_COLOR = (0, 0, 0)               # 纯黑背景
    BLOCK_BORDER = (60, 60, 60)        # 块边框
    FIXATION_COLOR = (220, 60, 60)     # 注视十字 (红)
    CUE_COLOR = (60, 220, 60)          # 提示高亮 (绿)
    FLICKER_HIGH = (255, 255, 255)     # 闪烁峰值 (白)


# 在类外初始化 PHASE_VALUES（避免 Python 类体内列表推导式作用域问题）
GridConfig.PHASE_VALUES = [
    k * GridConfig.PHASE_STEP * math.pi for k in range(GridConfig.N_PHASES)
]


def _detect_screen_refresh_rate(default_hz: float = 60.0) -> float:
    """检测主屏幕刷新率。"""
    try:
        screens = QApplication.screens()
        if not screens:
            return default_hz
        # 偏好 60 Hz 附近的屏幕
        for scr in screens:
            r = float(scr.refreshRate())
            if 55.0 <= r <= 75.0 and 30.0 <= r <= 360.0:
                return r
        r = float(screens[0].refreshRate())
        if 30.0 <= r <= 360.0:
            return r
    except Exception:
        pass
    return default_hz


# ═══════════════════════════════════════════════════════════════════
#  闪烁块数据对象
# ═══════════════════════════════════════════════════════════════════

class FlickerBlock:
    """单个 SSVEP 闪烁块的状态（非 Qt widget，纯数据+绘制）。"""

    __slots__ = (
        "block_idx", "row", "col", "freq_hz", "phase_rad",
        "rect", "current_lum", "fixation_positions", "cued_fp_idx",
    )

    def __init__(self, block_idx: int, row: int, col: int,
                 freq_hz: float, phase_rad: float):
        self.block_idx = block_idx
        self.row = row
        self.col = col
        self.freq_hz = freq_hz
        self.phase_rad = phase_rad
        self.rect = QRectF()
        self.current_lum = 0.5  # 初始 50% 灰
        self.fixation_positions: List[Tuple[float, float]] = []
        self.cued_fp_idx: int = -1  # -1 表示无提示

    @property
    def target_ids(self) -> range:
        """该块对应的 5 个全局目标 ID 范围。"""
        start = self.block_idx * GridConfig.N_FIXATIONS
        return range(start, start + GridConfig.N_FIXATIONS)


# ═══════════════════════════════════════════════════════════════════
#  SSVEP 网格画布 (QWidget)
# ═══════════════════════════════════════════════════════════════════

class SSVEPGridCanvas(QWidget):
    """SSVEP 混合编码刺激画布。

    Parameters
    ----------
    parent : QWidget
    target_count : 40 或 200
        40 = 仅频率-相位编码；200 = 频率-相位-空间混合编码。
    """

    # 回调信号可以用 Qt Signal，这里用简单回调保持解耦
    on_stim_started = None   # callable()
    on_stim_stopped = None   # callable()

    def __init__(self, parent=None, target_count: int = 40):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        # 确保有最小尺寸，防止布局时被压缩到 0
        self.setMinimumSize(200, 150)

        self._target_count = int(target_count)
        self._show_fixation_crosses = (self._target_count >= 200)

        # ── 频率 & 相位 ──
        cfg = GridConfig
        self._freqs: List[float] = [
            cfg.FREQ_START + i * cfg.FREQ_STEP
            for i in range(cfg.N_BLOCKS)
        ]
        self._phases: List[float] = []
        for row in range(cfg.N_ROWS):
            for col in range(cfg.N_COLS):
                phase_idx = (row + col) % cfg.N_PHASES
                self._phases.append(cfg.PHASE_VALUES[phase_idx])

        # ── 创建闪烁块 ──
        self._blocks: List[FlickerBlock] = []
        for i in range(cfg.N_BLOCKS):
            row, col = divmod(i, cfg.N_COLS)
            blk = FlickerBlock(i, row, col, self._freqs[i], self._phases[i])
            self._blocks.append(blk)

        # ── 渲染状态 ──
        self._refresh_hz = _detect_screen_refresh_rate()
        self._frame_interval_ms = max(1, int(1000.0 / self._refresh_hz))
        self._is_flickering = False
        self._t_start: float = 0.0
        self._frame_index: int = 0
        self._last_render_frame: int = -1
        self._layout_done = False

        # ── 计时器 ──
        self._stim_timer = QTimer(self)
        self._stim_timer.setTimerType(Qt.PreciseTimer)
        self._stim_timer.setInterval(self._frame_interval_ms)
        self._stim_timer.timeout.connect(self._on_timer_tick)

        # ── 高 DPI ──
        self._dpr = self.devicePixelRatioF()

    # ── 属性 ────────────────────────────────────────────────────

    @property
    def target_count(self) -> int:
        return self._target_count

    @property
    def is_flickering(self) -> bool:
        return self._is_flickering

    @property
    def refresh_hz(self) -> float:
        return self._refresh_hz

    @property
    def freqs(self) -> np.ndarray:
        return np.asarray(self._freqs, dtype=float)

    @property
    def phases(self) -> np.ndarray:
        return np.asarray(self._phases, dtype=float)

    def block_freq(self, block_idx: int) -> float:
        return self._freqs[block_idx]

    def block_phase(self, block_idx: int) -> float:
        return self._phases[block_idx]

    def global_target_id(self, block_idx: int, fp_idx: int) -> int:
        return block_idx * GridConfig.N_FIXATIONS + fp_idx

    # ── 布局计算 ───────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not self._layout_done:
            self._recalc_layout()
            self._layout_done = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalc_layout()

    def _recalc_layout(self):
        """根据窗口大小重新计算所有块和注视点的像素位置。"""
        cfg = GridConfig
        w = max(1, self.width())
        h = max(1, self.height())

        if w <= 1 and h <= 1:
            return  # 尚未获得有效尺寸

        # 块大小自适应：留 2% 边距
        margin = max(4, int(min(w, h) * 0.02))
        avail_w = w - 2 * margin
        avail_h = h - 2 * margin

        # 按行列数均分，保留间隙
        gap_px = max(2, int(min(avail_w / (cfg.N_COLS * 6), avail_h / (cfg.N_ROWS * 6))))
        block_w = (avail_w - gap_px * (cfg.N_COLS - 1)) / cfg.N_COLS
        block_h = (avail_h - gap_px * (cfg.N_ROWS - 1)) / cfg.N_ROWS
        block_size = min(block_w, block_h)

        if block_size <= 0:
            return  # 布局空间不足

        total_w = cfg.N_COLS * block_size + (cfg.N_COLS - 1) * gap_px
        total_h = cfg.N_ROWS * block_size + (cfg.N_ROWS - 1) * gap_px
        start_x = (w - total_w) / 2.0
        start_y = (h - total_h) / 2.0

        for blk in self._blocks:
            x = start_x + blk.col * (block_size + gap_px)
            y = start_y + blk.row * (block_size + gap_px)
            blk.rect = QRectF(x, y, block_size, block_size)

            # 注视点位置 (块内偏移)
            if self._show_fixation_crosses:
                cx = x + block_size / 2.0
                cy = y + block_size / 2.0
                offset = block_size * 0.22  # 注视点偏移 ~22% 块半边长
                blk.fixation_positions = [
                    (cx + dx * offset, cy + dy * offset)
                    for dx, dy in cfg.FIXATION_DIRECTIONS
                ]

        self._layout_done = True

    # ── 闪烁控制 ───────────────────────────────────────────────

    def start_flicker(self):
        """开始闪烁。"""
        if self._is_flickering:
            return
        self._is_flickering = True
        self._t_start = time.perf_counter()
        self._frame_index = 0
        self._last_render_frame = -1
        self._stim_timer.start()
        if self.on_stim_started:
            self.on_stim_started()

    def stop_flicker(self):
        """停止闪烁，恢复默认灰色（块仍可见）。"""
        self._is_flickering = False
        self._stim_timer.stop()
        for blk in self._blocks:
            blk.current_lum = 0.5  # 恢复灰色可见状态
        self.update()
        if self.on_stim_stopped:
            self.on_stim_stopped()

    def reset(self):
        """停止并清空提示。"""
        self.stop_flicker()
        for blk in self._blocks:
            blk.cued_fp_idx = -1

    # ── 提示控制 ───────────────────────────────────────────────

    def cue_target(self, block_idx: int, fp_idx: int = 0):
        """高亮指定目标的注视点（绿色）。传 -1 取消。"""
        for blk in self._blocks:
            blk.cued_fp_idx = -1
        if 0 <= block_idx < GridConfig.N_BLOCKS:
            self._blocks[block_idx].cued_fp_idx = fp_idx
        self.update()

    def clear_cue(self):
        self.cue_target(-1)

    # ── 帧更新 (计时器回调) ────────────────────────────────────

    def _on_timer_tick(self):
        if not self._is_flickering:
            return
        now = time.perf_counter()
        elapsed = now - self._t_start
        # 基于时间而非帧计数，避免累积漂移
        self._frame_index = int(elapsed * self._refresh_hz)

        if self._frame_index == self._last_render_frame:
            return
        self._last_render_frame = self._frame_index

        t_sec = self._frame_index / self._refresh_hz
        for blk in self._blocks:
            lum = 0.5 * (1.0 + math.sin(
                2.0 * math.pi * blk.freq_hz * t_sec + blk.phase_rad
            ))
            blk.current_lum = float(lum)
        self.update()

    def _current_time_sec(self) -> float:
        if self._is_flickering:
            return self._frame_index / self._refresh_hz
        return 0.0

    # ── 绘制 ───────────────────────────────────────────────────

    def paintEvent(self, event):
        # 如果布局尚未完成（首次显示前的 race condition），强制执行一次
        if not self._layout_done and self.width() > 1 and self.height() > 1:
            self._recalc_layout()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 背景
        painter.fillRect(self.rect(), QColor(*GridConfig.BG_COLOR))

        cfg = GridConfig

        for blk in self._blocks:
            rect = blk.rect
            if rect.isEmpty():
                continue

            # ── 闪烁块主体 ──
            lum = blk.current_lum
            gray = int(lum * 255)
            block_color = QColor(gray, gray, gray)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(block_color))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

            # ── 边框 ──
            border = QColor(*cfg.BLOCK_BORDER)
            pen = QPen(border, max(1, int(rect.width() * 0.008)))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

            # ── 注视十字 ──
            if self._show_fixation_crosses and blk.fixation_positions:
                cross_len = max(2.0, rect.width() * 0.06)
                cross_w = max(1.0, cross_len * 0.22)

                for fp_idx, (fx, fy) in enumerate(blk.fixation_positions):
                    # 提示高亮颜色
                    is_cued = (blk.cued_fp_idx == fp_idx)
                    cross_color = QColor(*cfg.CUE_COLOR) if is_cued else QColor(*cfg.FIXATION_COLOR)

                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(cross_color))
                    # 水平臂
                    painter.drawRect(QRectF(
                        fx - cross_len / 2, fy - cross_w / 2,
                        cross_len, cross_w,
                    ))
                    # 垂直臂
                    painter.drawRect(QRectF(
                        fx - cross_w / 2, fy - cross_len / 2,
                        cross_w, cross_len,
                    ))

        painter.end()

    # ── 信息查询 ───────────────────────────────────────────────

    def target_info(self, target_id: int) -> dict:
        """返回目标 ID 对应的完整信息。"""
        block_idx = target_id // GridConfig.N_FIXATIONS
        fp_idx = target_id % GridConfig.N_FIXATIONS
        if 0 <= block_idx < GridConfig.N_BLOCKS:
            blk = self._blocks[block_idx]
            return {
                "target_id": target_id,
                "block_idx": block_idx,
                "fp_idx": fp_idx,
                "freq_hz": blk.freq_hz,
                "phase_rad": blk.phase_rad,
                "row": blk.row,
                "col": blk.col,
            }
        return {}


# ═══════════════════════════════════════════════════════════════════
#  独立测试入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    win = QWidget()
    win.setWindowTitle("SSVEP Grid Canvas — Standalone Test")
    win.resize(1707, 960)
    win.setStyleSheet("background: #000;")

    from PyQt5.QtWidgets import QVBoxLayout
    layout = QVBoxLayout(win)
    layout.setContentsMargins(0, 0, 0, 0)

    canvas = SSVEPGridCanvas(target_count=40)
    layout.addWidget(canvas)

    # 按空格开始/暂停
    def on_key(e):
        if e.key() == Qt.Key_Space:
            if canvas.is_flickering:
                canvas.stop_flicker()
                print("[STOP]")
            else:
                canvas.start_flicker()
                print("[START]")
        elif e.key() == Qt.Key_C:
            # 随机提示一个目标
            import random
            bid = random.randint(0, 39)
            fid = random.randint(0, 4) if canvas._show_fixation_crosses else 0
            canvas.cue_target(bid, fid)
            print(f"[CUE] Block {bid+1}, FP {fid+1}")
        elif e.key() == Qt.Key_Escape:
            win.close()

    win.keyPressEvent = on_key
    win.setFocusPolicy(Qt.StrongFocus)

    win.show()
    sys.exit(app.exec_())
