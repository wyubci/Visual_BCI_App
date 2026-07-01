"""
脑控小车控制页面 v2
====================
SSVEP 40 块闪烁刺激 → 选择其中 5~8 个块映射到小车命令。
所有 40 块都闪烁（维持 SSVEP 频率编码），但解码时只关注小车命令块。

布局:
  ┌──────────────────────────────────────────────────┐
  │  SSVEP 刺激网格 (5×8=40块, 全部闪烁)             │
  │  [B01 前进] [B02 后退] ...  (绿色边框=已映射)     │
  ├──────────────────────┬───────────────────────────┤
  │                      │  模式: [训练采集/在线控制]  │
  │   状态 & 进度        │  标签编辑                  │
  │                      │  训练/在线 按钮            │
  │                      │  小车连接 & 命令映射        │
  │                      │  解码结果                  │
  └──────────────────────┴───────────────────────────┘
"""

import json
import os
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime
from glob import glob
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QGroupBox, QSplitter, QGridLayout, QLineEdit, QCheckBox,
    QMessageBox, QScrollArea, QSizePolicy,
)
from scipy.io import savemat, loadmat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.FBCCA import FBCCA
from models.TDCA import TDCA
from models.CCA import CCA
from config import config as cfg_manager
from interface.car_interface.acquisition import preprocess_model_input
from interface.car_interface.training_framework import extract_target_id, extract_training_sample
from interface.car_interface.ssvep_grid_canvas import SSVEPGridCanvas, GridConfig

# ═══════════════════════════════════════════════════════════════
#  小车客户端
# ═══════════════════════════════════════════════════════════════

class RobotClient:
    def __init__(self, ip="10.186.179.92", port=65432):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(1.5)
        self.addr = (ip, port)
        self.connected = False

    def connect(self):
        try:
            self.sock.connect(self.addr)
            self.connected = True
            return True
        except:
            return False

    def send(self, cmd):
        if self.connected:
            try:
                self.sock.send(cmd.encode("utf-8"))
            except:
                self.connected = False

    def set_motor(self, v1, v2, v3, v4):
        self.send(f"{int(v1)},{int(v2)},{int(v3)},{int(v4)}")

    def move(self, cmd_idx: int):
        """执行小车命令 (0-8, 对应 3×3 网格)。

        0=↖前进左 1=↑前进 2=↗前进右
        3=←左转   4=■停止 5=→右转
        6=↙后退左 7=↓后退 8=↘后退右
        """
        if cmd_idx == 4:  # 停止
            self.set_motor(0, 0, 0, 0)
            return

        # Mecanum 四轮速度映射
        # 左前(LF), 右前(RF), 左后(LB), 右后(RB)
        motions = {
            0: ( 30,  60,  60,  30),  # ↖ 前进左
            1: ( 50,  50,  50,  50),  # ↑ 前进
            2: ( 60,  30,  30,  60),  # ↗ 前进右
            3: (-40,  40, -40,  40),  # ← 左转
            5: ( 40, -40,  40, -40),  # → 右转
            6: (-30, -60, -60, -30),  # ↙ 后退左
            7: (-50, -50, -50, -50),  # ↓ 后退
            8: (-60, -30, -30, -60),  # ↘ 后退右
        }
        if cmd_idx in motions:
            v = motions[cmd_idx]
            self.set_motor(v[0], v[1], v[2], v[3])
            time.sleep(1.2)
        self.set_motor(0, 0, 0, 0)

    def close(self):
        self.sock.close()


# ═══════════════════════════════════════════════════════════════
#  主页面
# ═══════════════════════════════════════════════════════════════

_STYLE = """
QWidget#carPage { background: #000; }
QLabel { color: #E5E7EB; font-size: 12px; }
QGroupBox {
    border: 1px solid #2F2F2F; border-radius: 8px;
    margin-top: 8px; font-weight: bold; color: #D9E2F0;
    background: #0A0A0A; padding-top: 6px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #111; border: 1px solid #333; border-radius: 6px;
    padding: 4px 8px; color: #E5E7EB; min-height: 28px;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #7CB0FF; }
QPushButton {
    color: #F7FAFF; border-radius: 8px; padding: 6px 14px; font-weight: 600;
}
QPushButton:disabled { background: #333; color: #666; border: 1px solid #444; }
QProgressBar {
    border: 1px solid #333; border-radius: 4px; background: #111;
    text-align: center; color: #E5E7EB;
}
QProgressBar::chunk { background: #2F6FD6; border-radius: 3px; }
"""


class CarControlWindowV2(QWidget):
    """脑控小车 — 40 块 SSVEP 中选 N 个控制小车。"""

    # ── 默认小车命令配置 (3×3 网格) ──
    # (命令名, 块编号, 动作描述)
    # 布局:   B01(↖)  B02(↑)  B03(↗)
    #          B04(←)  B05(■)  B06(→)
    #          B07(↙)  B08(↓)  B09(↘)
    DEFAULT_COMMANDS = [
        ("↖前进左", 0,  "Mecanum 左前轮正转 + 右后轮正转"),
        ("↑前进",   1,  "四轮同时前进"),
        ("↗前进右", 2,  "Mecanum 右前轮正转 + 左后轮正转"),
        ("←左转",   3,  "原地左转 (左轮后转, 右轮前转)"),
        ("■停止",   4,  "紧急停止, 所有电机归零"),
        ("→右转",   5,  "原地右转 (右轮后转, 左轮前转)"),
        ("↙后退左", 6,  "Mecanum 斜向后左"),
        ("↓后退",   7,  "四轮同时后退"),
        ("↘后退右", 8,  "Mecanum 斜向后右"),
    ]

    def __init__(self, objectName="carControlWindow"):
        super().__init__()
        self.setObjectName(objectName)
        self.setStyleSheet(_STYLE)

        # ── 基础配置 ──
        self.subject = getattr(cfg_manager, "subjectName", "TestSubject")
        self.sample_rate = 250
        self.sti_freqs = [float(x) for x in cfg_manager.sti_lst[:9]]
        self.n_blocks = 9
        self._active_commands = list(self.DEFAULT_COMMANDS)  # 当前命令列表

        # ── 运行时状态 ──
        self.mode = "idle"        # idle | training | online
        self.train_phase = "idle" # idle | cue | stim | rest
        self._train_block = 0
        self._train_plan: List[int] = []
        self._train_idx = 0
        self._train_done = 0

        self._decoder = None
        self._model_name = "FBCCA"
        self._decision_buf = deque(maxlen=3)

        self._online_correct = 0
        self._online_total = 0

        # 小车
        self._robot = RobotClient()
        self._robot_enabled = False

        # EEG 缓冲
        self._eeg_lock = threading.RLock()
        self._eeg_chunks: list = []
        self._eeg_points = 0
        self._eeg_stim_start = 0.0

        # 计时器
        self._phase_timer = QTimer(self)
        self._phase_timer.setTimerType(Qt.PreciseTimer)
        self._phase_timer.timeout.connect(self._phase_tick)
        self._phase_t0 = 0.0
        self._phase_dur = 0.0

        # ── 建 UI ──
        self._build_ui()

        self._clock = QTimer(self)
        self._clock.timeout.connect(self._refresh)
        self._clock.start(500)

    # ═══════════════════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════════════════

    def _btn(self, text, color, handler):
        b = QPushButton(text)
        b.clicked.connect(handler)
        b.setStyleSheet(
            f"QPushButton{{background:{color};border:1px solid #5E8EDC;}}"
            f"QPushButton:hover{{background:#3E7CE2;}}"
        )
        return b

    def _lbl(self, text, bold=False, color="#E5E7EB", size=12):
        lb = QLabel(text)
        lb.setStyleSheet(f"color:{color};font-size:{size}px;"
                         f"{'font-weight:bold;' if bold else ''}")
        return lb

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── 主体：刺激画布 ──
        self.canvas = SSVEPGridCanvas(target_count=9)
        self.canvas.setMinimumSize(500, 300)
        root.addWidget(self.canvas, 1)

        # ── 底部控制栏 ──
        bar = QWidget()
        bar.setMaximumHeight(220)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 4, 0, 0)
        bar_layout.setSpacing(8)

        # 左：状态 & 进度
        left = QVBoxLayout()
        self.status_lbl = self._lbl("就绪 — 选择模式后开始", bold=True, size=13)
        left.addWidget(self.status_lbl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left.addWidget(self.progress)
        self.sub_status = self._lbl("3×3网格 | 9块闪烁 | 8方向+1停止 = 9命令",
                                    color="#6B7280", size=10)
        left.addWidget(self.sub_status)
        bar_layout.addLayout(left, 1)

        # 中：模式 & 控制
        mid = QVBoxLayout()
        mid.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(self._lbl("模式:", bold=True))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["空闲", "训练采集", "在线控制"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        row1.addWidget(self.mode_combo)

        row1.addWidget(self._lbl("模型:", bold=True))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["FBCCA", "TDCA", "CCA"])
        row1.addWidget(self.model_combo)

        row1.addWidget(self._lbl("窗口:", bold=True))
        self.win_combo = QComboBox()
        self.win_combo.addItems(["2s", "3s", "4s"])
        self.win_combo.setCurrentText("3s")
        row1.addWidget(self.win_combo)
        mid.addLayout(row1)

        row2 = QHBoxLayout()
        self.start_btn = self._btn("▶ 开始", "#2F6FD6", self._start)
        row2.addWidget(self.start_btn)
        self.stop_btn = self._btn("■ 停止", "#AA3333", self._stop)
        self.stop_btn.setEnabled(False)
        row2.addWidget(self.stop_btn)
        row2.addWidget(self._lbl("每类采集:", bold=True))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(10)
        row2.addWidget(self.count_spin)
        mid.addLayout(row2)

        # 标签编辑
        row3 = QHBoxLayout()
        row3.addWidget(self._lbl("命令标签:", bold=True))
        self.label_edit = QLineEdit(",".join(c[0] for c in self.DEFAULT_COMMANDS))
        self.label_edit.setToolTip("逗号分隔，对应 B01~B0N。训练和在线会显示这些标签。")
        row3.addWidget(self.label_edit, 1)
        mid.addLayout(row3)
        bar_layout.addLayout(mid, 2)

        # 右：小车 & 结果
        right = QVBoxLayout()
        right.setSpacing(3)

        car_row = QHBoxLayout()
        car_row.addWidget(self._lbl("小车 IP:", bold=True))
        self.car_ip = QLineEdit("10.186.179.92")
        self.car_ip.setMaximumWidth(110)
        car_row.addWidget(self.car_ip)
        self.car_btn = self._btn("连接", "#2F6FD6", self._toggle_car)
        self.car_btn.setMaximumWidth(70)
        car_row.addWidget(self.car_btn)
        right.addLayout(car_row)

        self.car_cb = QCheckBox("解码输出到小车")
        self.car_cb.setStyleSheet("color:#E5E7EB;")
        self.car_cb.toggled.connect(lambda v: setattr(self, '_robot_enabled', v))
        right.addWidget(self.car_cb)

        self.pred_lbl = self._lbl("当前命令: —", bold=True, color="#7CB0FF", size=16)
        right.addWidget(self.pred_lbl)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        right.addWidget(self.conf_bar)

        self.conf_lbl = self._lbl("置信度: 0.000", color="#9FB0C9", size=10)
        right.addWidget(self.conf_lbl)

        self.acc_lbl = self._lbl("准确率: —", color="#9FB0C9", size=10)
        right.addWidget(self.acc_lbl)

        self.net_lbl = self._lbl("小车: 未连接", color="#EF4444", size=10)
        right.addWidget(self.net_lbl)
        bar_layout.addLayout(right, 1)

        root.addWidget(bar)

    # ═══════════════════════════════════════════════════════════
    #  模式切换
    # ═══════════════════════════════════════════════════════════

    def _on_mode(self, idx):
        modes = ["idle", "training", "online"]
        self.mode = modes[idx]
        self.canvas.stop_flicker()
        self.start_btn.setEnabled(idx > 0)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(0)

    # ═══════════════════════════════════════════════════════════
    #  解析标签
    # ═══════════════════════════════════════════════════════════

    def _parse_labels(self) -> List[str]:
        text = self.label_edit.text().strip()
        if not text:
            return [c[0] for c in self.DEFAULT_COMMANDS]
        labels = [x.strip() for x in text.split(",") if x.strip()]
        return labels

    def _car_blocks(self) -> List[int]:
        """获取映射到小车命令的块编号列表。"""
        labels = self._parse_labels()
        return list(range(len(labels)))  # B01~B0N

    # ═══════════════════════════════════════════════════════════
    #  开始 / 停止
    # ═══════════════════════════════════════════════════════════

    def _start(self):
        if self.mode == "training":
            self._start_training()
        elif self.mode == "online":
            self._start_online()

    def _stop(self):
        self._phase_timer.stop()
        self.canvas.stop_flicker()
        self.canvas.clear_cue()
        self.mode = "idle"
        self.mode_combo.setCurrentIndex(0)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("已停止")

    # ═══════════════════════════════════════════════════════════
    #  训练采集
    # ═══════════════════════════════════════════════════════════

    def _start_training(self):
        blocks = self._car_blocks()
        n_per = self.count_spin.value()

        self._train_plan = blocks * n_per
        np.random.RandomState(int(time.time())).shuffle(self._train_plan)
        self._train_idx = 0
        self._train_done = 0
        self.progress.setMaximum(len(self._train_plan))
        self.progress.setValue(0)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._next_trial()

    def _next_trial(self):
        if self._train_idx >= len(self._train_plan):
            self._stop()
            msg = QMessageBox()
            msg.setWindowTitle("完成")
            msg.setText(f"采集完成! 共 {self._train_done} 个 trial。\n请切换到\"在线控制\"模式。")
            msg.exec_()
            return

        block = self._train_plan[self._train_idx]
        self._train_block = block
        labels = self._parse_labels()
        label = labels[block] if block < len(labels) else f"B{block + 1:02d}"
        freq = self.sti_freqs[block]

        # 高亮提示
        self.canvas.cue_target(block, 0)
        self.canvas.start_flicker()
        self.train_phase = "cue"
        self._phase_t0 = time.perf_counter()
        self._phase_dur = 1.0  # 提示 1 秒
        self._phase_timer.start(50)

        self.status_lbl.setText(f"👁 注视绿色高亮块: 「{label}」({freq:.1f}Hz)")

    def _phase_tick(self):
        elapsed = time.perf_counter() - self._phase_t0

        if self.mode == "training":
            if self.train_phase == "cue" and elapsed >= self._phase_dur:
                # 进入刺激采集阶段
                self.train_phase = "stim"
                self._phase_t0 = time.perf_counter()
                self._phase_dur = 3.0
                self.canvas.clear_cue()
                self._clear_eeg()
                block = self._train_block
                labels = self._parse_labels()
                label = labels[block] if block < len(labels) else f"B{block + 1:02d}"
                self.status_lbl.setText(f"⏳ 采集中... 请持续注视「{label}」")

            elif self.train_phase == "stim" and elapsed >= self._phase_dur:
                # 保存数据
                self._save_trial(self._train_block)
                self._train_done += 1
                self._train_idx += 1
                self.progress.setValue(self._train_idx)

                # 短暂休息
                self.train_phase = "rest"
                self._phase_t0 = time.perf_counter()
                self._phase_dur = 0.8
                self.canvas.stop_flicker()
                self.status_lbl.setText(f"☕ 休息... ({self._train_done} 完成)")

            elif self.train_phase == "rest" and elapsed >= self._phase_dur:
                self._next_trial()

        elif self.mode == "online":
            if elapsed >= self._phase_dur:
                # 解码
                self._online_decode()
                # 下一个窗口
                self._phase_t0 = time.perf_counter()
                self._phase_dur = float(self.win_combo.currentText().replace("s", ""))
                self._clear_eeg()

    # ═══════════════════════════════════════════════════════════
    #  EEG 缓冲
    # ═══════════════════════════════════════════════════════════

    def _clear_eeg(self):
        with self._eeg_lock:
            self._eeg_chunks.clear()
            self._eeg_points = 0
        self._eeg_stim_start = time.perf_counter()

    def push_eeg_data_threadsafe(self, data):
        if data is None:
            return
        with self._eeg_lock:
            if isinstance(data, np.ndarray) and data.ndim == 2:
                self._eeg_chunks.append(np.asarray(data, dtype=float))
                self._eeg_points += int(data.shape[-1])

    def _get_eeg(self) -> np.ndarray:
        with self._eeg_lock:
            if not self._eeg_chunks:
                return np.array([])
            if len(self._eeg_chunks) == 1:
                return self._eeg_chunks[0]
            try:
                return np.concatenate(self._eeg_chunks, axis=-1)
            except:
                return np.array([])

    # ═══════════════════════════════════════════════════════════
    #  数据保存
    # ═══════════════════════════════════════════════════════════

    def _save_trial(self, block_idx: int):
        data = self._get_eeg()
        if data.size == 0 or data.ndim != 2:
            return

        labels = self._parse_labels()
        label = labels[block_idx] if block_idx < len(labels) else f"B{block_idx + 1:02d}"
        freq = self.sti_freqs[block_idx]
        phase = self.canvas.block_phase(block_idx)

        save_dir = os.path.join(PROJECT_ROOT, "saveCarData", self.subject)
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{ts}_{label}.mat"

        try:
            savemat(os.path.join(save_dir, fname), {
                "eeg_data": data,
                "data": data,
                "block_idx": int(block_idx),
                "target_id": int(block_idx),
                "label_idx": int(block_idx),
                "label_text": label,
                "freq_hz": float(freq),
                "phase_rad": float(phase),
                "sample_rate_hz": self.sample_rate,
                "stim_freqs_hz": np.array(self.sti_freqs, dtype=float),
                "stim_phases_rad": np.array(self.canvas.phases, dtype=float),
                "timestamp": ts,
            })
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ═══════════════════════════════════════════════════════════
    #  在线控制
    # ═══════════════════════════════════════════════════════════

    def _start_online(self):
        self._model_name = self.model_combo.currentText()
        if not self._init_decoder():
            QMessageBox.warning(self, "错误",
                "解码器训练失败。请先在\"训练采集\"模式下采集数据。")
            return

        self._online_total = 0
        self._online_correct = 0
        self._decision_buf.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.canvas.start_flicker()
        self.mode = "online"
        self._clear_eeg()
        self._phase_t0 = time.perf_counter()
        self._phase_dur = float(self.win_combo.currentText().replace("s", ""))
        self._phase_timer.start(200)

        self.status_lbl.setText("🔴 在线解码中...")

    def _init_decoder(self) -> bool:
        """用已有数据训练解码器（只训练小车命令块）。"""
        car_blocks = self._car_blocks()
        data_dir = os.path.join(PROJECT_ROOT, "saveCarData", self.subject)
        if not os.path.isdir(data_dir):
            return False

        files = sorted(glob(os.path.join(data_dir, "*.mat")))
        if len(files) < len(car_blocks) * 2:
            return False

        win_sec = float(self.win_combo.currentText().replace("s", ""))
        req_pts = int(win_sec * self.sample_rate)

        X_list, y_list = [], []
        for fp in files:
            try:
                mat = loadmat(fp)
                bid = extract_target_id(mat)
                if bid not in car_blocks:
                    continue
                sample = extract_training_sample(mat, req_pts, self.sample_rate)
                if sample is None:
                    continue
                sample = preprocess_model_input(sample)
                if sample is None:
                    continue
                X_list.append(sample)
                y_list.append(int(bid))
            except:
                continue

        if len(X_list) < 10:
            return False

        X = np.stack(X_list, axis=0)
        y = np.array(y_list)

        if self._model_name == "TDCA":
            n_classes = len(np.unique(y))
            dec = TDCA(3, win_sec, self.sti_freqs[:n_classes],
                       Nh=8, sample_rate=self.sample_rate)
            dec.fit(X, y)
            self._decoder = dec
        else:
            # FBCCA/CCA: 传入相位信息, 启用频率-相位联合编码
            phases = list(self.canvas.phases[:len(car_blocks)])
            self._decoder = FBCCA(3, win_sec, self.sti_freqs[:len(car_blocks)],
                                  Nh=8, sample_rate=self.sample_rate,
                                  phases=phases)
        return True

    def _online_decode(self):
        data = self._get_eeg()
        if data.size == 0:
            return

        win_sec = float(self.win_combo.currentText().replace("s", ""))
        req_pts = int(win_sec * self.sample_rate)
        if data.shape[-1] < req_pts:
            return

        window = preprocess_model_input(data[:, -req_pts:])
        if window is None:
            return

        try:
            if self._model_name == "TDCA":
                _, scores, conf = self._decoder.classify_with_scores(window)
                pred = int(np.argmax(scores))
            else:
                scores = self._decoder.score_vector(window)
                pred = int(np.argmax(scores))
                if len(scores) >= 2:
                    top2 = np.partition(scores, -2)[-2:]
                    conf = float(top2[1] - top2[0])
                else:
                    conf = 0.0
        except Exception:
            return

        car_blocks = self._car_blocks()
        labels = self._parse_labels()

        # 投票
        self._decision_buf.append(pred)
        votes = list(self._decision_buf)
        if len(votes) >= 3:
            u, c = np.unique(votes, return_counts=True)
            winner = u[c.argmax()]
            if c.max() >= 2:
                cmd = labels[winner] if winner < len(labels) else f"B{winner + 1}"
                self.pred_lbl.setText(f"当前命令: {cmd}")
                self._decision_buf.clear()
                # 输出到小车
                if self._robot_enabled and self._robot.connected:
                    threading.Thread(target=self._robot.move,
                                     args=(winner,), daemon=True).start()
        else:
            cmd = labels[pred] if pred < len(labels) else f"B{pred + 1}"
            self.pred_lbl.setText(f"解码: {cmd}")

        self.conf_bar.setValue(int(min(conf * 100, 100)))
        self.conf_lbl.setText(f"置信度: {conf:.3f}")

    # ═══════════════════════════════════════════════════════════
    #  小车连接
    # ═══════════════════════════════════════════════════════════

    def _toggle_car(self):
        if self._robot.connected:
            self._robot.close()
            self._robot.connected = False
            self.car_btn.setText("连接")
            self.net_lbl.setText("小车: 未连接")
            self.net_lbl.setStyleSheet("color:#EF4444;font-size:10px;")
        else:
            self._robot = RobotClient(self.car_ip.text(), 65432)
            if self._robot.connect():
                self.car_btn.setText("断开")
                self.net_lbl.setText("小车: 已连接 ✓")
                self.net_lbl.setStyleSheet("color:#10B981;font-size:10px;")
            else:
                self.net_lbl.setText("小车: 连接失败")
                self.net_lbl.setStyleSheet("color:#EF4444;font-size:10px;")

    # ═══════════════════════════════════════════════════════════
    #  刷新
    # ═══════════════════════════════════════════════════════════

    def _refresh(self):
        with self._eeg_lock:
            pts = self._eeg_points
        if pts > 0:
            self.sub_status.setText(
                f"3×3网格 | 9块闪烁 | "
                f"{len(self._car_blocks())}个命令 | EEG: {pts}点"
            )
