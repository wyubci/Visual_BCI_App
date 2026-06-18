import socket
import pyttsx3
from collections import deque
import json
from datetime import datetime
import shutil
import ctypes
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
import threading
import numpy as np
import time
from scipy import signal
from scipy.io import savemat, loadmat
from glob import glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA
from config import config
from interface.car_interface.acquisition import (
    AcquisitionConfig,
    EegSampleBuffer,
    aligned_window_indices,
    extract_aligned_window,
    preprocess_model_input,
    update_receive_metadata,
)
from interface.car_interface.training_framework import (
    CarTrainingFramework,
    build_training_plan,
    extract_int,
    extract_label_text,
    list_weight_files,
)

# ==========================================
# PC CLIENT FOR X1
# Run this on Windows
# ==========================================

ROBOT_IP = '10.186.179.92'  # CHANGE THIS IF NEEDED
PORT = 65432

# SSVEP car control uses frame-locked square-wave stimuli.  On high-refresh
# displays (for example 165 Hz), a nominal frequency such as 6.67 Hz cannot be
# rendered exactly unless it is an integer divisor of the refresh rate.  The
# functions below choose integer frame periods first, then use the resulting
# actual frequencies for CCA/FBCCA/TDCA reference signals.  This keeps the
# visual stimulus and the decoder mathematically consistent.
LEGACY_CAR_FREQS = np.array([6.00, 7.50, 8.50, 10.00, 11.00], dtype=float)
SSVEP_TARGET_FREQS = np.array([6.0, 7.5, 8.5, 10.0, 11.0], dtype=float)
SSVEP_FREQ_MIN_HZ = 5.0
SSVEP_FREQ_MAX_HZ = 15.5
BENCHMARK_PRE_STIMULUS_SEC = 0.5
BENCHMARK_VISUAL_DELAY_SEC = 0.14
BENCHMARK_NUM_HARMONICS = 5
VALIDATION_THREE_CLASS_MODE = False


def _detect_screen_refresh_rate(default_hz=60.0):
    """Return the screen refresh rate, preferring 60 Hz if dual monitors."""
    try:
        screens = QApplication.screens()
        if not screens:
            return float(default_hz)
        rates = []
        for i, scr in enumerate(screens):
            r = float(scr.refreshRate())
            if 30.0 <= r <= 360.0 and np.isfinite(r):
                rates.append((i, r))
        if not rates:
            return float(default_hz)
        # 双屏场景：优先选 60Hz 附近的
        for idx, r in rates:
            if 55.0 <= r <= 75.0:
                print(f'[SSVEP] 使用屏幕{idx} ({r:.0f}Hz)')
                return float(r)
        # 否则用第一个
        idx, r = rates[0]
        print(f'[SSVEP] 使用屏幕{idx} ({r:.0f}Hz)')
        return float(r)
    except Exception:
        pass
    return float(default_hz)


def _common_refresh_periods(refresh_hz):
    """Hand-tuned periods for common monitor refresh rates."""
    r = float(refresh_hz)
    if 160.0 <= r <= 170.0:
        return [27, 22, 19, 16, 15]      # 165 Hz -> ~6.1, 7.5, 8.68, 10.3, 11.0 
    if 235.0 <= r <= 245.0:
        return [40, 32, 28, 24, 22]      # 240 Hz
    if 115.0 <= r <= 125.0:
        return [20, 16, 14, 12, 11]      # 120 Hz
    if 140.0 <= r <= 148.0:
        return [24, 19, 17, 14, 13]      # 144 Hz
    if 50.0 <= r <= 75.0:
        return [10, 8, 7, 6, 5]          # 60 Hz -> 6.0, 7.5, 8.57, 10.0, 12.0
    return None


def _adaptive_stimulus_periods(refresh_hz, n_targets=5):
    """Choose unique integer frame periods close to target SSVEP frequencies."""
    common = _common_refresh_periods(refresh_hz)
    if common is not None and len(common) >= n_targets:
        return [int(x) for x in common[:n_targets]]

    refresh_hz = float(refresh_hz)
    p_min = max(2, int(np.floor(refresh_hz / SSVEP_FREQ_MAX_HZ)))
    p_max = max(p_min + n_targets, int(np.ceil(refresh_hz / SSVEP_FREQ_MIN_HZ)))
    candidates = list(range(p_min, p_max + 1))

    selected = []
    selected_freqs = []
    for target in SSVEP_TARGET_FREQS[:n_targets]:
        best_p, best_score = None, None
        for p in candidates:
            if p in selected:
                continue
            f = refresh_hz / float(p)
            if f < SSVEP_FREQ_MIN_HZ or f > SSVEP_FREQ_MAX_HZ:
                continue
            distance = abs(f - float(target))
            too_close_penalty = sum(max(0.0, 0.45 - abs(f - old_f)) * 10.0 for old_f in selected_freqs)
            odd_penalty = 0.04 if (p % 2) else 0.0
            score = distance + too_close_penalty + odd_penalty
            if best_score is None or score < best_score:
                best_p, best_score = int(p), float(score)
        if best_p is None:
            # Fallback should rarely happen; keep a valid integer period anyway.
            best_p = max(2, int(round(refresh_hz / float(target))))
        selected.append(int(best_p))
        selected_freqs.append(refresh_hz / float(best_p))

    return selected


def build_stimulus_profile(refresh_hz, n_targets=5):
    """Return frame periods, duty frames and actual frequencies for the monitor."""
    periods = _adaptive_stimulus_periods(refresh_hz, n_targets=n_targets)
    freqs = np.array([float(refresh_hz) / float(p) for p in periods], dtype=float)
    print(f'[SSVEP] 屏幕={refresh_hz:.0f}Hz  帧周期={periods}  实际频率={np.round(freqs, 3).tolist()} Hz')
    duty = [max(1, int(round(float(p) * 0.5))) for p in periods]
    return {
        "refresh_hz": float(refresh_hz),
        "period_frames": [int(p) for p in periods],
        "duty_frames": [int(d) for d in duty],
        "actual_freqs_hz": freqs,
    }


_tts_lock = threading.Lock()


def speak_async_safe(text):
    if not _tts_lock.acquire(blocking=False):
        return

    def _worker():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS failed: {e}")
        finally:
            _tts_lock.release()

    threading.Thread(target=_worker, daemon=True).start()

class RobotClient:
    def __init__(self, ip, port):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        self.connected = False

    def connect(self):
        try:
            print(f"Connecting to {self.addr}...")
            self.sock.connect(self.addr)
            self.connected = True
            print("Connected!")
        except Exception as e:
            print(f"Connection Failed: {e}")
            print(f"Control link unavailable: {self.addr[0]}:{self.addr[1]}. Please check the car control service and port.")

    def send(self, cmd):
        if not self.connected: return
        try:
            self.sock.send(cmd.encode('utf-8'))
        except:
            self.connected = False

    def set_motor(self, v1, v2, v3, v4):
        cmd = f"{int(v1)},{int(v2)},{int(v3)},{int(v4)}"
        print(f"Sending: {cmd}")
        self.send(cmd)

    def move(self,ing):
        if(ing==0):
            v1=47
            v2=0
            v3=47
            v4=0
            self.set_motor(v1, v2, v3, v4)
            time.sleep(2.0)
            self.set_motor(0,0,0,0)
        elif(ing==1):
            v1=-35
            v2=0
            v3=-35
            v4=00
            self.set_motor(v1, v2, v3, v4)
            time.sleep(2.0)
            self.set_motor(0,0,0,0)
        elif(ing==2):
            # Left turn: ~45 degrees at reduced speed
            v1=-40
            v2=-40
            v3=40
            v4=40
            self.set_motor(v1, v2, v3, v4)
            time.sleep(1.35)
            self.set_motor(0,0,0,0)
        elif(ing==3):
            v1=0
            v2=0
            v3=0
            v4=0
            self.set_motor(v1, v2, v3, v4)

        elif(ing==4):
            # Right turn: ~45 degrees at reduced speed
            v1=40
            v2=40
            v3=-40
            v4=-40
            self.set_motor(v1, v2, v3, v4)
            time.sleep(1.35)
            self.set_motor(0,0,0,0)
        
    def stop(self):
        print("Stopping")
        self.send("stop")

    def beep(self):
        print("Beep")
        self.send("beep")

    def close(self):
        self.sock.close()


class StiRect(QLabel):
    def __init__(self, text, parent, sti, fontSize=60, color=255):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.fontSize = fontSize
        self.sti = float(sti)
        self.period_frames = None
        self.duty_frames = None
        self.text = text
        self.color_value = color
        self.current_color = QColor(0, 0, 0, 255)
        self.default_color = QColor(0, 0, 0, 255)
        self.text_color = QColor(255, 255, 255)
        self.border_color = QColor(90, 90, 90)
        self.stop_emphasis = False

    def setStopStyle(self, enabled=True):
        self.stop_emphasis = False
        self.border_color = QColor(0, 0, 0)
        self.update()

    def changeText(self, text):
        self.text = text
        self.update()

    def setFrequencyProfile(self, actual_freq_hz, period_frames=None, duty_frames=None):
        self.sti = float(actual_freq_hz)
        self.period_frames = int(period_frames) if period_frames is not None else None
        self.duty_frames = int(duty_frames) if duty_frames is not None else None

    def flicker(self, freq, now_time):
        light = 255 * np.sin(2 * np.pi * now_time * freq)
        return int(light)

    def changeColor(self, f, now_time):
        flick_value = self.flicker(f, now_time)
        is_black = flick_value >= 0
        self.current_color = QColor(0, 0, 0, 255) if is_black else QColor(255, 255, 255, 255)
        self.text_color = QColor(255, 255, 255) if is_black else QColor(0, 0, 0)
        self.update()

    def changeColorByFrame(self, freq, frame_index, refresh_hz):
        # Prefer integer-frame rendering: the stimulus and decoder both use
        # actual_freq_hz = refresh_hz / period_frames.  Falling back to the old
        # sinusoidal sign method is kept for compatibility with other views.
        if self.period_frames is not None and self.period_frames > 0:
            duty = self.duty_frames if self.duty_frames is not None else int(round(self.period_frames * 0.5))
            duty = max(1, min(int(duty), int(self.period_frames) - 1 if self.period_frames > 1 else 1))
            phase = int(frame_index) % int(self.period_frames)
            is_black = phase < duty
        else:
            t = float(frame_index) / max(float(refresh_hz), 1e-6)
            flick_value = 255.0 * np.sin(2 * np.pi * float(freq) * t)
            is_black = flick_value >= 0
        new_color = QColor(0, 0, 0, 255) if is_black else QColor(255, 255, 255, 255)
        new_text_color = QColor(255, 255, 255) if is_black else QColor(0, 0, 0)
        if self.current_color != new_color or self.text_color != new_text_color:
            self.current_color = new_color
            self.text_color = new_text_color
            self.update()

    def setDefaultColor(self):
        if self.current_color != self.default_color or self.text_color != QColor(0, 0, 0):
            self.current_color = self.default_color
            self.text_color = QColor(0, 0, 0)
            self.update()

    def paintEvent(self, a0):
        super().paintEvent(a0)

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(self.current_color)
        painter.setBrush(self.current_color)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        painter.setBrush(QBrush())
        pen = QPen()
        pen.setWidth(2)
        if self.stop_emphasis:
            pen.setStyle(Qt.DashLine)
            pen.setWidth(3)
        pen.setColor(self.border_color)
        painter.setPen(pen)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        text_color = self.text_color
        font = QFont()
        font.setPixelSize(self.fontSize)

        pen = QPen()
        pen.setColor(text_color)
        pen.setBrush(text_color)

        painter.setPen(pen)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(1, 1, -1, -1), Qt.AlignCenter, self.text)

        painter.end()


class StimulusTarget:
    def __init__(self, canvas, text, sti, font_size=34):
        self.canvas = canvas
        self.text = text
        self.sti = float(sti)
        self.fontSize = int(font_size)
        self.period_frames = None
        self.duty_frames = None
        self.current_color = QColor(255, 255, 255, 255)
        self.default_color = QColor(255, 255, 255, 255)
        self.text_color = QColor(0, 0, 0)
        self.border_color = QColor(0, 0, 0)
        self.stop_emphasis = False

    def setStopStyle(self, enabled=True):
        self.stop_emphasis = False
        self.border_color = QColor(0, 0, 0)
        self.canvas.update()

    def changeText(self, text):
        self.text = text
        self.canvas.update()

    def setFrequencyProfile(self, actual_freq_hz, period_frames=None, duty_frames=None):
        self.sti = float(actual_freq_hz)
        self.period_frames = int(period_frames) if period_frames is not None else None
        self.duty_frames = int(duty_frames) if duty_frames is not None else None

    def changeColorByFrame(self, freq, frame_index, refresh_hz):
        if self.period_frames is not None and self.period_frames > 0:
            duty = self.duty_frames if self.duty_frames is not None else int(round(self.period_frames * 0.5))
            duty = max(1, min(int(duty), int(self.period_frames) - 1 if self.period_frames > 1 else 1))
            phase = int(frame_index) % int(self.period_frames)
            is_white = phase < duty
        else:
            t = float(frame_index) / max(float(refresh_hz), 1e-6)
            is_white = 255.0 * np.sin(2 * np.pi * float(freq) * t) >= 0
        new_color = QColor(255, 255, 255, 255) if is_white else QColor(0, 0, 0, 255)
        new_text_color = QColor(255, 255, 255)
        if self.current_color != new_color or self.text_color != new_text_color:
            self.current_color = new_color
            self.text_color = new_text_color
            self.canvas.request_update()

    def setDefaultColor(self):
        if self.current_color != self.default_color or self.text_color != QColor(255, 255, 255):
            self.current_color = self.default_color
            self.text_color = QColor(255, 255, 255)
            self.canvas.request_update()


class StimulusCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.targets = []

    def addTarget(self, text, sti, font_size=34):
        target = StimulusTarget(self, text, sti, font_size)
        self.targets.append(target)
        return target

    def request_update(self):
        self.update()

    def _target_rects(self):
        w = max(1, self.width())
        h = max(1, self.height())
        size = int(min(156, max(96, min(w / 4.2, h / 2.45))))
        if len(self.targets) == 3:
            y = int(h * 0.34)
            return [
                QRect(int(w * 0.25 - size / 2), y, size, size),
                QRect(int(w * 0.50 - size / 2), y, size, size),
                QRect(int(w * 0.75 - size / 2), y, size, size),
            ]
        small = int(min(size, 146))
        top_y = int(h * 0.08)
        bottom_y = int(h * 0.58)
        return [
            QRect(int(w * 0.28 - size / 2), top_y, size, size),
            QRect(int(w * 0.72 - size / 2), top_y, size, size),
            QRect(int(w * 0.22 - small / 2), bottom_y, small, small),
            QRect(int(w * 0.50 - small / 2), bottom_y, small, small),
            QRect(int(w * 0.78 - small / 2), bottom_y, small, small),
        ]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        rects = self._target_rects()
        for target, rect in zip(self.targets, rects):
            painter.setPen(target.current_color)
            painter.setBrush(target.current_color)
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

            pen = QPen()
            pen.setWidth(3 if target.stop_emphasis else 2)
            if target.stop_emphasis:
                pen.setStyle(Qt.DashLine)
            pen.setColor(target.border_color)
            painter.setPen(pen)
            painter.setBrush(QBrush())
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

            font = QFont()
            font.setFamily("Microsoft YaHei")
            font.setPixelSize(target.fontSize)
            painter.setFont(font)
            text_rect = rect.adjusted(1, 1, -1, -1)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(text_rect.adjusted(2, 2, 2, 2), Qt.AlignCenter, target.text)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_rect, Qt.AlignCenter, target.text)
        painter.end()


class CarControlWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()
        
        self.validation_three_class_mode = False
        self.stim_target_count = 5
        refresh_hz = _detect_screen_refresh_rate(default_hz=60.0)
        self.stim_profile = build_stimulus_profile(refresh_hz, n_targets=self.stim_target_count)
        self.stim_refresh_hz = float(self.stim_profile["refresh_hz"])
        self.stim_period_frames = [int(x) for x in self.stim_profile["period_frames"]]
        self.stim_duty_frames = [int(x) for x in self.stim_profile["duty_frames"]]
        self.sti_lst = [float(x) for x in self.stim_profile["actual_freqs_hz"]]
        
        self.commands = [
            "\u524d\u8fdb", "\u540e\u9000", "\u5de6\u8f6c", "\u505c\u6b62", "\u53f3\u8f6c"
        ]
        self.robot_command_indices = list(range(len(self.commands)))

        self.cache_data = np.array([])
        self.cache_chunks = []
        self.cache_points = 0
        self.sample_buffer = EegSampleBuffer()
        self.eeg_buffer_lock = threading.RLock()
        self.start_flick = False
        self.finish = True
        self.start_cache = False
        self.continuous_mode = False
        self.robot_control_enabled = False
        self.robot_net_ok = False

        self.sample_rate_hz = 250
        self.analysis_delay_sec = 0.25
        self.model_times_sec = 4.0
        self.analysis_window_sec = self.model_times_sec - self.analysis_delay_sec
        self.training_window_sec = self.model_times_sec
        self.training_sample_points = int(round(self.model_times_sec * self.sample_rate_hz))
        self.online_window_sec = self.model_times_sec
        self.online_sample_points = int(round(self.model_times_sec * self.sample_rate_hz))
        self.model_name = "TDCA"
        self.min_confidence = 0.02

        # 通道数（从配置读取，LSL 下为选中通道数，NeuroDance 下为 8）
        from config import config as _cfg
        self.n_channels = (
            len(getattr(_cfg, 'lsl_selected_channels', list(range(22, 31))))
            if getattr(_cfg, 'device_type', 'neuro_dance_tcp') == 'lsl'
            else 8
        )

        self.trial_cue_sec = 0.5
        self.trial_rest_sec = 1.0
        self.acquisition_tail_sec = 0.5
        self.online_analysis_window_sec = 3.0
        self.online_sliding_step_sec = 0.5

        self.stim_frame_interval_ms = max(1, int(1000.0 / self.stim_refresh_hz))
        self.stim_poll_interval_ms = self.stim_frame_interval_ms

        self.fbcca = self._create_classifier(self.model_times_sec)
        self.fbcca_sliding = None
        self.enable_quality_tools = False
        self.quality_calc_timer = None
        self.quality_calc_pending = []
        self.quality_calc_rows = []
        self.quality_calc_batch_size = 0
        
        self._initLayout()
        self._initItems()

        self.decision_buffer = deque(maxlen=3)
        self.vote_threshold = 2
        self.command_cooldown_sec = 1.0
        self.last_command_time = 0.0
        self.last_command_idx = None
        self.online_strategy = "sliding_vote"
        self.execution_mode = self.online_strategy
        self.last_gate_reason = ""

        self.current_candidate = "-"
        self.current_confidence = 0.0
        self.current_votes = []
        self.result_history = deque(maxlen=24)
        self.result_history_count = 0
        self.online_eval_truth = "not_counted"
        self.online_eval_total = 0
        self.online_eval_correct = 0
        self.test_eval_plan = []
        self.test_eval_index = 0
        self.test_eval_total = 0
        self.test_eval_correct = 0
        self.test_eval_results = []
        self.class_score_scale = np.ones(len(self.commands), dtype=float)
        self.online_score_ema = np.ones(len(self.commands), dtype=float)
        self.online_score_ema_alpha = 0.08
        self.online_score_warmup = 0
        self.online_warmup_windows = 3
        self.robot_net_ok = False
        self._refresh_network_status()

        self.mode = "online"
        self.training_collecting = False
        self.training_collected = 0
        self.training_plan = []
        self.training_timeout_sec = 6.0
        self.training_target_label = ""
        self.training_collect_start_time = 0.0
        self.selected_weight_file = ""
        self.dataset_file_map = {}
        self._rebuild_classifier_with_adaptation()
        self.online_time_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.training_hint.setText(self._online_hint_text())

        self.online_window_active = False
        self.online_window_start_time = 0.0
        self.sliding_cycle_start_mono = 0.0
        self.sliding_cycle_count = 0
        self.online_phase = "idle"
        self.online_phase_start_ts = 0.0
        self.online_stim_frame_idx = 0
        self.online_last_render_frame_idx = -1
        self.online_last_progress_update_mono = 0.0
        self.online_stim_onset_mono = 0.0
        self.online_stim_onset_unix = 0.0
        self.online_trial_meta = {}

        self.training_phase = "idle"
        self.training_phase_start_ts = 0.0
        self.training_stim_frame_idx = 0
        self.training_last_render_frame_idx = -1
        self.training_last_progress_update_mono = 0.0
        self.training_stim_onset_mono = 0.0
        self.training_stim_onset_unix = 0.0
        self.training_pending_success = False
        self.training_trial_meta = {}
        self.stim_timing_log = []
        self.stim_timing_scope = ""
        self.stim_timing_last_frame = None
        self.stim_timing_last_mono = None
        self._timer_resolution_enabled = False
        if sys.platform.startswith("win"):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                self._timer_resolution_enabled = True
            except Exception:
                self._timer_resolution_enabled = False

        self.online_timer = QTimer(self)
        self.online_timer.setTimerType(Qt.PreciseTimer)
        self.online_timer.setInterval(self.stim_poll_interval_ms)
        self.online_timer.timeout.connect(self._online_tick)

        self.training_timer = QTimer(self)
        self.training_timer.setTimerType(Qt.PreciseTimer)
        self.training_timer.setInterval(self.stim_poll_interval_ms)
        self.training_timer.timeout.connect(self._training_tick)

        self.stim_render_timer = QTimer(self)
        self.stim_render_timer.setTimerType(Qt.PreciseTimer)
        self.stim_render_timer.setInterval(2)
        self.stim_render_timer.timeout.connect(self._stim_render_tick)

        self.setFocusPolicy(Qt.StrongFocus)

    def setQss(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qss_path = os.path.join(base_dir, 'source', 'qss', 'mainWindow.qss')
        with open(qss_path, encoding='utf-8') as f:
            self.setStyleSheet(f.read())

        self.setStyleSheet(self.styleSheet() + """
            QWidget#carControlWindow {
                background-color: #000000;
                font-family: 'Microsoft YaHei', 'Noto Sans CJK SC';
                color: #E8EDF5;
            }
            QLabel, CaptionLabel, BodyLabel, StrongBodyLabel {
                color: #E8EDF5;
            }
            QFrame#sideNav {
                background-color: #0E1A2B;
                border-radius: 12px;
                border: 1px solid #24324A;
            }
            QFrame#card {
                background-color: #000000;
                border-radius: 12px;
                border: 1px solid #2A2A2A;
            }
            QLabel#cardTitle {
                color: #F4F7FC;
                font-size: 15px;
                font-weight: 600;
            }
            QGroupBox {
                border: 1px solid #2F2F2F;
                border-radius: 10px;
                margin-top: 12px;
                font-weight: 600;
                color: #D9E2F0;
                background: #050505;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0A0A0A;
                border: 1px solid #2F2F2F;
                border-radius: 8px;
                min-height: 30px;
                padding: 2px 8px;
                font-size: 13px;
                font-weight: 500;
                color: #E8EDF5;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #7CB0FF;
                background: #141414;
            }
            QTreeWidget {
                background: #050505;
                border: 1px solid #2F2F2F;
                border-radius: 10px;
                alternate-background-color: #0A0A0A;
                color: #E8EDF5;
            }
            QHeaderView::section {
                background: #111111;
                color: #F4F7FC;
                border: none;
                padding: 6px;
                font-weight: 600;
            }
            QAbstractItemView::item {
                color: #E8EDF5;
            }
            QLabel#statusTagOk {
                background: #0A3D34;
                color: #8BE9C3;
                border-radius: 8px;
                padding: 4px 10px;
                font-weight: 600;
            }
            QLabel#statusTagBad {
                background: #4A2027;
                color: #FFB4B4;
                border-radius: 8px;
                padding: 4px 10px;
                font-weight: 600;
            }
            QPushButton#navActive {
                text-align: left;
                border: none;
                border-radius: 10px;
                padding: 8px 10px;
                background: #2F6FD6;
                color: #F7FAFF;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#navItem {
                text-align: left;
                border: none;
                border-radius: 10px;
                padding: 8px 10px;
                background: transparent;
                color: #C9D6EA;
                font-size: 13px;
            }
            QPushButton#navItem:hover {
                background: #1E2E47;
                color: #F5F8FF;
            }
        """)

    def _initLayout(self):
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(12)
        self.setLayout(self.layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout.addWidget(self.scroll_area, 1)

        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setWidget(self.content_widget)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)

    def _initItems(self):
        def make_card():
            card = QFrame()
            card.setObjectName("card")
            effect = QGraphicsDropShadowEffect(card)
            effect.setBlurRadius(18)
            effect.setOffset(0, 4)
            effect.setColor(QColor(22, 93, 255, 25))
            card.setGraphicsEffect(effect)
            return card

        top_card = make_card()
        self.content_layout.addWidget(top_card)
        top_layout = QVBoxLayout(top_card)
        top_layout.setContentsMargins(14, 12, 14, 12)
        top_layout.setSpacing(10)

        self.show_label = DisplayLabel()
        self.show_label.setText("\u7cfb\u7edf\u5c31\u7eea\uff0c\u8bf7\u9009\u62e9\u6a21\u5f0f\u540e\u5f00\u59cb")
        self.show_label.setMinimumHeight(48)
        self.show_label.setMaximumHeight(60)
        title_font = QFont()
        title_font.setPixelSize(24)
        title_font.setBold(False)
        title_font.setWeight(QFont.DemiBold)
        self.show_label.setFont(title_font)
        self.show_label.setStyleSheet("color: #F9FAFB; font-weight: 600;")

        top_title_row = QHBoxLayout()
        top_title_row.addWidget(self.show_label, stretch=1)

        right_status = QVBoxLayout()
        self.system_status = QLabel("\u7cfb\u7edf\u72b6\u6001: \u8fd0\u884c\u4e2d")
        self.system_status.setObjectName("statusTagOk")
        right_status.addWidget(self.system_status)
        self.top_network_status = QLabel("\u8fde\u63a5\u72b6\u6001: \u672a\u8fde\u63a5")
        self.top_network_status.setObjectName("statusTagBad")
        right_status.addWidget(self.top_network_status)
        self.time_status = QLabel(time.strftime("\u65f6\u95f4: %Y-%m-%d %H:%M:%S"))
        self.time_status.setStyleSheet("font-size: 12px; color: #9FB0C9;")
        right_status.addWidget(self.time_status)
        top_title_row.addLayout(right_status)
        top_layout.addLayout(top_title_row)

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(lambda: self.time_status.setText(time.strftime("\u65f6\u95f4: %Y-%m-%d %H:%M:%S")))
        self.clock_timer.start()

        param_row = QHBoxLayout()
        param_row.setSpacing(10)

        def make_field_label(text):
            lb = CaptionLabel(text)
            lb.setStyleSheet("font-size: 13px; color: #E5E7EB; font-weight: 500;")
            return lb

        self.mode_bar = QGroupBox("\u91c7\u96c6\u6a21\u5f0f")
        mode_layout = QGridLayout(self.mode_bar)
        mode_layout.setContentsMargins(10, 12, 10, 10)
        mode_layout.setSpacing(8)

        mode_layout.addWidget(make_field_label("\u6a21\u5f0f:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["\u5728\u7ebf\u6a21\u5f0f", "\u8bad\u7ec3\u6a21\u5f0f", "\u6d4b\u8bd5\u6a21\u5f0f"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo, 0, 1)
        mode_layout.addWidget(make_field_label("\u6bcf\u7c7b\u6b21\u6570:"), 0, 2)
        self.train_count_spin = QSpinBox()
        self.train_count_spin.setRange(1, 500)
        self.train_count_spin.setValue(20)
        self.train_count_spin.setMinimumWidth(70)
        self.train_count_spin.valueChanged.connect(self._on_training_plan_params_changed)
        mode_layout.addWidget(self.train_count_spin, 0, 3)
        mode_layout.addWidget(make_field_label("\u5728\u7ebf\u7a97\u53e3:"), 0, 4)
        self.online_time_combo = QComboBox()
        self.online_time_combo.addItems(["2s", "3s", "4s"])
        self.online_time_combo.setCurrentText("3s")
        self.online_time_combo.currentTextChanged.connect(self._on_online_time_changed)
        self.online_time_combo.setMinimumWidth(84)
        mode_layout.addWidget(self.online_time_combo, 0, 5)

        mode_layout.addWidget(make_field_label("\u8bad\u7ec3\u65f6\u957f:"), 0, 6)
        self.training_time_combo = QComboBox()
        self.training_time_combo.addItems(["2s", "3s", "4s"])
        self.training_time_combo.setCurrentText("3s")
        self.training_time_combo.currentTextChanged.connect(self._on_training_time_changed)
        self.training_time_combo.setMinimumWidth(84)
        mode_layout.addWidget(self.training_time_combo, 0, 7)

        mode_layout.addWidget(make_field_label("\u8bad\u7ec3\u6807\u7b7e:"), 1, 0)
        self.train_labels_edit = QLineEdit("\u524d\u8fdb,\u540e\u9000,\u5de6\u8f6c,\u505c\u6b62,\u53f3\u8f6c")
        self.train_labels_edit.setMinimumWidth(220)
        self.train_labels_edit.editingFinished.connect(self._on_training_plan_params_changed)
        mode_layout.addWidget(self.train_labels_edit, 1, 1, 1, 5)
        mode_layout.setColumnStretch(1, 2)
        mode_layout.setColumnStretch(5, 1)

        self.model_bar = QGroupBox("\u8bc6\u522b\u6a21\u578b")
        model_layout = QHBoxLayout(self.model_bar)
        model_layout.setContentsMargins(10, 12, 10, 10)
        model_layout.setSpacing(8)
        model_layout.addWidget(make_field_label("\u6a21\u578b:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TDCA", "FBCCA", "CCA"])
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(make_field_label("\u91c7\u6837\u7387:"))
        self.sampling_rate_combo = QComboBox()
        self.sampling_rate_combo.addItems(["250Hz"])
        self.sampling_rate_combo.setCurrentText("250Hz")
        self.sampling_rate_combo.setEnabled(False)
        self.sampling_rate_combo.currentTextChanged.connect(self._on_sampling_rate_changed)
        model_layout.addWidget(self.sampling_rate_combo)
        model_layout.addWidget(make_field_label("\u7f6e\u4fe1\u9608\u503c:"))

        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setRange(0.00, 1.00)
        self.conf_threshold_spin.setSingleStep(0.01)
        self.conf_threshold_spin.setValue(self.min_confidence)
        self.conf_threshold_spin.valueChanged.connect(self._on_min_confidence_changed)
        model_layout.addWidget(self.conf_threshold_spin)

        exec_group = QGroupBox("\u6267\u884c\u63a7\u5236")
        exec_layout = QHBoxLayout(exec_group)
        exec_layout.setContentsMargins(10, 12, 10, 10)
        exec_layout.setSpacing(8)
        exec_layout.addWidget(make_field_label("在线策略:"))
        self.exec_mode_combo = QComboBox()
        self.exec_mode_combo.addItems(["滑窗投票", "异步单窗"])
        self.exec_mode_combo.setCurrentIndex(0)
        self.exec_mode_combo.currentIndexChanged.connect(self._on_execution_mode_changed)
        exec_layout.addWidget(self.exec_mode_combo)
        self.training_hint = CaptionLabel("\u8bad\u7ec3/\u6d4b\u8bd5: \u56de\u8f66\u542f\u52a8 trial")
        self.training_hint.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        self.training_hint.setWordWrap(True)
        self.training_hint.setMaximumWidth(280)
        self.training_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        exec_layout.addWidget(self.training_hint, stretch=1)
        exec_layout.addWidget(make_field_label("\u5728\u7ebf\u771f\u503c:"))
        self.online_truth_combo = QComboBox()
        self.online_truth_combo.addItem("\u4e0d\u7edf\u8ba1")
        for cmd in self.commands:
            self.online_truth_combo.addItem(cmd)
        self.online_truth_combo.currentTextChanged.connect(self._on_online_truth_changed)
        exec_layout.addWidget(self.online_truth_combo)

        param_row.addWidget(self.mode_bar, stretch=3)
        param_row.addWidget(self.model_bar, stretch=2)
        param_row.addWidget(exec_group, stretch=3)
        top_layout.addLayout(param_row)

        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        top_layout.addWidget(self.progress)

        self.mainSplitter = QSplitter(Qt.Vertical)
        self.mainSplitter.setChildrenCollapsible(False)
        self.content_layout.addWidget(self.mainSplitter, stretch=1)

        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(10)
        self.mainSplitter.addWidget(upper_widget)

        lower_widget = QWidget()
        lower_layout = QHBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(10)
        self.mainSplitter.addWidget(lower_widget)
        self.mainSplitter.setSizes([980, 220])

        control_panel = make_card()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(8)
        upper_layout.addWidget(control_panel, stretch=1)
        control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        control_panel.setMinimumHeight(620)

        panel_title = StrongBodyLabel("\u6838\u5fc3\u63a7\u5236\u6309\u94ae\u533a")
        panel_title.setObjectName("cardTitle")
        control_layout.addWidget(panel_title)

        self.sti_rects = []
        scene_host = QWidget()
        scene_layout = QVBoxLayout(scene_host)
        scene_layout.setContentsMargins(4, 4, 4, 4)
        scene_layout.setSpacing(8)

        self.stimulus_canvas = StimulusCanvas(scene_host)
        self.stimulus_canvas.setMinimumHeight(330)
        self.stimulus_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.forward_rect = self.stimulus_canvas.addTarget("\u2191\n\u524d\u8fdb", self.sti_lst[0], font_size=34)
        self.sti_rects.append(self.forward_rect)
        self.backward_rect = self.stimulus_canvas.addTarget("\u2193\n\u540e\u9000", self.sti_lst[1], font_size=34)
        self.sti_rects.append(self.backward_rect)
        self.left_rect = self.stimulus_canvas.addTarget("\u2190\n\u5de6\u8f6c", self.sti_lst[2], font_size=32)
        self.sti_rects.append(self.left_rect)
        self.stop_rect = self.stimulus_canvas.addTarget("\u25a0\n\u505c\u6b62", self.sti_lst[3], font_size=34)
        self.sti_rects.append(self.stop_rect)
        self.right_rect = self.stimulus_canvas.addTarget("\u2192\n\u53f3\u8f6c", self.sti_lst[4], font_size=32)
        self.sti_rects.append(self.right_rect)
        self._apply_stimulus_profile_to_rects()

        scene_layout.addWidget(self.stimulus_canvas, 1)

        control_layout.addWidget(scene_host, 1)

        status_panel = make_card()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(10)
        lower_layout.addWidget(status_panel, stretch=3)
        status_title = StrongBodyLabel("\u72b6\u6001\u4fe1\u606f")
        status_title.setObjectName("cardTitle")
        status_layout.addWidget(status_title)

        self.network_status = CaptionLabel("\u7f51\u7edc\u72b6\u6001: \u63a7\u5236\u672a\u8fde\u63a5")
        self.network_status.setStyleSheet("font-size: 12px; color: #E5E7EB; font-weight: 600;")
        status_layout.addWidget(self.network_status)

        self.freq_status = CaptionLabel(self._online_status_text())
        self.freq_status.setStyleSheet("font-size: 12px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.freq_status)

        ctrl_row = QGridLayout()
        ctrl_row.setHorizontalSpacing(6)
        ctrl_row.setVerticalSpacing(6)
        ctrl_row.addWidget(make_field_label("\u63a7\u5236IP:"), 0, 0)
        self.robot_ip_edit = QLineEdit(ROBOT_IP)
        self.robot_ip_edit.setMinimumWidth(120)
        ctrl_row.addWidget(self.robot_ip_edit, 0, 1)
        ctrl_row.addWidget(make_field_label("\u7aef\u53e3:"), 0, 2)
        self.robot_port_spin = QSpinBox()
        self.robot_port_spin.setRange(1, 65535)
        self.robot_port_spin.setValue(PORT)
        self.robot_port_spin.setMinimumWidth(70)
        ctrl_row.addWidget(self.robot_port_spin, 0, 3)
        self.test_conn_btn = PushButton("\u8fde\u63a5\u5c0f\u8f66")
        self.test_conn_btn.setMinimumWidth(92)
        self.test_conn_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.test_conn_btn.clicked.connect(self._toggle_robot_connection)
        ctrl_row.addWidget(self.test_conn_btn, 0, 4)
        ctrl_row.setColumnStretch(1, 1)
        status_layout.addLayout(ctrl_row)

        self.decision_info = CaptionLabel("\u8bc6\u522b\u72b6\u6001: \u5f85\u5f00\u59cb")
        self.decision_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.decision_info)

        self.command_info = CaptionLabel("\u5f53\u524d\u6307\u4ee4: -")
        self.command_info.setStyleSheet("font-size: 16px; color: #7CB0FF; font-weight: 600;")
        status_layout.addWidget(self.command_info)

        self.confidence_bar = ProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        status_layout.addWidget(self.confidence_bar)
        self.confidence_info = CaptionLabel("\u7f6e\u4fe1\u5ea6: 0.0000")
        self.confidence_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.confidence_info)

        self.vote_info = CaptionLabel("\u6295\u7968\u72b6\u6001: -")
        self.vote_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        self.vote_info.setWordWrap(True)
        status_layout.addWidget(self.vote_info)

        self.result_history_title = CaptionLabel("\u5386\u53f2\u7ed3\u679c:")
        self.result_history_title.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 600;")
        status_layout.addWidget(self.result_history_title)
        self.result_history_area = QScrollArea()
        self.result_history_area.setWidgetResizable(True)
        self.result_history_area.setFrameShape(QFrame.NoFrame)
        self.result_history_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_history_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_history_area.setMinimumHeight(92)
        self.result_history_area.setMaximumHeight(150)
        self.result_history_area.setStyleSheet(
            "QScrollArea{background:#050505;border:1px solid #2A2F3A;border-radius:8px;}"
            "QScrollBar:vertical{background:#111827;width:10px;border-radius:5px;}"
            "QScrollBar::handle:vertical{background:#4B5563;border-radius:5px;}"
        )
        self.result_history_widget = QWidget()
        self.result_history_grid = QGridLayout(self.result_history_widget)
        self.result_history_grid.setContentsMargins(8, 8, 8, 8)
        self.result_history_grid.setHorizontalSpacing(6)
        self.result_history_grid.setVerticalSpacing(6)
        self.result_history_area.setWidget(self.result_history_widget)
        status_layout.addWidget(self.result_history_area)

        self.online_acc_info = CaptionLabel("\u5728\u7ebf\u51c6\u786e\u7387: -")
        self.online_acc_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.online_acc_info)

        self.train_info = CaptionLabel("\u8bad\u7ec3\u8fdb\u5ea6: 0/0")
        self.train_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.train_info)
        self.train_progress_bar = ProgressBar()
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        status_layout.addWidget(self.train_progress_bar)
        status_layout.addStretch()

        self.data_management_widget = self._build_data_management_widget(make_card, make_field_label)
    
    def _build_data_management_widget(self, make_card, make_field_label):
        page = QWidget()
        page.setStyleSheet(self.styleSheet())
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(12, 12, 12, 12)
        page_layout.setSpacing(0)
        data_panel = make_card()
        page_layout.addWidget(data_panel)
        data_layout = QVBoxLayout(data_panel)
        data_layout.setContentsMargins(12, 12, 12, 12)
        data_layout.setSpacing(10)

        data_title = StrongBodyLabel("\u8bad\u7ec3\u6570\u636e\u4e0e\u6743\u91cd\u7ba1\u7406")
        data_title.setObjectName("cardTitle")
        data_layout.addWidget(data_title)
        self.data_tabs = QTabWidget()
        self.data_tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #243449;border-radius:8px;}"
            "QTabBar::tab{background:#111827;color:#CBD5E1;padding:6px 12px;border:1px solid #243449;}"
            "QTabBar::tab:selected{background:#1E3355;color:#FFFFFF;}"
        )
        data_layout.addWidget(self.data_tabs, stretch=1)

        train_tab = QWidget()
        train_tab_layout = QVBoxLayout(train_tab)
        train_tab_layout.setContentsMargins(8, 8, 8, 8)
        train_tab_layout.setSpacing(8)
        self.data_tabs.addTab(train_tab, "\u8bad\u7ec3\u6570\u636e")

        score_tab = QWidget()
        score_layout = QVBoxLayout(score_tab)
        score_layout.setContentsMargins(8, 8, 8, 8)
        score_layout.setSpacing(8)
        self.data_tabs.addTab(score_tab, "\u6570\u636e\u8bc4\u5206")

        self.data_tabs.removeTab(self.data_tabs.indexOf(score_tab))

        train_op_row = QHBoxLayout()
        self.refresh_data_btn = PushButton("\u5237\u65b0\u6570\u636e")
        self.refresh_data_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.refresh_data_btn.clicked.connect(self.refresh_train_dataset_view)
        train_op_row.addWidget(self.refresh_data_btn)
        self.auto_pick_btn = PushButton("\u52fe\u9009\u5f53\u524d\u7a97\u53e3\u5168\u90e8")
        self.auto_pick_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.auto_pick_btn.clicked.connect(self.auto_check_high_quality_samples)
        train_op_row.addWidget(self.auto_pick_btn)
        self.score_pick_train_btn = PushButton("\u8bc4\u5206\u5e76\u52fe\u9009")
        self.score_pick_train_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.score_pick_train_btn.clicked.connect(self.score_and_check_train_samples)
        train_op_row.addWidget(self.score_pick_train_btn)
        self.score_pick_train_btn.setVisible(False)
        self.train_weight_btn = PushButton("\u7528\u52fe\u9009\u6570\u636e\u8bad\u7ec3\u6743\u91cd")
        self.train_weight_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.train_weight_btn.clicked.connect(self.train_weight_from_checked_files)
        train_op_row.addWidget(self.train_weight_btn)
        
        self.evaluate_checked_btn = PushButton("\u6d4b\u8bc4\u52fe\u9009\u6570\u636e")
        self.evaluate_checked_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.evaluate_checked_btn.clicked.connect(self.evaluate_checked_data)
        train_op_row.addWidget(self.evaluate_checked_btn)
        
        train_tab_layout.addLayout(train_op_row)

        self.train_data_tree = QTreeWidget()
        self.train_data_tree.setHeaderLabels(["\u65e5\u671f/\u6587\u4ef6", "\u6807\u7b7e", "\u91c7\u6837\u70b9", "\u8d28\u91cf"] )
        self.train_data_tree.setMinimumHeight(170)
        self.train_data_tree.setAlternatingRowColors(True)
        train_tab_layout.addWidget(self.train_data_tree)
        self.train_data_tree.setColumnHidden(3, True)

        weight_row = QHBoxLayout()
        weight_row.addWidget(make_field_label("\u5728\u7ebf\u6743\u91cd:"))
        self.weight_combo = QComboBox()
        weight_row.addWidget(self.weight_combo, stretch=1)
        self.load_weight_btn = PushButton("\u52a0\u8f7d\u6743\u91cd")
        self.load_weight_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 14px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.load_weight_btn.clicked.connect(self.load_selected_weight_file)
        weight_row.addWidget(self.load_weight_btn)
        train_tab_layout.addLayout(weight_row)

        self.weight_status = CaptionLabel("\u5f53\u524d\u6743\u91cd: \u9ed8\u8ba4")
        self.weight_status.setStyleSheet("font-size: 13px; color: #9FB0C9;")
        train_tab_layout.addWidget(self.weight_status)
        train_tab_layout.addStretch()

        score_op_row = QHBoxLayout()
        self.score_refresh_btn = PushButton("Scan and score")
        self.score_refresh_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.score_refresh_btn.clicked.connect(self.refresh_score_dataset_view)
        score_op_row.addWidget(self.score_refresh_btn)
        score_op_row.addWidget(make_field_label("\u6700\u4f4e\u5206:"))
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.0, 100.0)
        self.score_threshold_spin.setDecimals(1)
        self.score_threshold_spin.setSingleStep(5.0)
        self.score_threshold_spin.setValue(60.0)
        self.score_threshold_spin.setMinimumWidth(78)
        score_op_row.addWidget(self.score_threshold_spin)
        self.score_auto_btn = PushButton("Select high score")
        self.score_auto_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.score_auto_btn.clicked.connect(self.auto_check_scored_samples)
        score_op_row.addWidget(self.score_auto_btn)
        self.score_save_btn = PushButton("Save selected")
        self.score_save_btn.setStyleSheet("QPushButton{background:#15803D;color:#F0FDF4;border:1px solid #34D399;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#16A34A;}")
        self.score_save_btn.clicked.connect(self.save_checked_scored_samples)
        score_op_row.addWidget(self.score_save_btn)
        self.score_sync_btn = PushButton("Sync to train")
        self.score_sync_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.score_sync_btn.clicked.connect(self.sync_score_checked_to_train_samples)
        score_op_row.addWidget(self.score_sync_btn)
        score_op_row.addStretch()
        score_layout.addLayout(score_op_row)

        self.score_tree = QTreeWidget()
        self.score_tree.setColumnCount(11)
        self.score_tree.setHeaderLabels([
            "\u6587\u4ef6", "\u6807\u7b7e", "\u70b9\u6570", "FBCCA", "FB\u5206", "CCA", "CCA\u5206", "TDCA", "TDCA\u5206", "\u603b\u5206", "\u5efa\u8bae"
        ])
        self.score_tree.setMinimumHeight(260)
        self.score_tree.setAlternatingRowColors(True)
        self.score_tree.setSortingEnabled(True)
        score_layout.addWidget(self.score_tree, stretch=1)

        self.score_status = CaptionLabel("\u6570\u636e\u8bc4\u5206: \u672a\u626b\u63cf")
        self.score_status.setStyleSheet("font-size: 13px; color: #9FB0C9;")
        score_layout.addWidget(self.score_status)

        self.refresh_train_dataset_view()
        self.refresh_weight_file_list()
        return page

    def _stim_freq_summary(self):
        pairs = []
        for i, freq in enumerate(self.sti_lst):
            cmd = self.commands[i] if i < len(self.commands) else f"C{i + 1}"
            period = self.stim_period_frames[i] if i < len(self.stim_period_frames) else None
            if period is None:
                pairs.append(f"{cmd}:{freq:.2f}Hz")
            else:
                pairs.append(f"{cmd}:{freq:.2f}Hz/{period}\u5e27")
        return " | ".join(pairs)

    def _apply_stimulus_profile_to_rects(self):
        for i, rect in enumerate(getattr(self, "sti_rects", [])):
            if i >= len(self.sti_lst):
                continue
            period = self.stim_period_frames[i] if i < len(self.stim_period_frames) else None
            duty = self.stim_duty_frames[i] if i < len(self.stim_duty_frames) else None
            rect.setFrequencyProfile(self.sti_lst[i], period_frames=period, duty_frames=duty)

    def _current_stim_freqs_array(self):
        return np.asarray(self.sti_lst, dtype=float).reshape(-1)

    def _freqs_compatible(self, freqs, allow_legacy_without_meta=False):
        try:
            if freqs is None:
                if allow_legacy_without_meta:
                    return np.allclose(self._current_stim_freqs_array(), LEGACY_CAR_FREQS, atol=0.05)
                return False
            arr = np.asarray(freqs, dtype=float).reshape(-1)
            cur = self._current_stim_freqs_array()
            return arr.shape == cur.shape and np.allclose(arr, cur, atol=0.03, rtol=0.0)
        except Exception:
            return False

    def _mat_freqs_compatible(self, mat):
        if not isinstance(mat, dict) or "stim_freqs_hz" not in mat:
            return self._freqs_compatible(None, allow_legacy_without_meta=True)
        return self._freqs_compatible(mat.get("stim_freqs_hz"), allow_legacy_without_meta=False)

    def setDefaultColor(self):
        for rect in self.sti_rects:
            rect.setDefaultColor()

    def _now_mono(self):
        return time.perf_counter()

    def _enter_online_phase(self, phase):
        self.online_phase = str(phase)
        ts_mono = self._now_mono()
        ts_unix = time.time()
        self.online_phase_start_ts = ts_mono
        if isinstance(self.online_trial_meta, dict):
            self.online_trial_meta[f"{phase}_onset_monotonic"] = float(ts_mono)
            self.online_trial_meta[f"{phase}_onset_unix"] = float(ts_unix)

    def _enter_training_phase(self, phase):
        self.training_phase = str(phase)
        ts_mono = self._now_mono()
        ts_unix = time.time()
        self.training_phase_start_ts = ts_mono
        if isinstance(self.training_trial_meta, dict):
            self.training_trial_meta[f"{phase}_onset_monotonic"] = float(ts_mono)
            self.training_trial_meta[f"{phase}_onset_unix"] = float(ts_unix)

    def _phase_elapsed(self, start_ts):
        return max(0.0, self._now_mono() - float(start_ts))

    def _frame_index_from_onset(self, onset_mono):
        elapsed = max(0.0, self._now_mono() - float(onset_mono))
        return int(np.floor(elapsed * float(self.stim_refresh_hz)))

    def _begin_stim_timing_log(self, scope):
        self.stim_timing_scope = str(scope)
        self.stim_timing_log = []
        self.stim_timing_last_frame = None
        self.stim_timing_last_mono = None
        self.stim_timing_started_mono = self._now_mono()

    def _record_stim_timing(self, frame_index):
        now_mono = self._now_mono()
        frame_index = int(frame_index)
        if self.stim_timing_last_frame is None:
            frame_delta = 0
            dt_ms = 0.0
        else:
            frame_delta = frame_index - int(self.stim_timing_last_frame)
            dt_ms = (now_mono - float(self.stim_timing_last_mono)) * 1000.0
        self.stim_timing_last_frame = frame_index
        self.stim_timing_last_mono = now_mono
        self.stim_timing_log.append((frame_index, frame_delta, dt_ms))

    def _stim_timing_summary(self):
        if not self.stim_timing_log:
            return {
                "scope": self.stim_timing_scope,
                "render_events": 0,
                "skipped_frames": 0,
                "max_frame_delta": 0,
                "max_render_interval_ms": 0.0,
                "mean_render_interval_ms": 0.0,
                "effective_render_hz": 0.0,
                "estimated_freqs_hz": [],
            }
        deltas = np.asarray([x[1] for x in self.stim_timing_log[1:]], dtype=float)
        intervals = np.asarray([x[2] for x in self.stim_timing_log[1:]], dtype=float)
        skipped = np.maximum(deltas - 1.0, 0.0)
        span_sec = float(np.sum(intervals) / 1000.0) if intervals.size else 0.0
        frames = [int(x[0]) for x in self.stim_timing_log]
        estimated_freqs = []
        for period, duty in zip(getattr(self, "stim_period_frames", []), getattr(self, "stim_duty_frames", [])):
            try:
                period = int(period)
                duty = max(1, min(int(duty), period - 1 if period > 1 else 1))
                states = [(frame % period) < duty for frame in frames]
                transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
                estimated_freqs.append(float(transitions / (2.0 * span_sec)) if span_sec > 0 else 0.0)
            except Exception:
                estimated_freqs.append(0.0)
        return {
            "scope": self.stim_timing_scope,
            "render_events": int(len(self.stim_timing_log)),
            "skipped_frames": int(np.sum(skipped)) if skipped.size else 0,
            "max_frame_delta": int(np.max(deltas)) if deltas.size else 0,
            "max_render_interval_ms": float(np.max(intervals)) if intervals.size else 0.0,
            "mean_render_interval_ms": float(np.mean(intervals)) if intervals.size else 0.0,
            "effective_render_hz": float(1000.0 / np.mean(intervals)) if intervals.size and np.mean(intervals) > 0 else 0.0,
            "estimated_freqs_hz": estimated_freqs,
        }

    def _print_stim_frequency_check(self, label, stim_timing):
        try:
            expected = [float(x) for x in getattr(self, "sti_lst", [])]
            estimated = [float(x) for x in stim_timing.get("estimated_freqs_hz", [])]
            pairs = []
            for i, exp in enumerate(expected):
                est = estimated[i] if i < len(estimated) else 0.0
                err = est - exp
                name = self.commands[i] if i < len(self.commands) else f"C{i + 1}"
                pairs.append(f"{name}: expected={exp:.3f}Hz actual~={est:.3f}Hz err={err:+.3f}")
            print("[STIM_FREQ_CHECK] " + json.dumps({
                "label": str(label),
                "refresh_hz": float(self.stim_refresh_hz),
                "render_events": int(stim_timing.get("render_events", 0)),
                "skipped_frames": int(stim_timing.get("skipped_frames", 0)),
                "max_frame_delta": int(stim_timing.get("max_frame_delta", 0)),
                "max_render_interval_ms": round(float(stim_timing.get("max_render_interval_ms", 0.0)), 3),
                "mean_render_interval_ms": round(float(stim_timing.get("mean_render_interval_ms", 0.0)), 3),
                "effective_render_hz": round(float(stim_timing.get("effective_render_hz", 0.0)), 3),
                "freqs": pairs,
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(f"[STIM_FREQ_CHECK] failed: {exc}", flush=True)

    def _render_stim_by_frame(self, frame_index):
        self._record_stim_timing(frame_index)
        for rect in self.sti_rects:
            rect.changeColorByFrame(rect.sti, frame_index, self.stim_refresh_hz)
        if hasattr(self, "stimulus_canvas"):
            self.stimulus_canvas.request_update()

    def _start_stim_render_timer(self):
        if hasattr(self, "stim_render_timer") and not self.stim_render_timer.isActive():
            self.stim_render_timer.start()

    def _stop_stim_render_timer(self):
        if hasattr(self, "stim_render_timer") and self.stim_render_timer.isActive():
            self.stim_render_timer.stop()

    def _stim_render_tick(self):
        if self.training_phase == "stim" and self.training_stim_onset_mono > 0:
            cur_frame = self._frame_index_from_onset(self.training_stim_onset_mono)
            if cur_frame != self.training_last_render_frame_idx:
                self.training_stim_frame_idx = cur_frame
                self.training_last_render_frame_idx = cur_frame
                self._render_stim_by_frame(cur_frame)
            return
        if self.online_phase == "stim" and self.online_stim_onset_mono > 0:
            cur_frame = self._frame_index_from_onset(self.online_stim_onset_mono)
            if cur_frame != self.online_last_render_frame_idx:
                self.online_stim_frame_idx = cur_frame
                self.online_last_render_frame_idx = cur_frame
                self._render_stim_by_frame(cur_frame)
            return
        self._stop_stim_render_timer()

    def _stimulus_is_active(self):
        return (
            getattr(self, "training_phase", "idle") == "stim"
            or getattr(self, "online_phase", "idle") == "stim"
        )

    def _refresh_network_status(self):
        ip, port = self._get_robot_endpoint()
        robot_enabled = bool(getattr(self, "robot_control_enabled", False))
        if not robot_enabled:
            self.robot_net_ok = False
            self.network_status.setText("\u7f51\u7edc\u72b6\u6001: \u6570\u636e\u6d4b\u8bd5\u6a21\u5f0f\uff08\u4e0d\u53d1\u9001\u5c0f\u8f66\u6307\u4ee4\uff09")
            self.network_status.setStyleSheet("font-size: 13px; color: #60A5FA; font-weight: 600;")
            if hasattr(self, "top_network_status"):
                self.top_network_status.setText("\u8fde\u63a5\u72b6\u6001: \u6570\u636e\u6d4b\u8bd5")
                self.top_network_status.setObjectName("statusTagOk")
                self.top_network_status.style().unpolish(self.top_network_status)
                self.top_network_status.style().polish(self.top_network_status)
            if hasattr(self, "test_conn_btn"):
                self.test_conn_btn.setText("\u8fde\u63a5\u5c0f\u8f66")
            return

        robot_text = "\u63a7\u5236\u5df2\u8fde\u63a5" if self.robot_net_ok else "\u63a7\u5236\u672a\u8fde\u63a5"
        self.network_status.setText(f"\u7f51\u7edc\u72b6\u6001: {robot_text}({ip}:{port})")
        self.network_status.setStyleSheet(
            "font-size: 13px; color: #10B981; font-weight: 600;" if self.robot_net_ok
            else "font-size: 13px; color: #EF4444; font-weight: 600;"
        )
        if hasattr(self, "top_network_status"):
            self.top_network_status.setText(f"\u8fde\u63a5\u72b6\u6001: {robot_text}")
            self.top_network_status.setObjectName("statusTagOk" if self.robot_net_ok else "statusTagBad")
            self.top_network_status.style().unpolish(self.top_network_status)
            self.top_network_status.style().polish(self.top_network_status)
        if hasattr(self, "test_conn_btn"):
            self.test_conn_btn.setText("\u65ad\u5f00\u5c0f\u8f66" if self.robot_net_ok else "\u8fde\u63a5\u5c0f\u8f66")

    def _get_robot_endpoint(self):
        ip = ROBOT_IP
        port = PORT
        if hasattr(self, "robot_ip_edit"):
            txt = self.robot_ip_edit.text().strip()
            if len(txt) > 0:
                ip = txt
        if hasattr(self, "robot_port_spin"):
            try:
                port = int(self.robot_port_spin.value())
            except Exception:
                port = PORT
        return ip, port

    def _toggle_robot_connection(self):
        if bool(getattr(self, "robot_control_enabled", False)):
            self.robot_control_enabled = False
            self.robot_net_ok = False
            self._refresh_network_status()
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u5c0f\u8f66\u63a7\u5236\u5df2\u5173\u95ed\uff0c\u4ec5\u6d4b\u8bd5\u6570\u636e")
            return
        self._connect_robot_control()

    def _connect_robot_control(self):
        ip, port = self._get_robot_endpoint()
        try:
            client = RobotClient(ip, port)
            client.connect()
            if client.connected:
                self.robot_control_enabled = True
                self.robot_net_ok = True
                self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u63a7\u5236\u8fde\u63a5\u6210\u529f {ip}:{port}")
            else:
                self.robot_control_enabled = False
                self.robot_net_ok = False
                self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u63a7\u5236\u8fde\u63a5\u5931\u8d25 {ip}:{port}")
            client.close()
        except Exception:
            self.robot_control_enabled = False
            self.robot_net_ok = False
            self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u63a7\u5236\u8fde\u63a5\u5931\u8d25 {ip}:{port}")
        self._refresh_network_status()

    def _test_robot_connection(self):
        self._connect_robot_control()

    def _update_live_panel(self):
        self.command_info.setText(f"\u5f53\u524d\u6307\u4ee4: {self.current_candidate}")
        bar_val = int(max(0.0, min(1.0, self.current_confidence / 0.25)) * 100)
        self.confidence_bar.setValue(bar_val)
        self.confidence_info.setText(f"\u7f6e\u4fe1\u5ea6: {self.current_confidence:.4f}")
        color_map = {
            "\u524d\u8fdb": "#3B82F6",
            "\u540e\u9000": "#8B5CF6",
            "\u5de6\u8f6c": "#F59E0B",
            "\u53f3\u8f6c": "#0EA5E9",
            "\u505c\u6b62": "#EF4444",
        }
        if len(self.current_votes) > 0:
            tags = []
            for cmd in self.current_votes[-3:]:
                color = color_map.get(cmd, "#64748B")
                tags.append(
                    f"<span style='background:{color}; color:#FFFFFF; border-radius:8px; padding:2px 8px; margin-right:4px;'>"
                    f"{cmd}</span>"
                )
            self.vote_info.setText("\u6700\u8fd1\u6295\u7968: " + " ".join(tags))
        else:
            self.vote_info.setText("\u6700\u8fd1\u6295\u7968: -")

    def _append_result_history(self, command, idx, confidence=None, sent_to_robot=False):
        self.result_history_count += 1
        conf_text = "-" if confidence is None else f"{float(confidence):.3f}"
        self.result_history.append({
            "seq": int(self.result_history_count),
            "command": str(command),
            "idx": int(idx),
            "confidence": conf_text,
            "sent": bool(sent_to_robot),
        })
        self._refresh_result_history_grid()

    def _refresh_result_history_grid(self):
        if not hasattr(self, "result_history_grid"):
            return
        while self.result_history_grid.count() > 0:
            item = self.result_history_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        color_map = {
            "\u524d\u8fdb": "#2563EB",
            "\u540e\u9000": "#7C3AED",
            "\u5de6\u8f6c": "#D97706",
            "\u53f3\u8f6c": "#0284C7",
            "\u505c\u6b62": "#DC2626",
        }
        area_width = self.result_history_area.viewport().width() if hasattr(self, "result_history_area") else 520
        columns = max(3, min(6, int(area_width // 122)))
        for pos, entry in enumerate(self.result_history):
            command = entry.get("command", "-")
            bg = color_map.get(command, "#334155")
            border = "#86EFAC" if entry.get("sent") else "#3B82F6"
            text = f"{entry.get('seq', 0):02d} {command}  {entry.get('confidence', '-')}"
            chip = QLabel(text)
            chip.setAlignment(Qt.AlignCenter)
            chip.setMinimumWidth(112)
            chip.setFixedHeight(28)
            chip.setStyleSheet(
                f"QLabel{{background:{bg}; color:#FFFFFF; border:1px solid {border}; "
                "border-radius:8px; padding:3px 8px; font-size:13px; font-weight:700;}}"
            )
            self.result_history_grid.addWidget(chip, pos // columns, pos % columns)
        if hasattr(self, "result_history_area"):
            bar = self.result_history_area.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _reset_collection_state(self):
        self.finish = True
        self.start_cache = False
        self.start_flick = False
        self.online_window_active = False
        self.online_phase = "idle"
        self.online_phase_start_ts = 0.0
        self.online_stim_frame_idx = 0
        self.online_last_render_frame_idx = -1
        self.online_stim_onset_mono = 0.0
        self.online_stim_onset_unix = 0.0
        self.online_trial_meta = {}
        self.training_collecting = False
        self.training_phase = "idle"
        self.training_phase_start_ts = 0.0
        self.training_stim_frame_idx = 0
        self.training_last_render_frame_idx = -1
        self.training_stim_onset_mono = 0.0
        self.training_stim_onset_unix = 0.0
        self.training_pending_success = False
        self.training_trial_meta = {}
        self.training_target_label = ""
        self.online_timer.stop()
        self.training_timer.stop()
        self._stop_stim_render_timer()
        self._clear_cache_buffer()
        self.sliding_cycle_count = 0
        self.sliding_cycle_start_mono = 0.0
        self.setDefaultColor()
        self.progress.setValue(0)

    def _clear_cache_buffer(self):
        lock = getattr(self, "eeg_buffer_lock", None)
        if lock is None:
            self.eeg_buffer_lock = threading.RLock()
            lock = self.eeg_buffer_lock
        with lock:
            if not hasattr(self, "sample_buffer"):
                self.sample_buffer = EegSampleBuffer()
            self.sample_buffer.clear()
            self.cache_data = np.array([])
            self.cache_chunks = []
            self.cache_points = 0

    def _append_cache_chunk(self, data):
        lock = getattr(self, "eeg_buffer_lock", None)
        if lock is None:
            self.eeg_buffer_lock = threading.RLock()
            lock = self.eeg_buffer_lock
        with lock:
            if not hasattr(self, "sample_buffer"):
                self.sample_buffer = EegSampleBuffer()
            if self.sample_buffer.append(data):
                self.cache_chunks = []
                self.cache_points = self.sample_buffer.points

    def _cache_sample_count(self):
        lock = getattr(self, "eeg_buffer_lock", None)
        if lock is None:
            if hasattr(self, "sample_buffer"):
                return int(self.sample_buffer.points)
            return int(self.cache_points)
        with lock:
            if hasattr(self, "sample_buffer"):
                return int(self.sample_buffer.points)
            return int(self.cache_points)

    def _materialize_cache_data(self):
        lock = getattr(self, "eeg_buffer_lock", None)
        if lock is None:
            if hasattr(self, "sample_buffer"):
                return self.sample_buffer.materialize()
            return np.array([])
        with lock:
            if hasattr(self, "sample_buffer"):
                return self.sample_buffer.materialize()
            return np.array([])

    def _extract_aligned_window(self, full_data, recv_start_mono, stim_onset_mono, sample_points):
        return extract_aligned_window(
            full_data,
            recv_start_mono,
            stim_onset_mono,
            sample_points,
            AcquisitionConfig(self.sample_rate_hz, 0.0),
        )

    def _aligned_window_indices(self, recv_start_mono, stim_onset_mono, sample_points):
        return aligned_window_indices(
            recv_start_mono,
            stim_onset_mono,
            sample_points,
            AcquisitionConfig(self.sample_rate_hz, 0.0),
        )

    def _has_aligned_window_ready(self, full_data, recv_start_mono, stim_onset_mono, sample_points):
        used_data = self._extract_aligned_window(
            full_data=full_data,
            recv_start_mono=recv_start_mono,
            stim_onset_mono=stim_onset_mono,
            sample_points=int(sample_points),
        )
        return isinstance(used_data, np.ndarray) and used_data.ndim == 2 and used_data.shape[-1] >= int(sample_points)

    def _benchmark_required_trial_points(self, stim_window_sec=None):
        stim_window_sec = float(self.training_window_sec if stim_window_sec is None else stim_window_sec)
        pre_pts = int(round(BENCHMARK_PRE_STIMULUS_SEC * self.sample_rate_hz))
        delay_pts = int(round(BENCHMARK_VISUAL_DELAY_SEC * self.sample_rate_hz))
        stim_pts = int(round(stim_window_sec * self.sample_rate_hz))
        try:
            model = TDCA(
                BENCHMARK_NUM_HARMONICS,
                stim_window_sec + BENCHMARK_VISUAL_DELAY_SEC,
                self.sti_lst,
                sample_rate=self.sample_rate_hz,
                delay_sec=BENCHMARK_VISUAL_DELAY_SEC,
            )
            lag_pts = int(getattr(model, "lagging_len", 0))
        except Exception:
            lag_pts = 8
        return int(pre_pts + delay_pts + stim_pts + max(0, lag_pts))

    def _benchmark_trial_indices(self, recv_start_mono, stim_onset_mono, stim_window_sec=None):
        total_pts = self._benchmark_required_trial_points(stim_window_sec)
        start_mono = float(stim_onset_mono) - BENCHMARK_PRE_STIMULUS_SEC
        start_idx = int(round((start_mono - float(recv_start_mono)) * self.sample_rate_hz))
        return start_idx, start_idx + total_pts

    def _extract_benchmark_trial(self, full_data, recv_start_mono, stim_onset_mono, stim_window_sec=None):
        if not isinstance(full_data, np.ndarray) or full_data.ndim != 2:
            return None, (-1, -1)
        start_idx, end_idx = self._benchmark_trial_indices(recv_start_mono, stim_onset_mono, stim_window_sec)
        if start_idx < 0 or end_idx > int(full_data.shape[-1]):
            return None, (start_idx, end_idx)
        return np.asarray(full_data[:, start_idx:end_idx], dtype=float), (start_idx, end_idx)

    def _debug_alignment_failure(self, scope, full_data, recv_start_mono, stim_onset_mono, sample_points, meta=None):
        try:
            is_array_2d = isinstance(full_data, np.ndarray) and full_data.ndim == 2
            n_points = int(full_data.shape[-1]) if is_array_2d else -1
            req_points = int(sample_points)
            idx = self._aligned_window_indices(recv_start_mono, stim_onset_mono, req_points)
            start_idx = int(idx[0]) if isinstance(idx, tuple) else None
            end_idx = int(idx[1]) if isinstance(idx, tuple) else None

            if not is_array_2d:
                reason = "full_data_is_not_2d_array"
            elif n_points < req_points:
                reason = "buffer_shorter_than_analysis_window"
            elif idx is None:
                reason = "aligned_index_is_none"
            elif start_idx < 0:
                reason = "aligned_window_starts_before_buffer"
            elif end_idx > n_points:
                reason = "aligned_window_ends_after_buffer"
            else:
                reason = "aligned_slice_invalid_for_unknown_reason"

            target_start_mono = float(stim_onset_mono)
            payload = {
                "scope": str(scope),
                "reason": reason,
                "sample_rate_hz": int(self.sample_rate_hz),
                "analysis_delay_sec": float(self.analysis_delay_sec),
                "analysis_window_sec": float(self.analysis_window_sec),
                "trial_window_sec": float(self.training_window_sec if str(scope).startswith("training") else self.online_window_sec),
                "required_analysis_points": int(req_points),
                "buffer_points": int(n_points),
                "aligned_start_idx": start_idx,
                "aligned_end_idx": end_idx,
                "missing_points": int(max(0, (end_idx or 0) - n_points)),
                "recv_start_mono": float(recv_start_mono),
                "stim_onset_mono": float(stim_onset_mono),
                "target_start_mono": float(target_start_mono),
                "target_offset_sec": float(target_start_mono - float(recv_start_mono)),
            }
            if isinstance(meta, dict):
                for key in (
                    "expected_samples",
                    "analysis_samples",
                    "recv_points",
                    "recv_chunks",
                    "recv_end_monotonic",
                    "stim_end_monotonic",
                ):
                    if key in meta:
                        payload[key] = meta.get(key)
            line = "[ALIGN_DEBUG] " + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", "align_debug.log"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
        except Exception as exc:
            line = f"[ALIGN_DEBUG] failed_to_build_debug_payload: {exc}"
            print(line, flush=True)
            try:
                os.makedirs("logs", exist_ok=True)
                with open(os.path.join("logs", "align_debug.log"), "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
            except Exception:
                pass

    def _prepare_model_input(self, data):
        return preprocess_model_input(data)

    def _on_mode_changed(self, index):
        if self.training_collecting or self.start_flick:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex({"online": 0, "train": 1, "test": 2}.get(self.mode, 0))
            self.mode_combo.blockSignals(False)
            return
        self.mode = "online" if index == 0 else ("train" if index == 1 else "test")
        self._reset_collection_state()
        if self.mode == "online":
            self.show_label.setText("\u5728\u7ebf\u6a21\u5f0f\u5df2\u9009\u62e9")
            self.online_time_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            if hasattr(self, "training_time_combo"):
                self.training_time_combo.setEnabled(True)
            if hasattr(self, "sampling_rate_combo"):
                self.sampling_rate_combo.setEnabled(False)
            self._reset_online_accuracy_stats()
            self.training_hint.setText(self._online_hint_text())
            self.refresh_weight_file_list()
        elif self.mode == "train":
            self.show_label.setText("\u8bad\u7ec3\u6a21\u5f0f\u5df2\u9009\u62e9")
            self.online_time_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            if hasattr(self, "training_time_combo"):
                self.training_time_combo.setEnabled(True)
            if hasattr(self, "sampling_rate_combo"):
                self.sampling_rate_combo.setEnabled(False)
            self._reset_online_accuracy_stats()
            self.training_hint.setText(
                f"\u8bad\u7ec3\u6a21\u5f0f: \u56de\u8f66\u542f\u52a8 trial(\u63d0\u793a{self.trial_cue_sec:.1f}s -> \u523a\u6fc0{self.training_window_sec:.2f}s -> \u95f4\u9694{self.trial_rest_sec:.1f}s)"
            )
            self.training_collected = 0
            self.training_plan = self._build_training_plan()
            self._update_training_progress()
            self.refresh_train_dataset_view()
        else:
            self.show_label.setText("\u6d4b\u8bd5\u6a21\u5f0f\u5df2\u9009\u62e9")
            self.online_time_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            if hasattr(self, "training_time_combo"):
                self.training_time_combo.setEnabled(True)
            if hasattr(self, "sampling_rate_combo"):
                self.sampling_rate_combo.setEnabled(False)
            self._reset_test_accuracy_stats(reset_plan=True)
            self.training_hint.setText(self._online_hint_text())


            self.refresh_weight_file_list()

    def _create_classifier(self, times_sec):
        nh = getattr(self, 'n_channels', 8)
        if self.model_name == "FBCCA":
            return FBCCA(3, times_sec, self.sti_lst, Nh=nh, sample_rate=self.sample_rate_hz)
        if self.model_name == "CCA":
            return CCA(3, times_sec, self.sti_lst, Nh=nh, sample_rate=self.sample_rate_hz)
        return TDCA(3, times_sec, self.sti_lst, Nh=nh, sample_rate=self.sample_rate_hz)

    def _online_analysis_sec(self):
        cycle = float(getattr(self, "online_window_sec", 3.0))
        if getattr(self, "online_strategy", "sliding_vote") == "sliding_vote":
            # 滑窗模式，子窗口比总窗口短 1.0s，由0.5s步长滑动刚好产生3次投票
            return float(max(0.2, cycle - 1.0))
        return float(max(0.2, cycle))

    def _online_strategy_label(self):
        if getattr(self, "online_strategy", "sliding_vote") == "async_fbcca":
            return "异步单窗"
        return "滑窗投票"

    def _online_hint_text(self):
        return (
            f"在线模式: 回车启动/停止 | {self._online_strategy_label()} | "
            f"采集{self.online_window_sec:.1f}s后输出 | "
            f"休息{self.trial_rest_sec:.1f}s后自动下一轮"
        )

    def _create_online_fbcca_classifier(self, window_sec=None):
        nh = getattr(self, 'n_channels', 8)
        win_sec = self._online_analysis_sec() if window_sec is None else float(window_sec)
        clf = FBCCA(3, win_sec, self.sti_lst, Nh=nh, sample_rate=self.sample_rate_hz)
        try:
            if hasattr(self.fbcca, "get_frequency_weights"):
                weights = self.fbcca.get_frequency_weights()
                if len(weights) == clf.Nf:
                    clf.set_frequency_weights(weights)
        except Exception:
            pass
        return clf

    def _create_sliding_classifier(self):
        """FBCCA classifier for online sub-windows."""
        return self._create_online_fbcca_classifier(self._online_analysis_sec())

    def _update_sample_points(self):
        self.analysis_delay_sec = 0.25
        online_times = 4.0
        try:
            if hasattr(self, "online_time_combo"):
                online_times = float(str(self.online_time_combo.currentText()).replace("s", "").strip())
        except Exception:
            online_times = 4.0
        training_times = 4.0
        try:
            if hasattr(self, "training_time_combo"):
                training_times = float(str(self.training_time_combo.currentText()).replace("s", "").strip())
        except Exception:
            training_times = 4.0
        self.model_times_sec = float(np.clip(online_times, 1.0, 4.0))
        self.analysis_window_sec = self.model_times_sec - self.analysis_delay_sec
        self.training_window_sec = float(np.clip(training_times, 1.0, 4.0))
        self.online_window_sec = self.model_times_sec
        self.online_analysis_window_sec = self._online_analysis_sec()
        self.training_sample_points = max(1, int(round(self.training_window_sec * self.sample_rate_hz)))
        self.online_sample_points = max(1, int(round(self.model_times_sec * self.sample_rate_hz)))

    def _rebuild_classifier_with_adaptation(self):
        self._update_sample_points()
        self.fbcca = self._create_classifier(self.model_times_sec)
        self.fbcca_sliding = None

        if self.model_name == "FBCCA":
            self.online_score_ema_alpha = 0.06
            self.online_warmup_windows = 2
        else:
            self.online_score_ema_alpha = 0.08
            self.online_warmup_windows = 3

        self._reload_selected_weights_if_any()
        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())

    def _online_status_text(self):
        return (
            f"FBCCA在线: 策略 {self._online_strategy_label()} | 采集/输出 {self.online_window_sec:.2f}s"
            f"(+\u5ef6\u8fdf {self.analysis_delay_sec:.2f}s) | EEG {self.sample_rate_hz}Hz | \u5237\u65b0\u7387 {self.stim_refresh_hz:.1f}Hz | "
            f"\u523a\u6fc0\u9891\u7387: {self._stim_freq_summary()}"
        )

    def _reload_selected_weights_if_any(self):
        selected_weight_file = getattr(self, "selected_weight_file", "")
        if selected_weight_file and os.path.exists(selected_weight_file):
            try:
                with open(selected_weight_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg_model = str(cfg.get("model_name", "")).upper()
                cfg_fs = int(cfg.get("sample_rate_hz", self.sample_rate_hz))
                cfg_freqs = cfg.get("stim_freqs_hz", None)
                if not self._freqs_compatible(cfg_freqs, allow_legacy_without_meta=True):
                    self.weight_status.setText(
                        "\u5f53\u524d\u6743\u91cd: \u523a\u6fc0\u9891\u7387\u4e0d\u5339\u914d\uff0c\u8bf7\u7528\u5f53\u524d\u5237\u65b0\u7387\u91cd\u65b0\u91c7\u96c6/\u8bad\u7ec3"
                    )
                    if hasattr(self.fbcca, "reset_frequency_weights"):
                        self.fbcca.reset_frequency_weights()
                    self.fbcca_sliding = None
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                    return
                if cfg_model and cfg_model != self.model_name:
                    self.weight_status.setText(
                        f"\u5f53\u524d\u6743\u91cd: \u6a21\u578b\u4e0d\u5339\u914d({cfg_model} != {self.model_name})"
                    )
                    if hasattr(self.fbcca, "reset_frequency_weights"):
                        self.fbcca.reset_frequency_weights()
                    self.fbcca_sliding = None
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                    return
                if cfg_fs != self.sample_rate_hz:
                    self.weight_status.setText(
                        f"\u5f53\u524d\u6743\u91cd: \u91c7\u6837\u7387\u4e0d\u5339\u914d({cfg_fs}Hz != {self.sample_rate_hz}Hz)"
                    )
                    if hasattr(self.fbcca, "reset_frequency_weights"):
                        self.fbcca.reset_frequency_weights()
                    self.fbcca_sliding = None
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                    return
                weights = np.asarray(cfg.get("weights", []), dtype=float)
                self.fbcca.set_frequency_weights(weights)
                self.fbcca_sliding = None
                scale = np.asarray(cfg.get("class_score_scale", np.ones(self.fbcca.Nf)), dtype=float)
                if scale.shape[0] == self.fbcca.Nf:
                    self.class_score_scale = np.clip(scale, 0.5, 2.0)
                else:
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                if self.model_name == "TDCA":
                    train_files = cfg.get("train_files", [])
                    ok, msg = self._fit_tdca_from_files(train_files)
                    if not ok:
                        self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: TDCA\u52a0\u8f7d\u5931\u8d25({msg})")
            except Exception:
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)

    def _create_training_classifier(self):
        return self._create_classifier(float(self.training_window_sec))

    def _tdca_required_points(self):
        if self.model_name != "TDCA":
            return int(self.training_sample_points)
        try:
            model = TDCA(3, float(self.training_window_sec), self.sti_lst, Nh=getattr(self, 'n_channels', 8), sample_rate=self.sample_rate_hz)
            required_points = int(getattr(model, "required_points", self.training_sample_points))
        except Exception:
            required_points = int(getattr(self.fbcca, "required_points", self.training_sample_points))
        return max(int(self.training_sample_points), required_points)

    def _fit_tdca_from_files(self, file_list):
        return self._training_framework(classifier=self.fbcca).fit_tdca_from_files(file_list)

    def _ensure_tdca_ready(self):
        if self.model_name != "TDCA":
            return True
        if getattr(self.fbcca, "is_fitted", False):
            return True
        self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: TDCA\u672a\u8bad\u7ec3\uff0c\u8bf7\u5148\u8bad\u7ec3\u6216\u52a0\u8f7d\u6743\u91cd")
        self.show_label.setText("TDCA\u672a\u8bad\u7ec3\uff0c\u8bf7\u5148\u8bad\u7ec3\u540e\u518d\u5728\u7ebf\u6d4b\u8bd5")
        return False

    def _apply_class_score_balance(self, scores):
        arr = np.asarray(scores, dtype=float).reshape(-1)
        if arr.size == 0:
            return arr
        scale = np.asarray(getattr(self, "class_score_scale", np.ones(arr.size)), dtype=float).reshape(-1)
        if scale.size != arr.size:
            scale = np.ones(arr.size, dtype=float)
        return arr / (scale + 1e-8)

    def _confidence_from_scores(self, scores):
        arr = np.asarray(scores, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            return float(arr[0])
        top2 = np.partition(arr, -2)[-2:]
        return float(top2[1] - top2[0])

    def _normalize_online_scores(self, scores):
        arr = np.asarray(scores, dtype=float).reshape(-1)
        if arr.size == 0:
            return arr

        arr = np.maximum(arr, 1e-8)
        if self.online_score_ema.shape[0] != arr.shape[0]:
            self.online_score_ema = np.ones(arr.shape[0], dtype=float)
            self.online_score_warmup = 0

        norm_scores = arr / (self.online_score_ema + 1e-8)
        alpha = float(self.online_score_ema_alpha)
        self.online_score_ema = (1.0 - alpha) * self.online_score_ema + alpha * arr
        self.online_score_warmup += 1
        return norm_scores

    def _on_model_changed(self, text):
        self.model_name = str(text).strip().upper() if str(text).strip() else "TDCA"
        if self.model_name not in ("TDCA", "FBCCA", "CCA"):
            self.model_name = "TDCA"

        if getattr(self, "mode", "online") in ("online", "test") and self.start_flick:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("Params changed, press Enter to restart")
            self.decision_info.setText("Recognition: stopped after parameter change")

        self._rebuild_classifier_with_adaptation()
        if getattr(self, "mode", "online") == "train":
            self._reset_training_plan_for_param_change()

        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if getattr(self, "mode", "online") in ("online", "test") and hasattr(self, "training_hint"):
            self.training_hint.setText(self._online_hint_text())

    def _on_sampling_rate_changed(self, text):
        fs = 250
        self.sample_rate_hz = fs

        if getattr(self, "mode", "online") in ("online", "test") and self.start_flick:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("Params changed, press Enter to restart")
            self.decision_info.setText("Recognition: stopped after model switch")

        self._rebuild_classifier_with_adaptation()
        if getattr(self, "mode", "online") == "train":
            self._reset_training_plan_for_param_change()

        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if getattr(self, "mode", "online") in ("online", "test") and hasattr(self, "training_hint"):
            self.training_hint.setText(self._online_hint_text())

    def _on_online_time_changed(self, text):
        if getattr(self, "mode", "online") in ("online", "test") and self.start_flick:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("Params changed, press Enter to restart")
            self.decision_info.setText("Recognition: stopped after window change")

        self._rebuild_classifier_with_adaptation()
        if getattr(self, "mode", "online") == "train":
            self._reset_training_plan_for_param_change()

        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if getattr(self, "mode", "online") in ("online", "test") and hasattr(self, "training_hint"):
            self.training_hint.setText(self._online_hint_text())

    def _on_training_time_changed(self, text):
        self._rebuild_classifier_with_adaptation()
        self._reset_training_plan_for_param_change()
        if getattr(self, "mode", "online") == "train" and hasattr(self, "training_hint"):
            self.training_hint.setText(
                f"Training mode: Enter starts trial (cue {self.trial_cue_sec:.1f}s -> stim {self.training_window_sec:.2f}s -> rest {self.trial_rest_sec:.1f}s)"
            )
        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if hasattr(self, "train_data_tree"):
            self.refresh_train_dataset_view()

    def _on_training_plan_params_changed(self, *args):
        self._reset_training_plan_for_param_change()

    def _reset_training_plan_for_param_change(self):
        if getattr(self, "training_collecting", False):
            return
        if not hasattr(self, "train_count_spin") or not hasattr(self, "train_labels_edit"):
            return
        self.training_collected = 0
        self.training_plan = self._build_training_plan()
        self.training_target_label = ""
        self.training_pending_success = False
        if hasattr(self, "train_info"):
            self._update_training_progress()
        if getattr(self, "mode", "online") == "train":
            self.show_label.setText("Training params updated, press Enter to start")
            self.decision_info.setText("Recognition: training plan reset")
            if hasattr(self, "training_hint"):
                self.training_hint.setText(
                    f"Training mode: Enter starts trial (cue {self.trial_cue_sec:.1f}s -> stim {self.training_window_sec:.2f}s -> rest {self.trial_rest_sec:.1f}s)"
                )
        elif getattr(self, "mode", "online") == "test":
            self._reset_test_accuracy_stats(reset_plan=True)
            self.show_label.setText("Test labels updated, press Enter to start")
            self.decision_info.setText("Recognition: test plan reset")
        if hasattr(self, "train_data_tree"):
            self.refresh_train_dataset_view()

    def _on_min_confidence_changed(self, value):
        try:
            self.min_confidence = float(value)
        except Exception:
            self.min_confidence = 0.02

    def _on_execution_mode_changed(self, index):
        self.online_strategy = "sliding_vote" if int(index) == 0 else "async_fbcca"
        self.execution_mode = self.online_strategy
        self.fbcca_sliding = None
        if hasattr(self, "training_hint"):
            self.training_hint.setText(self._online_hint_text())
        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())

    def _reset_online_accuracy_stats(self):
        self.online_eval_total = 0
        self.online_eval_correct = 0
        if not hasattr(self, "online_acc_info"):
            return
        if self.online_eval_truth == "not_counted":
            self.online_acc_info.setText("\u5728\u7ebf\u51c6\u786e\u7387: -")
        else:
            self.online_acc_info.setText(f"\u5728\u7ebf\u51c6\u786e\u7387: 0.00% (0/0) | \u771f\u503c: {self.online_eval_truth}")

    def _on_online_truth_changed(self, text):
        value = str(text).strip()
        self.online_eval_truth = value if value in self.commands else "not_counted"
        self._reset_online_accuracy_stats()

    def _reset_test_accuracy_stats(self, reset_plan=False):
        self.test_eval_total = 0
        self.test_eval_correct = 0
        self.test_eval_index = 0
        self.test_eval_results = []
        if reset_plan:
            self.test_eval_plan = self._build_training_plan()
        if hasattr(self, "online_acc_info"):
            total = len(getattr(self, "test_eval_plan", []))
            self.online_acc_info.setText(f"\u6d4b\u8bd5\u51c6\u786e\u7387: 0.00% (0/0) | \u8ba1\u5212: {total}")

    def _current_test_truth(self):
        plan = getattr(self, "test_eval_plan", [])
        idx = int(getattr(self, "test_eval_index", 0))
        if idx < 0 or idx >= len(plan):
            return ""
        return str(plan[idx])

    def _update_test_accuracy(self, pred_idx, confidence):
        truth = self._current_test_truth()
        truth_idx = self.commands.index(truth) if truth in self.commands else -1
        pred_idx = int(pred_idx)
        hit = truth_idx >= 0 and pred_idx == truth_idx
        self.test_eval_total += 1
        if hit:
            self.test_eval_correct += 1
        pred_text = self.commands[pred_idx] if 0 <= pred_idx < len(self.commands) else "-"
        acc = 100.0 * self.test_eval_correct / max(self.test_eval_total, 1)
        self.test_eval_results.append({
            "truth": truth,
            "pred": pred_text,
            "hit": bool(hit),
            "confidence": float(confidence),
            "window_sec": float(self.online_window_sec),
            "model": str(self.model_name),
        })
        if hasattr(self, "online_acc_info"):
            self.online_acc_info.setText(
                f"\u6d4b\u8bd5\u51c6\u786e\u7387: {acc:.2f}% ({self.test_eval_correct}/{self.test_eval_total}) | {truth}->{pred_text}"
            )
        print(
            f"[TEST_PROGRESS] {self.test_eval_total}/{len(self.test_eval_plan)} "
            f"truth={truth} pred={pred_text} hit={int(hit)} conf={float(confidence):.4f} "
            f"acc={acc:.2f}% window={self.online_window_sec:.2f}s model={self.model_name}",
            flush=True,
        )
        self.test_eval_index += 1
        return hit, acc

    def _update_online_accuracy(self, pred_idx):
        if self.online_eval_truth == "not_counted":
            return
        if self.online_eval_truth not in self.commands:
            return
        if pred_idx < 0 or pred_idx >= len(self.commands):
            return
        if not hasattr(self, "online_acc_info"):
            return

        truth_idx = self.commands.index(self.online_eval_truth)
        self.online_eval_total += 1
        if int(pred_idx) == int(truth_idx):
            self.online_eval_correct += 1
        acc = 100.0 * self.online_eval_correct / max(self.online_eval_total, 1)
        self.online_acc_info.setText(
            f"\u5728\u7ebf\u51c6\u786e\u7387: {acc:.2f}% ({self.online_eval_correct}/{self.online_eval_total}) | \u771f\u503c: {self.online_eval_truth}"
        )

    def _cooldown_ok(self, idx):
        now = time.time()
        if (now - self.last_command_time) < self.command_cooldown_sec and idx == self.last_command_idx:
            self.last_gate_reason = "\u51b7\u5374\u4e2d"
            return False
        self.last_command_time = now
        self.last_command_idx = idx
        return True

    def _build_training_plan(self):
        return build_training_plan(
            total=int(self.train_count_spin.value()),
            labels_text=self.train_labels_edit.text().strip(),
            default_labels=self.commands,
        )

    def _update_training_progress(self):
        total = len(self.training_plan)
        self.train_info.setText(f"\u8bad\u7ec3\u8fdb\u5ea6: {self.training_collected}/{total}")
        if hasattr(self, "train_progress_bar"):
            percent = int((self.training_collected / total) * 100) if total > 0 else 0
            self.train_progress_bar.setValue(max(0, min(100, percent)))

    def _subject_root(self):
        return self._training_framework().subject_root()

    def _train_root(self):
        return self._training_framework().train_root()

    def _weights_root(self):
        return self._training_framework().weights_root()

    def _extract_label_text(self, label_val):
        return extract_label_text(label_val)

    def _extract_label_idx(self, idx_val):
        return extract_int(idx_val, default=-1)

    def _training_framework(self, classifier=None):
        classifier = classifier if classifier is not None else self._create_training_classifier()
        return CarTrainingFramework(
            subject=config.subjectName,
            commands=self.commands,
            sample_rate_hz=self.sample_rate_hz,
            model_name=self.model_name,
            classifier=classifier,
            prepare_model_input=self._prepare_model_input,
            required_points_func=self._tdca_required_points if self.model_name == "TDCA" else lambda: self.training_sample_points,
        )

    def _score_model_once(self, model, sample):
        try:
            scores = np.asarray(model.score_vector(sample), dtype=float).reshape(-1)
            if scores.size == 0:
                return -1, 0.0
            pred = int(np.argmax(scores))
            if scores.size >= 2:
                top = float(scores[pred])
                second = float(np.partition(scores, -2)[-2])
                margin = max(0.0, (top - second) / (abs(top) + abs(second) + 1e-8))
            else:
                margin = 1.0
            return pred, float(np.clip(100.0 * margin, 0.0, 100.0))
        except Exception:
            return -1, 0.0

    def _current_bucket_training_files(self):
        root = self._train_root()
        if not os.path.exists(root):
            return []
        target = float(self.training_window_sec)
        files = []
        for fp in sorted(glob(os.path.join(root, "**", "*.mat"), recursive=True)):
            if "bad_samples" in fp.replace("\\", "/"):
                continue
            try:
                m = loadmat(fp, variable_names=["trial_stim_sec", "data", "label_idx", "sample_rate_hz", "stim_freqs_hz"])
                stim = float(np.asarray(m.get("trial_stim_sec", [[-1.0]])).reshape(-1)[0])
                sr = self._extract_label_idx(m.get("sample_rate_hz", self.sample_rate_hz))
                data = m.get("data", None)
                label_idx = self._extract_label_idx(m.get("label_idx", -1))
                if abs(stim - target) > 1e-6:
                    continue
                if not self._mat_freqs_compatible(m):
                    continue
                if sr > 0 and int(sr) != int(self.sample_rate_hz):
                    continue
                if not isinstance(data, np.ndarray) or data.ndim != 2:
                    continue
                if data.shape[-1] < int(self.training_sample_points * 0.95):
                    continue
                if label_idx < 0 or label_idx >= len(self.commands):
                    continue
                files.append(fp)
            except Exception:
                continue
        return files

    def _load_score_samples(self, files):
        rows = []
        for fp in files:
            try:
                m = loadmat(fp)
                if not self._mat_freqs_compatible(m):
                    continue
                data = np.asarray(m.get("data"), dtype=float)
                label_idx = self._extract_label_idx(m.get("label_idx", -1))
                if data.ndim != 2 or data.shape[-1] < int(self.training_sample_points * 0.95):
                    continue
                req_points = min(int(self.training_sample_points), data.shape[-1])
                sample = np.asarray(data[:, -req_points:], dtype=float)
                sample = self._prepare_model_input(sample)
                if not isinstance(sample, np.ndarray) or sample.ndim != 2:
                    continue
                rows.append({"fp": fp, "sample": sample, "label_idx": int(label_idx), "points": int(data.shape[-1])})
            except Exception:
                continue
        return rows

    def _tdca_leave_one_out_scores(self, rows, model_times):
        preds = [-1] * len(rows)
        margins = [0.0] * len(rows)
        if len(rows) < max(2, len(self.commands)):
            return preds, margins
        labels = np.asarray([row["label_idx"] for row in rows], dtype=int)
        
        # Check if we have enough coverage for TDCA
        unique_classes = set(labels.tolist())
        target_classes = len(self.commands)
        
        for i in range(len(rows)):
            train_rows = [row for j, row in enumerate(rows) if j != i]
            train_y = np.asarray([row["label_idx"] for row in train_rows], dtype=int)
            # Relax the restriction to allow TDCA evaluation even if some classes are entirely missing
            # However, TDCA still strictly needs at least 2 distinct classes to perform CCA correctly.
            if len(set(train_y.tolist())) < 2:
                continue
            try:
                model = TDCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
                
                # Override the default classes requirement for TDCA to adapt to missing data classes
                model.classes_ = np.unique(train_y)
                train_x = np.asarray([row["sample"] for row in train_rows], dtype=float)
                model.fit(train_x, train_y)
                pred, margin = self._score_model_once(model, rows[i]["sample"])
                preds[i] = int(pred)
                margins[i] = float(margin)
            except Exception:
                continue
        return preds, margins

    def _combine_detection_score(self, fb_hit, fb_margin, cca_hit, cca_margin, tdca_hit=False, tdca_margin=0.0, tdca_valid=False):
        if tdca_valid:
            accuracy_score = (
                (55.0 if fb_hit else 0.0)
                + (25.0 if cca_hit else 0.0)
                + (20.0 if tdca_hit else 0.0)
            )
            confidence_score = (
                0.55 * (float(fb_margin) if fb_hit else 0.0)
                + 0.25 * (float(cca_margin) if cca_hit else 0.0)
                + 0.20 * (float(tdca_margin) if tdca_hit else 0.0)
            )
        else:
            accuracy_score = (80.0 if fb_hit else 0.0) + (20.0 if cca_hit else 0.0)
            confidence_score = (
                0.80 * (float(fb_margin) if fb_hit else 0.0)
                + 0.20 * (float(cca_margin) if cca_hit else 0.0)
            )
        total_score = 0.85 * accuracy_score + 0.15 * confidence_score
        return (
            float(np.clip(accuracy_score, 0.0, 100.0)),
            float(np.clip(confidence_score, 0.0, 100.0)),
            float(np.clip(total_score, 0.0, 100.0)),
        )

    def _score_current_bucket_rows(self, include_tdca=True, files=None, progress_label="score"):
        if files is None:
            files = self._current_bucket_training_files()
        rows = self._load_score_samples(files)
        if len(rows) == 0:
            return []

        total_rows = len(rows)
        print(f"[SCORE_PROGRESS] {progress_label} start total={total_rows} include_tdca={include_tdca}", flush=True)
        model_times = float(self.analysis_window_sec + self.analysis_delay_sec)
        fbcca = FBCCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        cca = CCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        if include_tdca:
            print(f"[SCORE_PROGRESS] {progress_label} tdca_leave_one_out start", flush=True)
            tdca_preds, tdca_margins = self._tdca_leave_one_out_scores(rows, model_times)
            print(f"[SCORE_PROGRESS] {progress_label} tdca_leave_one_out done", flush=True)
        else:
            tdca_preds = [-1] * len(rows)
            tdca_margins = [0.0] * len(rows)
        threshold = float(self.score_threshold_spin.value()) if hasattr(self, "score_threshold_spin") else 60.0

        scored = []
        for idx, row in enumerate(rows):
            label_idx = int(row["label_idx"])
            sample = row["sample"]
            fb_pred, fb_margin = self._score_model_once(fbcca, sample)
            cca_pred, cca_margin = self._score_model_once(cca, sample)
            tdca_pred = int(tdca_preds[idx])
            tdca_margin = float(tdca_margins[idx])

            fb_hit = fb_pred == label_idx
            cca_hit = cca_pred == label_idx
            tdca_hit = tdca_pred == label_idx
            tdca_valid = 0 <= tdca_pred < len(self.commands)
            accuracy_score, confidence_score, total_score = self._combine_detection_score(
                fb_hit, fb_margin, cca_hit, cca_margin, tdca_hit, tdca_margin, tdca_valid
            )
            print(
                f"[SCORE_PROGRESS] {progress_label} {idx + 1}/{total_rows} "
                f"file={os.path.basename(row['fp'])} score={total_score:.1f} "
                f"acc={accuracy_score:.1f} conf={confidence_score:.1f} "
                f"fb={'hit' if fb_hit else 'miss'} cca={'hit' if cca_hit else 'miss'}",
                flush=True,
            )
            scored.append({
                "fp": row["fp"],
                "label_idx": label_idx,
                "points": row["points"],
                "fb_pred": int(fb_pred),
                "fb_margin": float(fb_margin),
                "fb_hit": bool(fb_hit),
                "cca_pred": int(cca_pred),
                "cca_margin": float(cca_margin),
                "cca_hit": bool(cca_hit),
                "tdca_pred": int(tdca_pred),
                "tdca_margin": float(tdca_margin),
                "tdca_hit": bool(tdca_hit),
                "tdca_valid": bool(tdca_valid),
                "accuracy_score": accuracy_score,
                "confidence_score": confidence_score,
                "total_score": total_score,
                "keep": bool(fb_hit and total_score >= threshold),
            })
        return scored

    def score_and_check_train_samples(self):
        self._stop_quality_calc()
        t0 = time.perf_counter()
        files = []
        for item in self._iter_train_leaf_items():
            fp = item.data(0, Qt.UserRole)
            if isinstance(fp, str) and len(fp) > 0 and self._is_item_in_current_stim_bucket(item) and item.checkState(0) == Qt.Checked:
                files.append(fp)
        print(f"[SCORE_PROGRESS] train_check requested files={len(files)}", flush=True)
        if hasattr(self, "weight_status"):
            self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u6b63\u5728\u8bc4\u5206 {len(files)} \u6761...")
        QApplication.processEvents()
        scored = self._score_current_bucket_rows(include_tdca=False, files=files, progress_label="train_check")
        if len(scored) == 0:
            self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u5f53\u524d {self.training_window_sec:.2f}s \u65e0\u53ef\u8bc4\u5206\u6837\u672c")
            print("[SCORE_PROGRESS] train_check no scorable samples", flush=True)
            return

        score_map = {os.path.normcase(os.path.abspath(row["fp"])): row for row in scored}
        selected = 0
        matched = 0
        for item in self._iter_train_leaf_items():
            fp = item.data(0, Qt.UserRole)
            if not isinstance(fp, str) or len(fp) == 0:
                continue
            key = os.path.normcase(os.path.abspath(fp))
            if key not in score_map:
                if self._is_item_in_current_stim_bucket(item):
                    item.setCheckState(0, Qt.Unchecked)
                continue

            row = score_map[key]
            matched += 1
            score = float(row["total_score"])
            self._apply_quality_to_item(item, score)
            item.setCheckState(0, Qt.Checked if row["keep"] else Qt.Unchecked)
            if row["keep"]:
                selected += 1

            def pred_name(pred):
                return self.commands[pred] if 0 <= int(pred) < len(self.commands) else "-"

            item.setToolTip(
                3,
                (
                    f"FBCCA: {pred_name(row['fb_pred'])} / {row['fb_margin']:.1f} / {'hit' if row['fb_hit'] else 'miss'}\n"
                    f"CCA: {pred_name(row['cca_pred'])} / {row['cca_margin']:.1f} / {'hit' if row['cca_hit'] else 'miss'}\n"
                    f"TDCA: {pred_name(row['tdca_pred'])} / {row['tdca_margin']:.1f} / {'hit' if row['tdca_hit'] else 'miss'}\n"
                    f"鍛戒腑鍒?{row['accuracy_score']:.1f} | 缃俊鍒?{row['confidence_score']:.1f} | 鎬诲垎: {row['total_score']:.1f}"
                ),
            )

        self.weight_status.setText(
            f"Current weight: selected by score {selected}/{matched}, training still uses original data paths"
        )
        if hasattr(self, "score_status"):
            self.score_status.setText(f"Score: synced to training list {selected}/{matched}")
        print(
            f"[SCORE_PROGRESS] train_check done selected={selected}/{matched} "
            f"elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def refresh_score_dataset_view(self):
        if not hasattr(self, "score_tree"):
            return
        t0 = time.perf_counter()
        self.score_tree.setSortingEnabled(False)
        self.score_tree.clear()
        files = self._current_bucket_training_files()
        rows = self._load_score_samples(files)
        print(f"[SCORE_PROGRESS] score_tab requested files={len(files)} rows={len(rows)}", flush=True)
        if len(rows) == 0:
            self.score_status.setText(f"Score: no scorable samples for {self.training_window_sec:.2f}s")
            self.score_tree.setSortingEnabled(True)
            print("[SCORE_PROGRESS] score_tab no scorable samples", flush=True)
            return

        model_times = float(self.analysis_window_sec + self.analysis_delay_sec)
        fbcca = FBCCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        cca = CCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        print("[SCORE_PROGRESS] score_tab tdca_leave_one_out start", flush=True)
        tdca_preds, tdca_margins = self._tdca_leave_one_out_scores(rows, model_times)
        print("[SCORE_PROGRESS] score_tab tdca_leave_one_out done", flush=True)

        keep_count = 0
        for idx, row in enumerate(rows):
            sample = row["sample"]
            label_idx = int(row["label_idx"])
            fb_pred, fb_margin = self._score_model_once(fbcca, sample)
            cca_pred, cca_margin = self._score_model_once(cca, sample)
            tdca_pred = int(tdca_preds[idx])
            tdca_margin = float(tdca_margins[idx])

            fb_hit = fb_pred == label_idx
            cca_hit = cca_pred == label_idx
            tdca_hit = tdca_pred == label_idx
            tdca_valid = 0 <= tdca_pred < len(self.commands)
            accuracy_score, confidence_score, total_score = self._combine_detection_score(
                fb_hit, fb_margin, cca_hit, cca_margin, tdca_hit, tdca_margin, tdca_valid
            )

            suggestion = "淇濈暀" if fb_hit and total_score >= float(self.score_threshold_spin.value()) else "澶嶆煡"
            if suggestion == "淇濈暀":
                keep_count += 1

            print(
                f"[SCORE_PROGRESS] score_tab {idx + 1}/{len(rows)} "
                f"file={os.path.basename(row['fp'])} score={total_score:.1f} "
                f"acc={accuracy_score:.1f} conf={confidence_score:.1f} "
                f"fb={'hit' if fb_hit else 'miss'} cca={'hit' if cca_hit else 'miss'} tdca={'hit' if tdca_hit else 'miss'}",
                flush=True,
            )

            def name_of(pred):
                return self.commands[pred] if 0 <= int(pred) < len(self.commands) else "-"

            item = QTreeWidgetItem([
                os.path.basename(row["fp"]),
                self.commands[label_idx],
                str(row["points"]),
                name_of(fb_pred),
                f"{fb_margin:.1f}",
                name_of(cca_pred),
                f"{cca_margin:.1f}",
                name_of(tdca_pred),
                f"{tdca_margin:.1f}" if tdca_valid else "-",
                f"{total_score:.1f}",
                suggestion,
            ])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if suggestion == "淇濈暀" else Qt.Unchecked)
            item.setData(0, Qt.UserRole, row["fp"])
            item.setData(9, Qt.UserRole, total_score)
            if suggestion == "淇濈暀":
                item.setForeground(10, QBrush(QColor(110, 231, 183)))
            else:
                item.setForeground(10, QBrush(QColor(252, 211, 77)))
            self.score_tree.addTopLevelItem(item)

        for col in range(self.score_tree.columnCount()):
            self.score_tree.resizeColumnToContents(col)
        self.score_tree.setSortingEnabled(True)
        self.score_status.setText(
            f"Score: scored {len(rows)} | current window {self.training_window_sec:.2f}s | suggested keep {keep_count}"
        )

    def _iter_score_items(self):
        if not hasattr(self, "score_tree"):
            return []
        return [self.score_tree.topLevelItem(i) for i in range(self.score_tree.topLevelItemCount())]

    def auto_check_scored_samples(self):
        threshold = float(self.score_threshold_spin.value())
        selected = 0
        total = 0
        for item in self._iter_score_items():
            total += 1
            score = item.data(9, Qt.UserRole)
            try:
                score = float(score)
            except Exception:
                score = 0.0
            keep = item.text(10) == "淇濈暀" and score >= threshold
            item.setCheckState(0, Qt.Checked if keep else Qt.Unchecked)
            if keep:
                selected += 1
        self.score_status.setText(f"Score: selected {selected}/{total}")

    def sync_score_checked_to_train_samples(self):
        checked_paths = set()
        for item in self._iter_score_items():
            fp = item.data(0, Qt.UserRole)
            if item.checkState(0) == Qt.Checked and isinstance(fp, str):
                checked_paths.add(os.path.normcase(os.path.abspath(fp)))
        if len(checked_paths) == 0:
            self.score_status.setText("\u6570\u636e\u8bc4\u5206: \u6ca1\u6709\u53ef\u540c\u6b65\u7684\u52fe\u9009\u6837\u672c")
            return

        selected = 0
        matched = 0
        for item in self._iter_train_leaf_items():
            fp = item.data(0, Qt.UserRole)
            if not isinstance(fp, str) or len(fp) == 0:
                continue
            key = os.path.normcase(os.path.abspath(fp))
            if key in checked_paths:
                item.setCheckState(0, Qt.Checked)
                matched += 1
                selected += 1
            elif self._is_item_in_current_stim_bucket(item):
                item.setCheckState(0, Qt.Unchecked)
        self.data_tabs.setCurrentIndex(0)
        self.weight_status.setText(f"Current weight: synced selected training samples {selected}/{matched}")
        self.score_status.setText(f"Score: synced {selected}/{matched} to training list")

    def save_checked_scored_samples(self):
        checked = []
        for item in self._iter_score_items():
            if item.checkState(0) == Qt.Checked:
                fp = item.data(0, Qt.UserRole)
                if isinstance(fp, str) and os.path.exists(fp):
                    checked.append((fp, item))
        if len(checked) == 0:
            self.score_status.setText("Score: no selected samples")
            return

        day_dir = datetime.now().strftime("%Y-%m-%d")
        bucket = f"{float(self.training_window_sec):.2f}s"
        out_dir = os.path.join(self._subject_root(), "curated", day_dir, bucket)
        os.makedirs(out_dir, exist_ok=True)
        copied = 0
        for fp, item in checked:
            score_text = item.text(9).replace(".", "p")
            base = os.path.basename(fp)
            dst = os.path.join(out_dir, f"score_{score_text}_{base}")
            try:
                shutil.copy2(fp, dst)
                copied += 1
            except Exception:
                continue
        self.score_status.setText(f"\u6570\u636e\u8bc4\u5206: \u5df2\u4fdd\u5b58 {copied}/{len(checked)} \u6761\u5230 {out_dir}")

    def _calc_sample_quality(self, data, label_idx):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            return None
        if data.shape[-1] < int(self.training_sample_points * 0.95):
            return None

        req_pts = min(int(self.training_sample_points), data.shape[-1])
        sample = data[:, -req_pts:]
        try:
            scores = self.fbcca.score_vector(sample)
        except Exception:
            return None
        if not isinstance(scores, np.ndarray) or scores.size == 0:
            return None

        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])
        second_score = float(np.partition(scores, -2)[-2]) if scores.size >= 2 else 0.0
        top_margin = max(0.0, top_score - second_score) / (abs(top_score) + abs(second_score) + 1e-8)

        if 0 <= label_idx < scores.size:
            true_score = float(scores[label_idx])
            other_scores = np.delete(scores, label_idx)
            best_other = float(np.max(other_scores)) if other_scores.size > 0 else 0.0
            label_margin = max(0.0, (true_score - best_other) / (abs(true_score) + abs(best_other) + 1e-8))
            hit_bonus = 1.0 if top_idx == label_idx else 0.0
            raw_quality = 0.7 * label_margin + 0.2 * top_margin + 0.1 * hit_bonus
        else:
            raw_quality = 0.6 * top_margin

        return float(np.clip(raw_quality, 0.0, 1.0))

    def _percentile_rank(self, values, value):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        less = float(np.sum(arr < value))
        equal = float(np.sum(arr == value))
        return float((less + 0.5 * equal) / arr.size)

    def _apply_quality_to_item(self, item, quality_val):
        if quality_val is None:
            item.setText(3, "-")
            item.setData(3, Qt.UserRole, None)
            item.setForeground(3, QBrush(QColor(203, 213, 225)))
            return
        item.setText(3, f"{quality_val:.1f}")
        item.setData(3, Qt.UserRole, quality_val)
        if quality_val >= 70:
            item.setForeground(3, QBrush(QColor(110, 231, 183)))
        elif quality_val < 40:
            item.setForeground(3, QBrush(QColor(252, 165, 165)))
        else:
            item.setForeground(3, QBrush(QColor(226, 232, 240)))

    def _instant_quality_score(self, input_ok, drop_ratio, effective_fs, measured_conf, label_match):
        return self._instant_quality_score_with_sr(
            sample_rate=float(self.sample_rate_hz),
            input_ok=input_ok,
            drop_ratio=drop_ratio,
            effective_fs=effective_fs,
            measured_conf=measured_conf,
            label_match=label_match,
        )

    def _instant_quality_score_with_sr(self, sample_rate, input_ok, drop_ratio, effective_fs, measured_conf, label_match):
        fs_term = float(np.clip(float(effective_fs) / max(float(sample_rate), 1e-6), 0.0, 1.0))
        drop_term = float(np.clip(1.0 - float(drop_ratio), 0.0, 1.0))
        conf_term = float(np.clip(float(measured_conf) / 0.20, 0.0, 1.0))
        ok_term = 1.0 if int(input_ok) == 1 else 0.0
        match_term = 1.0 if bool(label_match) else 0.0

        score01 = 0.45 * conf_term + 0.20 * fs_term + 0.20 * drop_term + 0.15 * ok_term
        if float(drop_ratio) > 0.05:
            score01 *= 0.85
        if float(effective_fs) < (0.95 * float(sample_rate)):
            score01 *= 0.85
        if match_term < 0.5:
            score01 *= 0.70
        return float(np.clip(100.0 * score01, 0.0, 100.0))

    def _iter_train_leaf_items(self):
        leaf_items = []

        def walk(node):
            if node.childCount() == 0:
                leaf_items.append(node)
                return
            for idx in range(node.childCount()):
                walk(node.child(idx))

        for i in range(self.train_data_tree.topLevelItemCount()):
            walk(self.train_data_tree.topLevelItem(i))
        return leaf_items

    def _item_stim_bucket_sec(self, item):
        cur = item
        while cur is not None:
            txt = str(cur.text(0)).strip()
            if txt.endswith("s"):
                try:
                    return float(txt[:-1])
                except Exception:
                    pass
            cur = cur.parent()
        return None

    def _is_item_in_current_stim_bucket(self, item):
        target = float(self.training_window_sec)
        bucket = self._item_stim_bucket_sec(item)
        if isinstance(bucket, (int, float)):
            target_str = f"{target:.1f}"
            bucket_str = f"{bucket:.1f}"
            return float(target_str) == float(bucket_str)

        fp = item.data(0, Qt.UserRole)
        if not isinstance(fp, str) or len(fp) == 0:
            return False
        try:
            m = loadmat(fp, variable_names=['trial_stim_sec', 'stim_freqs_hz'])
            stim = float(np.array(m.get('trial_stim_sec', [[-1.0]])).reshape(-1)[0])
            return abs(stim - target) <= 1e-6 and self._mat_freqs_compatible(m)
        except Exception:
            return False

    def _stop_quality_calc(self):
        timer = getattr(self, "quality_calc_timer", None)
        if timer is not None and timer.isActive():
            self.quality_calc_timer.stop()
        self.quality_calc_pending = []
        self.quality_calc_rows = []

    def _start_quality_calc(self):
        if not getattr(self, "enable_quality_tools", False):
            self.quality_calc_pending = []
            self.quality_calc_rows = []
            return
        if getattr(self, "training_collecting", False):
            return
        if hasattr(self, "quality_calc_timer") and self.quality_calc_timer.isActive():
            self.quality_calc_timer.stop()
        if not hasattr(self, "fbcca"):
            return
        if not hasattr(self, "quality_calc_timer"):
            return
        if len(getattr(self, "quality_calc_pending", [])) == 0:
            if len(getattr(self, "quality_calc_rows", [])) > 0:
                self._normalize_quality_by_label(self.quality_calc_rows)
            return
        for row in self.quality_calc_pending:
            item = row.get("item")
            if item is not None and item.treeWidget() is not None:
                self._apply_quality_to_item(item, None)
        self.quality_calc_pending = []
        if len(getattr(self, "quality_calc_rows", [])) > 0:
            self._normalize_quality_by_label(self.quality_calc_rows)

    def _process_quality_calc_batch(self):
        if not getattr(self, "enable_quality_tools", False):
            self._stop_quality_calc()
            return
        if self.training_collecting or self.start_flick:
            return
        if len(self.quality_calc_pending) == 0:
            self.quality_calc_timer.stop()
            self._normalize_quality_by_label(self.quality_calc_rows)
            return

        batch = self.quality_calc_pending[:self.quality_calc_batch_size]
        self.quality_calc_pending = self.quality_calc_pending[self.quality_calc_batch_size:]
        for row in batch:
            item = row.get("item")
            if item is None or item.treeWidget() is None:
                continue
            fp = row.get("fp", "")
            label_idx = int(row.get("label_idx", -1))
            raw_quality = None
            try:
                m = loadmat(
                    fp,
                    variable_names=[
                        'instant_quality_score', 'sample_rate_hz', 'effective_sample_rate_hz',
                        'drop_ratio', 'input_quality_ok', 'measured_confidence', 'label_match'
                    ],
                )
                if 'instant_quality_score' in m:
                    q = float(np.array(m.get('instant_quality_score')).reshape(-1)[0])
                    raw_quality = float(np.clip(q / 100.0, 0.0, 1.0))
                else:
                    file_sr = self._extract_label_idx(m.get('sample_rate_hz', -1))
                    if file_sr <= 0:
                        file_sr = int(self.sample_rate_hz)
                    effective_fs = float(np.array(m.get('effective_sample_rate_hz', [[file_sr]])).reshape(-1)[0])
                    drop_ratio = float(np.array(m.get('drop_ratio', [[0.0]])).reshape(-1)[0])
                    input_ok = int(np.array(m.get('input_quality_ok', [[0]])).reshape(-1)[0])
                    measured_conf = float(np.array(m.get('measured_confidence', [[0.0]])).reshape(-1)[0])
                    label_match = int(np.array(m.get('label_match', [[1]])).reshape(-1)[0])

                    score = self._instant_quality_score_with_sr(
                        sample_rate=float(file_sr),
                        input_ok=input_ok,
                        drop_ratio=drop_ratio,
                        effective_fs=effective_fs,
                        measured_conf=measured_conf,
                        label_match=label_match,
                    )
                    raw_quality = float(np.clip(score / 100.0, 0.0, 1.0))
            except Exception:
                raw_quality = None
            row["raw_quality"] = raw_quality
            self.quality_calc_rows.append(row)

    def _normalize_quality_by_label(self, sample_rows):
        label_raw_map = {}
        for row in sample_rows:
            label_idx = row.get("label_idx", -1)
            raw_quality = row.get("raw_quality", None)
            if isinstance(raw_quality, (int, float)) and label_idx >= 0:
                label_raw_map.setdefault(label_idx, []).append(float(raw_quality))

        for row in sample_rows:
            item = row["item"]
            if item is None or item.treeWidget() is None:
                continue
            label_idx = row.get("label_idx", -1)
            raw_quality = row.get("raw_quality", None)

            if not isinstance(raw_quality, (int, float)):
                self._apply_quality_to_item(item, None)
                continue

            per_label_values = label_raw_map.get(label_idx, [])
            if len(per_label_values) >= 3:
                rank01 = self._percentile_rank(per_label_values, float(raw_quality))
                quality_val = 100.0 * rank01
            else:
                quality_val = 100.0 * float(raw_quality)

            self._apply_quality_to_item(item, float(np.clip(quality_val, 0.0, 100.0)))

    def refresh_train_dataset_view(self):
        if getattr(self, "_refresh_busy", False):
            return
        self._refresh_busy = True
        try:
            self._stop_quality_calc()
            self.train_data_tree.clear()
            self.dataset_file_map = {}
            self.quality_calc_pending = []
            self.quality_calc_rows = []

            root = self._train_root()
            if not os.path.exists(root):
                return

            date_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            date_dirs.sort(reverse=True)
            for date_dir in date_dirs:
                date_path = os.path.join(root, date_dir)
                top = QTreeWidgetItem(self.train_data_tree, [date_dir, "", "", ""])
                top.setFlags(top.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                top.setCheckState(0, Qt.Unchecked)

                mat_files = sorted(glob(os.path.join(date_path, "**", "*.mat"), recursive=True))
                stim_group_nodes = {}
                for fp in mat_files:
                    if "bad_samples" in fp.replace("\\", "/"):
                        continue
                    label_idx = -1
                    try:
                        m = loadmat(
                            fp,
                            variable_names=[
                                'label_text', 'label_idx', 'trial_stim_sec', 'instant_quality_score',
                                'quality_score', 'analysis_samples', 'actual_samples', 'expected_samples'
                            ],
                        )
                        label_text = self._extract_label_text(m.get('label_text', ''))
                        pts_val = self._extract_label_idx(m.get('analysis_samples', -1))
                        if pts_val <= 0:
                            pts_val = self._extract_label_idx(m.get('actual_samples', -1))
                        if pts_val <= 0:
                            pts_val = self._extract_label_idx(m.get('expected_samples', -1))
                        pts = str(pts_val) if pts_val > 0 else "?"
                        label_idx = self._extract_label_idx(m.get('label_idx', -1))
                        if 0 <= label_idx < len(self.commands):
                            label_text = self.commands[label_idx]
                        stim_val = float(np.array(m.get('trial_stim_sec', [[-1.0]])).reshape(-1)[0])
                        if stim_val > 0:
                            stim_key = f"{stim_val:.1f}s"
                        else:
                            stim_key = "鏈煡鏃堕暱"
                        saved_q = None
                        if 'instant_quality_score' in m:
                            saved_q = float(np.array(m.get('instant_quality_score')).reshape(-1)[0])
                        elif 'quality_score' in m:
                            saved_q = float(np.array(m.get('quality_score')).reshape(-1)[0])
                    except Exception:
                        label_text = "璇诲彇澶辫触"
                        pts = "?"
                        stim_key = "鏈煡鏃堕暱"
                        saved_q = None

                    if stim_key not in stim_group_nodes:
                        gnode = QTreeWidgetItem(top, [stim_key, "", "", ""])
                        gnode.setFlags(gnode.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                        gnode.setCheckState(0, Qt.Unchecked)
                        stim_group_nodes[stim_key] = gnode
                    group_node = stim_group_nodes[stim_key]

                    name = os.path.basename(fp)
                    child = QTreeWidgetItem(group_node, [name, label_text, pts, "-"])
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Unchecked)
                    child.setData(0, Qt.UserRole, fp)
                    if isinstance(saved_q, (int, float)):
                        self.quality_calc_rows.append({
                            "item": child,
                            "label_idx": label_idx,
                            "raw_quality": float(np.clip(saved_q / 100.0, 0.0, 1.0)),
                        })
                    else:
                        self.quality_calc_pending.append({"item": child, "label_idx": label_idx, "raw_quality": None, "fp": fp})

                top.setExpanded(True)
                for _, gnode in sorted(stim_group_nodes.items(), key=lambda kv: kv[0]):
                    gnode.setExpanded(True)

            legacy_files = sorted(glob(os.path.join(root, "*.mat")))
            if len(legacy_files) > 0:
                top = QTreeWidgetItem(self.train_data_tree, ["legacy", "", "", ""])
                top.setFlags(top.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                top.setCheckState(0, Qt.Unchecked)
                for fp in legacy_files:
                    label_idx = -1
                    try:
                        m = loadmat(
                            fp,
                            variable_names=[
                                'label_text', 'label_idx', 'instant_quality_score', 'quality_score',
                                'analysis_samples', 'actual_samples', 'expected_samples'
                            ],
                        )
                        label_text = self._extract_label_text(m.get('label_text', ''))
                        pts_val = self._extract_label_idx(m.get('analysis_samples', -1))
                        if pts_val <= 0:
                            pts_val = self._extract_label_idx(m.get('actual_samples', -1))
                        if pts_val <= 0:
                            pts_val = self._extract_label_idx(m.get('expected_samples', -1))
                        pts = str(pts_val) if pts_val > 0 else "?"
                        label_idx = self._extract_label_idx(m.get('label_idx', -1))
                        if 0 <= label_idx < len(self.commands):
                            label_text = self.commands[label_idx]
                        saved_q = None
                        if 'instant_quality_score' in m:
                            saved_q = float(np.array(m.get('instant_quality_score')).reshape(-1)[0])
                        elif 'quality_score' in m:
                            saved_q = float(np.array(m.get('quality_score')).reshape(-1)[0])
                    except Exception:
                        label_text = "璇诲彇澶辫触"
                        pts = "?"
                        saved_q = None
                    child = QTreeWidgetItem(top, [os.path.basename(fp), label_text, pts, "-"])
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Unchecked)
                    child.setData(0, Qt.UserRole, fp)
                    if isinstance(saved_q, (int, float)):
                        self.quality_calc_rows.append({
                            "item": child,
                            "label_idx": label_idx,
                            "raw_quality": float(np.clip(saved_q / 100.0, 0.0, 1.0)),
                        })
                    else:
                        self.quality_calc_pending.append({"item": child, "label_idx": label_idx, "raw_quality": None, "fp": fp})
                top.setExpanded(True)

                self._start_quality_calc()
        finally:
            self._refresh_busy = False

    def auto_check_high_quality_samples(self, _checked=False):
        selected = 0
        total = 0
        for child in self._iter_train_leaf_items():
            fp = child.data(0, Qt.UserRole)
            if not isinstance(fp, str) or len(fp) == 0:
                continue
            if not self._is_item_in_current_stim_bucket(child):
                child.setCheckState(0, Qt.Unchecked)
                continue
            total += 1
            child.setCheckState(0, Qt.Checked)
            selected += 1
        self.weight_status.setText(
            f"Current weight: auto-selected {self.training_window_sec:.2f}s samples {selected}/{total}"
        )

    def _iter_checked_train_files(self):
        checked = []
        for child in self._iter_train_leaf_items():
            if child.checkState(0) == Qt.Checked:
                if not self._is_item_in_current_stim_bucket(child):
                    continue
                fp = child.data(0, Qt.UserRole)
                if isinstance(fp, str) and len(fp) > 0:
                    checked.append(fp)
        return checked

    def train_weight_from_checked_files(self):
        checked_files = self._iter_checked_train_files()
        if len(checked_files) == 0:
            self.weight_status.setText("Current weight: no selected training samples")
            return

        result = self._training_framework().train_weights(checked_files)
        if not result.ok:
            self.weight_status.setText("褰撳墠鏉冮噸: " + result.message)
            if hasattr(self, "decision_info") and isinstance(result.point_hist, dict):
                top_pts = sorted(result.point_hist.items(), key=lambda x: x[1], reverse=True)
                if top_pts:
                    self.decision_info.setText(
                        f"Recognition: training data points mismatch, required={self._tdca_required_points() if self.model_name == 'TDCA' else self.training_sample_points}, top={top_pts[0][0]}"
                    )
            return

        self.class_score_scale = np.asarray(result.class_score_scale, dtype=float)
        self.selected_weight_file = result.save_file
        try:
            if result.weights is not None and hasattr(self.fbcca, "set_frequency_weights"):
                self.fbcca.set_frequency_weights(np.asarray(result.weights, dtype=float))
                self.fbcca_sliding = None
        except Exception:
            pass
        self.weight_status.setText("褰撳墠鏉冮噸: " + result.message)
        self.refresh_weight_file_list(select_file=result.save_file)

    def evaluate_checked_data(self):
        checked_files = self._iter_checked_train_files()
        if len(checked_files) == 0:
            QMessageBox.warning(self, "Test warning", "No selected training data. Select data or train weights first.")
            return
        
        if hasattr(self, "weight_status"):
            self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u6b63\u5728\u8bc4\u6d4b {len(checked_files)} \u6761\u52fe\u9009\u6570\u636e...")
        QApplication.processEvents()
        
        rows = self._load_score_samples(checked_files)
        if len(rows) == 0:
            QMessageBox.warning(self, "Test warning", "Current model has no usable weight. Please train or load weights first.")
            if hasattr(self, "weight_status"):
                self.weight_status.setText("\u5f53\u524d\u6743\u91cd: \u6d4b\u8bc4\u5931\u8d25")
            return
            
        model_times = float(self.analysis_window_sec + self.analysis_delay_sec)
        # Import manually locally to prevent missing sklearn issues
        try:
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            has_sklearn = True
        except ImportError:
            has_sklearn = False

        fbcca = FBCCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        cca = CCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        
        print("[EVAL_PROGRESS] evaluate_checked_data start LOO", flush=True)
        tdca_preds, _ = self._tdca_leave_one_out_scores(rows, model_times)
        
        y_true = []
        fb_preds = []
        cca_preds = []
        
        for idx, row in enumerate(rows):
            sample = row["sample"]
            y_true.append(row["label_idx"])
            fb_p, _ = self._score_model_once(fbcca, sample)
            cca_p, _ = self._score_model_once(cca, sample)
            fb_preds.append(int(fb_p))
            cca_preds.append(int(cca_p))
            
        y_true = np.array(y_true)
        fb_preds = np.array(fb_preds)
        cca_preds = np.array(cca_preds)
        tdca_preds = np.array(tdca_preds)

        def get_metrics(y_t, y_p):
            valid = y_p >= 0
            if not np.any(valid):
                return 0.0, 0.0, 0.0, False
            if has_sklearn:
                acc = accuracy_score(y_t[valid], y_p[valid]) * 100
                bacc = balanced_accuracy_score(y_t[valid], y_p[valid]) * 100
                f1 = f1_score(y_t[valid], y_p[valid], average="macro") * 100
            else:
                acc = np.mean(y_t[valid] == y_p[valid]) * 100
                bacc = 0.0
                f1 = 0.0
            return acc, bacc, f1, True

        fb_acc, fb_bacc, fb_f1, fb_ok = get_metrics(y_true, fb_preds)
        cca_acc, cca_bacc, cca_f1, cca_ok = get_metrics(y_true, cca_preds)
        tdca_acc, tdca_bacc, tdca_f1, tdca_ok = get_metrics(y_true, tdca_preds)
        
        dialog = MessageBoxBase(self)
        dialog.titleLabel = SubtitleLabel(f"鍩轰簬 {len(rows)} 鏉″嬀閫夋暟鎹殑娴嬭瘎缁撴灉", dialog)
        
        table = TableWidget(dialog)
        table.setColumnCount(4)
        table.setRowCount(3)
        table.setHorizontalHeaderLabels(["妯″瀷", "ACC (%)", "bACC (%)", "Macro F1 (%)"])
        
        models = ["FBCCA", "CCA", "TDCA (鐣欎竴娉?"]
        metrics = [(fb_acc, fb_bacc, fb_f1), (cca_acc, cca_bacc, cca_f1), (tdca_acc, tdca_bacc, tdca_f1)]
        status_ok = [fb_ok, cca_ok, tdca_ok]
        
        def create_item(text):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            return item

        for row, (model, mets, ok) in enumerate(zip(models, metrics, status_ok)):
            table.setItem(row, 0, create_item(model))
            if row == 2 and not ok:
                item = create_item("Weight file / Auto loaded weight")
                table.setItem(row, 1, item)
                table.setSpan(row, 1, 1, 3)
            else:
                table.setItem(row, 1, create_item(f"{mets[0]:.2f}"))
                table.setItem(row, 2, create_item(f"{mets[1]:.2f}"))
                table.setItem(row, 3, create_item(f"{mets[2]:.2f}"))
                
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setMinimumHeight(120)
        table.setStyleSheet("""
            QTableWidget { border: 1px solid #35507A; border-radius: 8px; background-color: white; color: black; }
            QTableWidget::item { color: black; }
            QTableWidget::item:selected { color: white; background-color: #0078D7; }
            QHeaderView::section { color: black; background-color: #F0F4F8; font-weight: bold; }
        """)

        dialog.viewLayout.addWidget(dialog.titleLabel)
        dialog.viewLayout.addSpacing(10)
        dialog.viewLayout.addWidget(table)
        if not has_sklearn:
            no_sk_label = CaptionLabel("(\u63d0\u793a: \u672a\u5b89\u88c5 scikit-learn\uff0c\u4ec5\u663e\u793a\u57fa\u672c ACC)")
            no_sk_label.setStyleSheet("color: #9FB0C9;")
            dialog.viewLayout.addWidget(no_sk_label)
        
        dialog.widget.setMinimumWidth(550)
        dialog.yesButton.setText("\u786e\u5b9a")
        dialog.cancelButton.hide()
        
        if hasattr(self, "weight_status"):
            self.weight_status.setText("\u5f53\u524d\u6743\u91cd: \u6d4b\u8bc4\u5b8c\u6210")
            
        dialog.exec_()


    def refresh_weight_file_list(self, select_file=None):
        self.weight_combo.clear()
        self.weight_combo.addItem("\u9ed8\u8ba4(\u65e0)", "")

        wroot = self._weights_root()
        if not os.path.exists(wroot):
            self.weight_combo.setCurrentIndex(0)
            return

        files = list_weight_files(wroot)
        for fp in files:
            self.weight_combo.addItem(os.path.basename(fp), fp)

        if select_file is not None:
            for i in range(self.weight_combo.count()):
                if self.weight_combo.itemData(i) == select_file:
                    self.weight_combo.setCurrentIndex(i)
                    return
        self.weight_combo.setCurrentIndex(0)

    def load_selected_weight_file(self):
        fp = self.weight_combo.currentData()
        if not fp:
            self.fbcca.reset_frequency_weights()
            self.fbcca_sliding = None
            if self.model_name == "TDCA" and hasattr(self.fbcca, "clear_fit"):
                self.fbcca.clear_fit()
            self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
            self.selected_weight_file = ""
            self.weight_status.setText("\u5f53\u524d\u6743\u91cd: \u9ed8\u8ba4(\u65e0)")
            return

        try:
            with open(fp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_model = str(cfg.get("model_name", "")).upper()
            cfg_fs = int(cfg.get("sample_rate_hz", self.sample_rate_hz))
            cfg_freqs = cfg.get("stim_freqs_hz", None)
            if not self._freqs_compatible(cfg_freqs, allow_legacy_without_meta=True):
                self.fbcca.reset_frequency_weights()
                self.fbcca_sliding = None
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                self.selected_weight_file = ""
                self.weight_status.setText("\u5f53\u524d\u6743\u91cd: \u523a\u6fc0\u9891\u7387\u4e0d\u5339\u914d\uff0c\u8bf7\u91cd\u65b0\u91c7\u96c6/\u8bad\u7ec3")
                return
            if cfg_model and cfg_model != self.model_name:
                self.fbcca.reset_frequency_weights()
                self.fbcca_sliding = None
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                self.selected_weight_file = ""
                self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u6a21\u578b\u4e0d\u5339\u914d({cfg_model} != {self.model_name})")
                return
            if cfg_fs != self.sample_rate_hz:
                self.fbcca.reset_frequency_weights()
                self.fbcca_sliding = None
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                self.selected_weight_file = ""
                self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u91c7\u6837\u7387\u4e0d\u5339\u914d({cfg_fs}Hz != {self.sample_rate_hz}Hz)")
                return
            weights = np.asarray(cfg.get("weights", []), dtype=float)
            self.fbcca.set_frequency_weights(weights)
            self.fbcca_sliding = None
            scale = np.asarray(cfg.get("class_score_scale", np.ones(self.fbcca.Nf)), dtype=float)
            if scale.shape[0] == self.fbcca.Nf:
                self.class_score_scale = np.clip(scale, 0.5, 2.0)
            else:
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
            if self.model_name == "TDCA":
                train_files = cfg.get("train_files", [])
                ok, msg = self._fit_tdca_from_files(train_files)
                if not ok:
                    self.selected_weight_file = ""
                    self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: TDCA\u52a0\u8f7d\u5931\u8d25({msg})")
                    return
            self.selected_weight_file = fp
            self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u5df2\u52a0\u8f7d {os.path.basename(fp)}")
        except Exception as e:
            self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
            self.weight_status.setText(f"\u5f53\u524d\u6743\u91cd: \u52a0\u8f7d\u5931\u8d25 {str(e)}")

    def _label_to_index(self, label):
        if label in self.commands:
            return self.commands.index(label)
        try:
            idx = int(label)
            if 0 <= idx < len(self.commands):
                return idx
        except Exception:
            pass
        return -1

    def _classify_training_sample(self, sample_data, fallback_label_idx=-1):
        if self.model_name == "TDCA" and getattr(self.fbcca, "is_fitted", False):
            verifier = self.fbcca
        else:
            verifier = self._create_classifier(self.model_times_sec)

        selected_weight_file = getattr(self, "selected_weight_file", "")
        if verifier is not self.fbcca and selected_weight_file and os.path.exists(selected_weight_file):
            try:
                with open(selected_weight_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg_model = str(cfg.get("model_name", "")).upper()
                cfg_fs = int(cfg.get("sample_rate_hz", self.sample_rate_hz))
                if (not cfg_model or cfg_model == self.model_name) and cfg_fs == self.sample_rate_hz:
                    weights = np.asarray(cfg.get("weights", []), dtype=float)
                    if hasattr(verifier, "set_frequency_weights"):
                        verifier.set_frequency_weights(weights)
            except Exception:
                pass

        try:
            measured_idx, _, measured_conf = verifier.classify_with_scores(sample_data)
        except Exception:
            if self.model_name == "TDCA" and 0 <= int(fallback_label_idx) < len(self.commands):
                measured_idx = int(fallback_label_idx)
                measured_conf = 0.0
            else:
                measured_idx = -1
                measured_conf = 0.0
        measured_idx = int(measured_idx)
        measured_text = self.commands[measured_idx] if 0 <= measured_idx < len(self.commands) else "-"
        return measured_idx, measured_text, float(measured_conf)

    def _start_training_collect(self):
        if self.training_collecting:
            return
        self._stop_quality_calc()
        if len(self.training_plan) == 0:
            self.training_plan = self._build_training_plan()
            self.training_collected = 0

        if self.training_collected >= len(self.training_plan):
            self.show_label.setText("Training data ready. Press Enter to start training/test flow")
            self.decision_info.setText("Recognition: training data ready")
            return

        self.training_target_label = self.training_plan[self.training_collected]
        self.training_collecting = True
        self.training_collect_start_time = time.time()
        self._clear_cache_buffer()
        self.start_cache = True
        self.training_trial_meta = {
            "expected_samples": int(round(self.training_window_sec * self.sample_rate_hz)),
            "analysis_samples": int(self.training_sample_points),
            "sample_rate_hz": int(self.sample_rate_hz),
            "trial_target_label": str(self.training_target_label),
        }
        self.training_stim_frame_idx = 0
        self.training_last_render_frame_idx = -1
        self.training_stim_onset_mono = 0.0
        self.training_stim_onset_unix = 0.0
        self.training_pending_success = False
        self._enter_training_phase("cue")
        self.show_label.setText(f"\u8bad\u7ec3\u63d0\u793a: \u51c6\u5907\u6ce8\u89c6 {self.training_target_label}")
        self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u8bad\u7ec3\u63d0\u793a\u9636\u6bb5")
        self.setDefaultColor()
        self.training_timer.start()

    def _training_tick(self):
        if not self.training_collecting:
            self.training_timer.stop()
            return

        if self.training_phase == "cue":
            elapsed = self._phase_elapsed(self.training_phase_start_ts)
            self.progress.setValue(int(min(1.0, elapsed / max(self.trial_cue_sec, 1e-6)) * 100))
            self.show_label.setText(f"\u8bad\u7ec3\u63d0\u793a: \u8bf7\u6ce8\u89c6 {self.training_target_label}")
            if elapsed >= self.trial_cue_sec:
                self._enter_training_phase("stim")
                self.start_cache = True
                self.training_stim_frame_idx = 0
                self.training_last_render_frame_idx = -1
                self.training_last_progress_update_mono = 0.0
                self.training_stim_onset_mono = self._now_mono()
                self.training_stim_onset_unix = time.time()
                self.training_trial_meta["stim_onset_monotonic"] = float(self.training_stim_onset_mono)
                self.training_trial_meta["stim_onset_unix"] = float(self.training_stim_onset_unix)
                self._begin_stim_timing_log("training")
                self._render_stim_by_frame(0)
                self._start_stim_render_timer()
                self.show_label.setText(f"\u8bad\u7ec3\u523a\u6fc0\u4e2d: {self.training_target_label}")
                self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u8bad\u7ec3\u523a\u6fc0\u9636\u6bb5")
            return

        if self.training_phase == "stim":
            elapsed = self._phase_elapsed(self.training_phase_start_ts)
            now_mono = self._now_mono()
            if now_mono - float(self.training_last_progress_update_mono) >= 0.20:
                self.training_last_progress_update_mono = now_mono
                self.progress.setValue(int(min(1.0, elapsed / max(self.training_window_sec, 1e-6)) * 100))
            if elapsed < self.training_window_sec:
                return
            if elapsed >= self.training_window_sec:
                self._stop_stim_render_timer()
                self.setDefaultColor()
                self.training_pending_success = False
                self.training_trial_meta["stim_end_monotonic"] = float(self._now_mono())
                self.training_trial_meta["stim_end_unix"] = float(time.time())
                self._enter_training_phase("rest")
                self.show_label.setText(f"\u8bad\u7ec3\u95f4\u9694: {self.training_target_label}")
                self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u8bad\u7ec3\u95f4\u9694\u9636\u6bb5")
            return

        if self.training_phase == "rest":
            elapsed = self._phase_elapsed(self.training_phase_start_ts)
            self.progress.setValue(int(max(0.0, 100.0 * (1.0 - min(1.0, elapsed / max(self.trial_rest_sec, 1e-6))))))
            full_data = self._materialize_cache_data()
            recv_start = float(self.training_trial_meta.get("recv_start_monotonic", np.nan))
            self.training_pending_success = self._has_aligned_window_ready(
                full_data,
                recv_start,
                float(self.training_stim_onset_mono),
                int(self.training_sample_points),
            )
            if elapsed >= self.trial_rest_sec:
                self._finish_training_collect(success=bool(self.training_pending_success))
            return

    def _finish_training_collect(self, success):
        self.training_timer.stop()
        self._stop_stim_render_timer()

        self.training_trial_meta["trial_end_monotonic"] = float(self._now_mono())
        self.training_trial_meta["trial_end_unix"] = float(time.time())

        self.start_cache = False
        self.setDefaultColor()

        required_samples = int(self.training_trial_meta.get("expected_samples", self.training_sample_points))
        if not success:
            self._print_stim_frequency_check(self.training_target_label, self._stim_timing_summary())
            self._debug_alignment_failure(
                "training_precheck",
                self._materialize_cache_data(),
                float(self.training_trial_meta.get("recv_start_monotonic", np.nan)),
                float(self.training_stim_onset_mono),
                int(self.training_sample_points),
                self.training_trial_meta,
            )
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("\u8bad\u7ec3\u91c7\u96c6\u8d85\u65f6: \u672a\u6536\u5230\u8db3\u591f EEG \u6570\u636e")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u8bad\u7ec3\u91c7\u96c6\u8d85\u65f6")
            return

        self.progress.setValue(100)

        raw_samples = int(self._cache_sample_count())
        full_data = self._materialize_cache_data()
        if not isinstance(full_data, np.ndarray) or full_data.ndim != 2 or full_data.shape[-1] < int(self.training_sample_points * 0.95):
            self._debug_alignment_failure(
                "training_buffer",
                full_data,
                float(self.training_trial_meta.get("recv_start_monotonic", np.nan)),
                float(self.training_stim_onset_mono),
                int(self.training_sample_points),
                self.training_trial_meta,
            )
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("Training collection warning: no weight file will be saved")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u7f13\u5b58\u6570\u636e\u5f02\u5e38")
            self._clear_cache_buffer()
            return
        recv_start = float(self.training_trial_meta.get("recv_start_monotonic", np.nan))
        req_pts = int(self.training_sample_points)
        used_data = self._extract_aligned_window(
            full_data=full_data,
            recv_start_mono=recv_start,
            stim_onset_mono=float(self.training_stim_onset_mono),
            sample_points=req_pts,
        )
        if not isinstance(used_data, np.ndarray) or used_data.ndim != 2 or used_data.shape[-1] < req_pts:
            self._debug_alignment_failure(
                "training",
                full_data,
                recv_start,
                float(self.training_stim_onset_mono),
                req_pts,
                self.training_trial_meta,
            )
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("Training collection warning: failed to save weight")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u6743\u91cd\u4fdd\u5b58\u5931\u8d25")
            self._clear_cache_buffer()
            return

        benchmark_trial_data, benchmark_idx = self._extract_benchmark_trial(
            full_data=full_data,
            recv_start_mono=recv_start,
            stim_onset_mono=float(self.training_stim_onset_mono),
            stim_window_sec=float(self.training_window_sec),
        )
        if not isinstance(benchmark_trial_data, np.ndarray) or benchmark_trial_data.ndim != 2:
            self._debug_alignment_failure(
                "training_benchmark_trial",
                full_data,
                recv_start,
                float(self.training_stim_onset_mono) - BENCHMARK_PRE_STIMULUS_SEC,
                self._benchmark_required_trial_points(self.training_window_sec),
                self.training_trial_meta,
            )
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("Training collection saved benchmark trial data")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: benchmark trial \u5df2\u4fdd\u5b58")
            self._clear_cache_buffer()
            return

        model_data = self._prepare_model_input(used_data)
        if not isinstance(model_data, np.ndarray) or model_data.ndim != 2:
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("\u8bad\u7ec3\u91c7\u96c6\u5931\u8d25: \u6837\u672c\u9884\u5904\u7406\u5f02\u5e38")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u6743\u91cd\u5df2\u52a0\u8f7d")
            self._clear_cache_buffer()
            return
        self._clear_cache_buffer()

        expected_samples = int(self.training_trial_meta.get("expected_samples", self.training_sample_points))
        analysis_samples = int(used_data.shape[-1])
        cache_samples = int(raw_samples)
        # Quality statistics should be based on the aligned stimulus window,
        # not the full cache that also includes cue/rest samples.
        actual_samples = int(analysis_samples)
        stim_onset = float(self.training_trial_meta.get("stim_onset_monotonic", self.training_stim_onset_mono))
        stim_end = float(self.training_trial_meta.get("stim_end_monotonic", self._now_mono()))
        stim_dur = max(1e-6, stim_end - stim_onset)
        effective_fs = float(actual_samples / stim_dur)
        drop_ratio = max(0.0, float(expected_samples - actual_samples) / max(expected_samples, 1))
        recv_start = float(self.training_trial_meta.get("recv_start_monotonic", stim_onset))
        recv_end = float(self.training_trial_meta.get("recv_end_monotonic", stim_end))
        recv_dur = max(0.0, recv_end - recv_start)
        input_ok = int((actual_samples >= int(0.95 * expected_samples)) and (effective_fs >= 0.75 * self.sample_rate_hz))

        day_dir = datetime.now().strftime("%Y-%m-%d")
        stim_bucket = f"{float(self.training_window_sec):.2f}s"
        savePath = os.path.join(self._train_root(), day_dir, stim_bucket)
        os.makedirs(savePath, exist_ok=True)

        label_idx = self._label_to_index(self.training_target_label)
        measured_idx = int(label_idx)
        measured_text = self.training_target_label
        measured_conf = 0.0
        label_match = True
        instant_quality = self._instant_quality_score(
            input_ok=input_ok,
            drop_ratio=drop_ratio,
            effective_fs=effective_fs,
            measured_conf=measured_conf,
            label_match=label_match,
        )

        file_id = self.training_collected + 1
        stamp = datetime.now().strftime("%H%M%S")
        saveFile = os.path.join(savePath, f'{file_id:04d}_{self.training_target_label}_{stamp}.mat')
        align_idx = self._aligned_window_indices(
            recv_start_mono=recv_start,
            stim_onset_mono=float(stim_onset),
            sample_points=int(self.training_sample_points),
        )
        align_start = int(align_idx[0]) if isinstance(align_idx, tuple) else -1
        align_end = int(align_idx[1]) if isinstance(align_idx, tuple) else -1
        benchmark_start = int(benchmark_idx[0]) if isinstance(benchmark_idx, tuple) else -1
        benchmark_end = int(benchmark_idx[1]) if isinstance(benchmark_idx, tuple) else -1
        benchmark_onset_samples = int(round((BENCHMARK_PRE_STIMULUS_SEC + BENCHMARK_VISUAL_DELAY_SEC) * self.sample_rate_hz))
        stim_timing = self._stim_timing_summary()
        self._print_stim_frequency_check(self.training_target_label, stim_timing)

        try:
            savemat(saveFile, {
                'data': benchmark_trial_data,
                'raw_data': benchmark_trial_data,
                'analysis_data': used_data,
                'model_data': model_data,
                'benchmark_trial_data': benchmark_trial_data,
                'benchmark_data_4d': benchmark_trial_data[:, :, np.newaxis, np.newaxis],
                'benchmark_format_version': 1,
                'benchmark_pre_stimulus_sec': float(BENCHMARK_PRE_STIMULUS_SEC),
                'benchmark_visual_delay_sec': float(BENCHMARK_VISUAL_DELAY_SEC),
                'benchmark_num_harmonics': int(BENCHMARK_NUM_HARMONICS),
                'benchmark_onset_samples': int(benchmark_onset_samples),
                'benchmark_trial_start_idx': int(benchmark_start),
                'benchmark_trial_end_idx': int(benchmark_end),
                'stim_render_events': int(stim_timing.get('render_events', 0)),
                'stim_skipped_frames': int(stim_timing.get('skipped_frames', 0)),
                'stim_max_frame_delta': int(stim_timing.get('max_frame_delta', 0)),
                'stim_max_render_interval_ms': float(stim_timing.get('max_render_interval_ms', 0.0)),
                'stim_mean_render_interval_ms': float(stim_timing.get('mean_render_interval_ms', 0.0)),
                'stim_effective_render_hz': float(stim_timing.get('effective_render_hz', 0.0)),
                'stim_estimated_freqs_hz': np.array(stim_timing.get('estimated_freqs_hz', []), dtype=float),
                'sample_rate_hz': int(self.sample_rate_hz),
                'model_name': self.model_name,
                'trial_cue_sec': float(self.trial_cue_sec),
                'trial_stim_sec': float(self.training_window_sec),
                'analysis_window_sec': float(self.analysis_window_sec),
                'analysis_delay_sec': float(self.analysis_delay_sec),
                'trial_rest_sec': float(self.trial_rest_sec),
                'cue_onset_monotonic': float(self.training_trial_meta.get('cue_onset_monotonic', 0.0)),
                'cue_onset_unix': float(self.training_trial_meta.get('cue_onset_unix', 0.0)),
                'stim_onset_monotonic': float(self.training_stim_onset_mono),
                'stim_onset_unix': float(self.training_stim_onset_unix),
                'stim_end_monotonic': float(self.training_trial_meta.get('stim_end_monotonic', 0.0)),
                'stim_end_unix': float(self.training_trial_meta.get('stim_end_unix', 0.0)),
                'rest_onset_monotonic': float(self.training_trial_meta.get('rest_onset_monotonic', 0.0)),
                'rest_onset_unix': float(self.training_trial_meta.get('rest_onset_unix', 0.0)),
                'trial_end_monotonic': float(self.training_trial_meta.get('trial_end_monotonic', 0.0)),
                'trial_end_unix': float(self.training_trial_meta.get('trial_end_unix', 0.0)),
                'expected_samples': int(expected_samples),
                'actual_samples': int(actual_samples),
                'cache_samples': int(cache_samples),
                'analysis_samples': int(analysis_samples),
                'drop_ratio': float(drop_ratio),
                'effective_sample_rate_hz': float(effective_fs),
                'recv_start_monotonic': float(self.training_trial_meta.get('recv_start_monotonic', 0.0)),
                'recv_end_monotonic': float(self.training_trial_meta.get('recv_end_monotonic', 0.0)),
                'recv_duration_sec': float(recv_dur),
                'aligned_start_idx': int(align_start),
                'aligned_end_idx': int(align_end),
                'recv_chunks': int(self.training_trial_meta.get('recv_chunks', 0)),
                'recv_points': int(self.training_trial_meta.get('recv_points', 0)),
                'input_quality_ok': int(input_ok),
                'stim_freqs_hz': np.array(self.sti_lst, dtype=float),
                'stim_frame_periods': np.array(self.stim_period_frames, dtype=int),
                'stim_duty_frames': np.array(self.stim_duty_frames, dtype=int),
                'display_refresh_hz': float(self.stim_refresh_hz),
                'label_text': self.training_target_label,
                'label_idx': label_idx,
                'pred_label_text': self.training_target_label,
                'pred_label_idx': label_idx,
                'measured_label_text': measured_text,
                'measured_label_idx': measured_idx,
                'measured_confidence': measured_conf,
                'label_match': int(label_match),
                'instant_quality_score': float(instant_quality),
                'instant_quality_source': 'collect-time-lite',
                'commands': np.array(self.commands, dtype=object),
            })
        except Exception as e:
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("\u8bad\u7ec3\u6837\u672c\u4fdd\u5b58\u5931\u8d25")
            self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u4fdd\u5b58\u5931\u8d25 {str(e)}")
            return

        self.decision_info.setText(
            f"\u8bc6\u522b\u72b6\u6001: \u7a97\u53e3 ok={bool(input_ok)} | \u6837\u672c {actual_samples}/{expected_samples} | fs={effective_fs:.1f}Hz"
        )

        self.training_collected += 1
        self._update_training_progress()
        if self.training_collected >= len(self.training_plan):
            self.show_label.setText(f"\u8bad\u7ec3\u5b8c\u6210: \u5df2\u4fdd\u5b58 {self.training_collected}/{len(self.training_plan)}")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u8bad\u7ec3\u6570\u636e\u5df2\u5c31\u7eea")
            QTimer.singleShot(100, self.refresh_train_dataset_view)
        else:
            self.show_label.setText(
                f"\u8bad\u7ec3\u5df2\u4fdd\u5b58: {self.training_target_label} ({self.training_collected}/{len(self.training_plan)})"
            )
        self.training_collecting = False
        self.training_phase = "idle"
        self.training_target_label = ""

    def _vote_command(self, idx, confidence):
        if confidence < self.min_confidence:
            self.last_gate_reason = f"\u7f6e\u4fe1\u5ea6\u8fc7\u4f4e({confidence:.3f}<{self.min_confidence:.3f})"
            return None

        self.decision_buffer.append(idx)
        self.current_votes = [self.commands[i] for i in self.decision_buffer]
        self._update_live_panel()
        if len(self.decision_buffer) < self.vote_threshold:
            self.last_gate_reason = f"\u6295\u7968\u4e0d\u8db3({len(self.decision_buffer)}/{self.vote_threshold})"
            return None

        values, counts = np.unique(np.array(self.decision_buffer), return_counts=True)
        winner = int(values[np.argmax(counts)])
        agree = int(np.max(counts))
        if agree < self.vote_threshold:
            self.last_gate_reason = f"\u6295\u7968\u672a\u8fbe\u9608\u503c({agree}/{self.vote_threshold})"
            return None

        if not self._cooldown_ok(winner):
            return None

        self.last_gate_reason = "\u6295\u7968\u901a\u8fc7"
        return winner

    def _send_robot_command_async(self, idx):
        if not bool(getattr(self, "robot_control_enabled", False)):
            return
        th = threading.Thread(target=self._send_robot_command, args=(idx,), daemon=True)
        th.start()

    def _send_robot_command(self, idx):
        if not bool(getattr(self, "robot_control_enabled", False)):
            return
        ip, port = self._get_robot_endpoint()
        try:
            client = RobotClient(ip, port)
            client.connect()
            if client.connected:
                self.robot_net_ok = True
                self._refresh_network_status()
                robot_idx = int(self.robot_command_indices[idx]) if idx < len(self.robot_command_indices) else int(idx)
                client.move(robot_idx)
                print(f"Socket\u6307\u4ee4\u5df2\u53d1\u9001: {self.commands[idx]}")
            else:
                self.robot_control_enabled = False
                self.robot_net_ok = False
                self._refresh_network_status()
                self.decision_info.setText(
                    f"\u8bc6\u522b\u72b6\u6001: \u63a7\u5236\u8fde\u63a5\u5931\u8d25 {ip}:{port}\uff0c\u5df2\u5173\u95ed\u5c0f\u8f66\u53d1\u9001"
                )
                print("\u8fde\u63a5\u5931\u8d25")
            client.close()
        except Exception as e:
            self.robot_control_enabled = False
            self.robot_net_ok = False
            self._refresh_network_status()
            self.decision_info.setText(
                f"\u8bc6\u522b\u72b6\u6001: \u63a7\u5236\u5f02\u5e38 {ip}:{port}\uff0c\u5df2\u5173\u95ed\u5c0f\u8f66\u53d1\u9001"
            )
            print(f"\u63a7\u5236\u5f02\u5e38: {str(e)}")

    def _current_online_truth_label(self):
        if getattr(self, "mode", "online") == "test":
            return self._current_test_truth()
        truth = getattr(self, "online_eval_truth", "not_counted")
        return truth if truth in self.commands else ""

    def start_sti_event(self):
        if not self.start_flick:
            if getattr(self, "online_strategy", "sliding_vote") not in ("async_fbcca", "sliding_vote") and not self._ensure_tdca_ready():
                return
            if getattr(self, "mode", "online") == "test":
                self.test_eval_plan = self._build_training_plan()
                if len(self.test_eval_plan) == 0:
                    self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u6d4b\u8bd5\u6807\u7b7e\u4e3a\u7a7a")
                    return
                self._reset_test_accuracy_stats(reset_plan=False)
            self.start_flick = True
            self.finish = False
            self.continuous_mode = True
            self.start_cache = False
            self._clear_cache_buffer()
            self.online_score_warmup = 0
            self.online_score_ema = np.ones(len(self.commands), dtype=float)
            self.decision_buffer.clear()
            self.sliding_cycle_count = 0
            if getattr(self, "mode", "online") == "test":
                self.show_label.setText(f"\u6d4b\u8bd5\u6a21\u5f0f: \u91c7\u96c6 {self.online_window_sec:.2f}s \u7a97\u53e3\uff0c\u8bf7\u6ce8\u89c6\u76ee\u6807")
            else:
                self._reset_online_accuracy_stats()
                self.show_label.setText(self._online_hint_text())
            self._start_online_window()
        else:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("\u5c31\u7eea")
            self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u5c31\u7eea")

    def _start_online_window(self):
        if self.finish or not self.start_flick:
            return
        if getattr(self, "mode", "online") == "test":
            if self.test_eval_index >= len(getattr(self, "test_eval_plan", [])):
                acc = 100.0 * self.test_eval_correct / max(self.test_eval_total, 1)
                self._reset_collection_state()
                self.continuous_mode = False
                self.show_label.setText(f"\u6d4b\u8bd5\u5b8c\u6210: {acc:.2f}% ({self.test_eval_correct}/{self.test_eval_total})")
                self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u6d4b\u8bd5\u5b8c\u6210")
                return

        if self.fbcca_sliding is None:
            self.fbcca_sliding = self._create_sliding_classifier()

        self._clear_cache_buffer()
        self.start_cache = True
        self.online_trial_meta = {
            "expected_samples": int(round(self.online_window_sec * self.sample_rate_hz)),
            "analysis_samples": int(round(self._online_analysis_sec() * self.sample_rate_hz)),
            "sample_rate_hz": int(self.sample_rate_hz),
            "online_strategy": str(getattr(self, "online_strategy", "sliding_vote")),
            "online_cycle_sec": float(self.online_window_sec),
            "online_analysis_window_sec": float(self._online_analysis_sec()),
        }
        truth = self._current_online_truth_label()
        if truth:
            self.online_trial_meta["trial_target_label"] = str(truth)
        self.online_window_active = True
        self.online_window_start_time = time.time()
        # Start continuous stimulation immediately \u2014 skip cue
        self._enter_online_phase("stim")
        self.online_stim_frame_idx = 0
        self.online_last_render_frame_idx = -1
        self.online_last_progress_update_mono = 0.0
        self.online_stim_onset_mono = self._now_mono()
        self.online_stim_onset_unix = time.time()
        self.online_trial_meta["stim_onset_monotonic"] = float(self.online_stim_onset_mono)
        self.online_trial_meta["stim_onset_unix"] = float(self.online_stim_onset_unix)
        self._begin_stim_timing_log("online")
        self._render_stim_by_frame(0)
        self._start_stim_render_timer()
        self.sliding_cycle_start_mono = float(self.online_stim_onset_mono)
        if getattr(self, "mode", "online") == "test":
            truth = self._current_test_truth()
            self.show_label.setText(f"Test stimulus: target {truth} (trial {self.test_eval_index + 1}/{len(self.test_eval_plan)})")
        else:
            self.show_label.setText(f"Online stimulus: {self._online_strategy_label()} | cycle {self.sliding_cycle_count + 1}")
        self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u5728\u7ebf\u523a\u6fc0\u4e2d")
        if not self.online_timer.isActive():
            self.online_timer.start()

    def _classify_online_fbcca_window(self, window):
        model_data = self._prepare_model_input(window)
        if model_data is None:
            return -1, None, 0.0
        _, raw_scores, _ = self.fbcca_sliding.classify_with_scores(model_data)
        balanced = self._apply_class_score_balance(raw_scores)
        online_scores = self._normalize_online_scores(balanced)
        pred = int(np.argmax(online_scores))
        confidence = self._confidence_from_scores(online_scores)
        return pred, np.asarray(online_scores, dtype=float), float(confidence)

    def _online_window_offsets(self, cycle_samples, window_samples):
        cycle_samples = int(cycle_samples)
        window_samples = int(window_samples)
        step_samples = max(1, int(round(float(self.online_sliding_step_sec) * self.sample_rate_hz)))
        if cycle_samples <= window_samples:
            return [0]
        offsets = list(range(0, cycle_samples - window_samples + 1, step_samples))
        last = cycle_samples - window_samples
        if offsets[-1] != last:
            offsets.append(last)
        return offsets

    def _pick_online_vote_winner(self, results, scores_list):
        valid = [int(x) for x in results if 0 <= int(x) < len(self.commands)]
        valid_scores = [s for s in scores_list if isinstance(s, np.ndarray)]
        if len(valid) == 0:
            return -1, 0.0

        values, counts = np.unique(np.asarray(valid, dtype=int), return_counts=True)
        max_count = int(np.max(counts))
        tied = [int(v) for v, c in zip(values, counts) if int(c) == max_count]
        if len(valid_scores) > 0:
            sum_scores = np.sum(valid_scores, axis=0)
        else:
            sum_scores = np.zeros(len(self.commands), dtype=float)

        if len(tied) == 1:
            winner = tied[0]
        else:
            winner = max(tied, key=lambda idx: float(sum_scores[idx]) if idx < sum_scores.size else 0.0)

        vote_conf = float(max_count / max(len(valid), 1))
        score_conf = self._confidence_from_scores(sum_scores) if sum_scores.size else 0.0
        return int(winner), float(max(vote_conf, score_conf))

    def _finish_online_cycle(self, used_data, cycle_start, results, scores_list, offsets):
        self._stop_stim_render_timer()
        self.setDefaultColor()
        self.start_cache = False
        self.online_trial_meta["stim_end_monotonic"] = float(self._now_mono())
        self.online_trial_meta["stim_end_unix"] = float(time.time())
        self.sliding_cycle_count += 1

        if getattr(self, "online_strategy", "sliding_vote") == "async_fbcca":
            valid_scores = [s for s in scores_list if isinstance(s, np.ndarray)]
            if len(valid_scores) > 0:
                sum_scores = np.sum(valid_scores, axis=0)
                winner = int(np.argmax(sum_scores))
                confidence = self._confidence_from_scores(sum_scores)
            else:
                winner, confidence = -1, 0.0
        else:
            winner, confidence = self._pick_online_vote_winner(results, scores_list)

        self.current_votes = [self.commands[r] if 0 <= int(r) < len(self.commands) else "?" for r in results]
        if 0 <= winner < len(self.commands):
            self.current_candidate = self.commands[winner]
            self.current_confidence = float(confidence)
        else:
            self.current_candidate = "-"
            self.current_confidence = 0.0
        self._update_live_panel()

        self._save_online_cycle_data(
            used_data=used_data,
            cycle_start=cycle_start,
            results=results,
            scores_list=scores_list,
            offsets=offsets,
            final_idx=winner,
            confidence=confidence,
        )

        if getattr(self, "mode", "online") == "test":
            self._update_test_accuracy(winner, confidence)
        elif winner >= 0:
            self._update_online_accuracy(winner)

        if winner >= 0 and confidence < self.min_confidence:
            self.last_gate_reason = f"置信度过低({confidence:.4f}<{self.min_confidence:.4f})"
            self.decision_info.setText(
                f"识别状态: 候选 {self.commands[winner]} 未执行 | conf={confidence:.4f} | {self.last_gate_reason}"
            )
        elif winner >= 0 and self._cooldown_ok(winner):
            self.set_result(winner, confidence)
        elif winner >= 0:
            self.decision_info.setText(
                f"识别状态: 候选 {self.commands[winner]} 未执行 | conf={confidence:.4f} | {self.last_gate_reason}"
            )
        else:
            self.decision_info.setText(f"识别状态: 本轮无有效结果 | votes={self.current_votes}")

        self._enter_online_phase("rest")
        self.show_label.setText(f"休息 {self.trial_rest_sec:.1f}s，准备下一轮")
        return

    def _online_tick(self):
        if self.finish or not self.start_flick:
            self.online_timer.stop()
            self._stop_stim_render_timer()
            self.online_window_active = False
            self.start_cache = False
            self.start_flick = False
            self.setDefaultColor()
            return

        if not self.online_window_active:
            self._start_online_window()
            return

        if self.online_phase == "cue":
            elapsed = self._phase_elapsed(self.online_phase_start_ts)
            self.progress.setValue(int(min(1.0, elapsed / max(self.trial_cue_sec, 1e-6)) * 100))
            if elapsed >= self.trial_cue_sec:
                self._enter_online_phase("stim")
                self.start_cache = True
                self.online_stim_frame_idx = 0
                self.online_last_render_frame_idx = -1
                self.online_last_progress_update_mono = 0.0
                self.online_stim_onset_mono = self._now_mono()
                self.online_stim_onset_unix = time.time()
                self.online_trial_meta["stim_onset_monotonic"] = float(self.online_stim_onset_mono)
                self.online_trial_meta["stim_onset_unix"] = float(self.online_stim_onset_unix)
                self._begin_stim_timing_log("online")
                self._render_stim_by_frame(0)
                self._start_stim_render_timer()
                if getattr(self, "mode", "online") == "test":
                    self.show_label.setText(f"Test stimulus: {self._current_test_truth()} ({self.online_window_sec:.2f}s)")
                    self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u6d4b\u8bd5\u523a\u6fc0\u9636\u6bb5")
                else:
                    self.show_label.setText(f"Online stimulus: {self.online_window_sec:.2f}s")
                    self.decision_info.setText("\u8bc6\u522b\u72b6\u6001: \u5728\u7ebf\u523a\u6fc0\u9636\u6bb5")
            return

        if self.online_phase == "stim":
            recv_start = float(self.online_trial_meta.get("recv_start_monotonic", np.nan))
            now_mono = self._now_mono()

            if np.isnan(recv_start):
                if now_mono - float(self.sliding_cycle_start_mono) > 5.0:
                    self._clear_cache_buffer()
                    self.start_cache = True
                    self.sliding_cycle_start_mono = now_mono
                return

            cycle_elapsed = now_mono - recv_start
            if now_mono - float(self.online_last_progress_update_mono) >= 0.20:
                self.online_last_progress_update_mono = now_mono
                self.progress.setValue(int(min(1.0, cycle_elapsed / max(self.online_window_sec, 1e-6)) * 100))

            if cycle_elapsed < self.online_window_sec:
                return

            cycle_samples = int(round(float(self.online_window_sec) * self.sample_rate_hz))
            window_samples = int(round(float(self._online_analysis_sec()) * self.sample_rate_hz))
            full_data = self._materialize_cache_data()
            used_data = self._extract_aligned_window(
                full_data=full_data,
                recv_start_mono=recv_start,
                stim_onset_mono=recv_start,
                sample_points=int(cycle_samples),
            )

            if not isinstance(used_data, np.ndarray) or used_data.ndim != 2 or used_data.shape[-1] < cycle_samples:
                self.decision_info.setText("Recognition: waiting for EEG data...")
                return

            results = []
            scores_list = []
            offsets = []
            if getattr(self, "online_strategy", "sliding_vote") == "async_fbcca":
                offsets = [max(0, int(cycle_samples - window_samples))]
            else:
                offsets = self._online_window_offsets(cycle_samples, window_samples)

            for offset_samples in offsets:
                end_sample = int(offset_samples) + int(window_samples)
                if end_sample > used_data.shape[-1]:
                    continue
                window = used_data[:, int(offset_samples):end_sample]
                try:
                    pred, scores, _ = self._classify_online_fbcca_window(window)
                    results.append(int(pred))
                    scores_list.append(scores)
                except Exception:
                    results.append(-1)
                    scores_list.append(None)

            self._finish_online_cycle(used_data, recv_start, results, scores_list, offsets)
            return

        if self.online_phase == "rest":
            elapsed = self._phase_elapsed(self.online_phase_start_ts)
            self.progress.setValue(int(max(0.0, 100.0 * (1.0 - min(1.0, elapsed / max(self.trial_rest_sec, 1e-6))))))
            if elapsed >= self.trial_rest_sec:
                self.online_trial_meta["trial_end_monotonic"] = float(self._now_mono())
                self.online_trial_meta["trial_end_unix"] = float(time.time())
                self._start_online_window()
            return

    def _save_online_cycle_data(self, used_data, cycle_start, results, scores_list=None, offsets=None, final_idx=-1, confidence=0.0):
        """Save raw data and labels for one online cycle."""
        try:
            savePath = self._subject_root()
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            fileNums = len(glob(os.path.join(savePath, '*.mat')))
            saveFile = os.path.join(savePath, f'{fileNums + 1}.mat')
            scores_list = scores_list or []
            offsets = offsets or []
            score_rows = []
            for scores in scores_list:
                if isinstance(scores, np.ndarray):
                    score_rows.append(np.asarray(scores, dtype=float).reshape(-1))
                else:
                    score_rows.append(np.full(len(self.commands), np.nan, dtype=float))
            scores_matrix = np.vstack(score_rows) if len(score_rows) > 0 else np.empty((0, len(self.commands)))

            final_idx = int(final_idx)
            pred_text = self.commands[final_idx] if 0 <= final_idx < len(self.commands) else ""
            truth_text = self._current_online_truth_label()
            truth_idx = self._label_to_index(truth_text) if truth_text else -1
            if truth_idx >= 0:
                label_idx = truth_idx
                label_text = truth_text
                label_source = "truth"
            else:
                label_idx = final_idx
                label_text = pred_text
                label_source = "prediction"

            savemat(saveFile, {
                'data': used_data,
                'raw_data': used_data,
                'sample_rate_hz': int(self.sample_rate_hz),
                'model_name': 'FBCCA_online',
                'online_strategy': str(getattr(self, "online_strategy", "sliding_vote")),
                'analysis_window_sec': float(self._online_analysis_sec()),
                'online_cycle_sec': float(self.online_window_sec),
                'analysis_delay_sec': float(self.analysis_delay_sec),
                'stim_onset_monotonic': float(cycle_start),
                'stim_onset_unix': float(time.time()),
                'sliding_results': np.array(results, dtype=int),
                'sliding_scores': scores_matrix,
                'sliding_offsets_samples': np.array(offsets, dtype=int),
                'sliding_offsets_sec': np.array(offsets, dtype=float) / float(self.sample_rate_hz),
                'cycle_count': int(self.sliding_cycle_count),
                'pred_label_text': pred_text,
                'pred_label_idx': int(final_idx),
                'label_text': label_text,
                'label_idx': int(label_idx),
                'label_source': label_source,
                'truth_label_text': truth_text,
                'truth_label_idx': int(truth_idx),
                'confidence': float(confidence),
                'stim_freqs_hz': np.array(self.sti_lst, dtype=float),
                'display_refresh_hz': float(self.stim_refresh_hz),
                'commands': np.array(self.commands, dtype=object),
            })
        except Exception as exc:
            print(f"[ONLINE_SAVE] failed: {exc}", flush=True)

    def set_result(self, idx, confidence=None):
        """
        澶勭悊璇嗗埆缁撴灉銆?
        鍙傛暟:
            idx: 鍛戒护绱㈠紩锛岃寖鍥?0-4銆?                0: 鍓嶈繘 (6.67Hz)
                1: 鍚庨€€ (7.5Hz)
                2: 宸﹁浆 (8.57Hz)
                3: 鍋滄 (12.0Hz)
                4: 鍙宠浆 (15.0Hz)
        """
        if idx < len(self.commands):
            command = self.commands[idx]
            send_to_robot = bool(getattr(self, "robot_control_enabled", False))
            if send_to_robot:
                self.show_label.setText(f"\u6267\u884c\u6307\u4ee4: {command}")
            else:
                self.show_label.setText(f"\u8bc6\u522b\u7ed3\u679c: {command}")
            if confidence is None:
                if send_to_robot:
                    self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u5df2\u6267\u884c {command}")
                else:
                    self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u6570\u636e\u6d4b\u8bd5 {command}\uff0c\u672a\u53d1\u9001\u5c0f\u8f66\u6307\u4ee4")
            else:
                if send_to_robot:
                    self.decision_info.setText(f"\u8bc6\u522b\u72b6\u6001: \u5df2\u6267\u884c {command} | \u7f6e\u4fe1\u5ea6 {confidence:.4f}")
                else:
                    self.decision_info.setText(
                        f"\u8bc6\u522b\u72b6\u6001: \u6570\u636e\u6d4b\u8bd5 {command} | \u7f6e\u4fe1\u5ea6 {confidence:.4f} | \u672a\u53d1\u9001\u5c0f\u8f66\u6307\u4ee4"
                    )
            self.current_candidate = command
            if confidence is not None:
                self.current_confidence = float(confidence)
            self._update_live_panel()

            speak_async_safe(command)
            
            print(f"=" * 50)
            print(f"璇嗗埆缁撴灉: {command}")
            print(f"鍛戒护绱㈠紩: {idx}")
            print(f"鍒烘縺棰戠巼: {self.sti_lst[idx]} Hz")
            print(f"=" * 50)
            self._append_result_history(command, idx, confidence=confidence, sent_to_robot=send_to_robot)
            if send_to_robot:
                self._send_robot_command_async(idx)
            else:
                print(f"\u6570\u636e\u6d4b\u8bd5\u6a21\u5f0f\uff0c\u672a\u53d1\u9001Socket\u6307\u4ee4: {command}")
            
        else:
            print(f"璇嗗埆缁撴灉绱㈠紩瓒呭嚭鑼冨洿: {idx}")

    def getData(self, data):
        self.push_eeg_data_threadsafe(data)

    def push_eeg_data_threadsafe(self, data):
        if self.finish and not self.training_collecting:
            return

        if self.start_cache:
            try:
                lock = getattr(self, "eeg_buffer_lock", None)
                if lock is None:
                    self.eeg_buffer_lock = threading.RLock()
                    lock = self.eeg_buffer_lock
                with lock:
                    now_mono = self._now_mono()
                    now_unix = time.time()
                    if self.mode in ("online", "test"):
                        meta = self.online_trial_meta if isinstance(self.online_trial_meta, dict) else None
                    else:
                        meta = self.training_trial_meta if isinstance(self.training_trial_meta, dict) else None
                    if isinstance(meta, dict):
                        update_receive_metadata(meta, data, now_mono, now_unix, self.sample_rate_hz)

                    self._append_cache_chunk(data)
            except Exception:
                pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.mode in ("online", "test"):
                self.start_sti_event()
            else:
                self._start_training_collect()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._reset_collection_state()
        if getattr(self, "_timer_resolution_enabled", False):
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._timer_resolution_enabled = False
        super().closeEvent(event)

def test_car_control():
    """Simple terminal smoke test for the car socket client."""
    ip = input(f"Car IP (default: {ROBOT_IP}): ").strip() or ROBOT_IP
    port_text = input(f"Port (default: {PORT}): ").strip()
    port = PORT if not port_text else int(port_text)
    client = RobotClient(ip, port)
    client.connect()
    if not client.connected:
        print("Connection failed")
        return
    commands = {0: "鍓嶈繘", 1: "鍚庨€€", 2: "宸﹁浆", 3: "鍋滄", 4: "鍙宠浆"}
    try:
        while True:
            value = input("Command [0-4], q to quit: ").strip()
            if value.lower() == "q":
                break
            idx = int(value)
            if idx not in commands:
                print("Command must be 0-4")
                continue
            print(f"Sending {idx}: {commands[idx]}")
            client.move(idx)
    finally:
        client.close()


if __name__ == "__main__":
    test_car_control()
