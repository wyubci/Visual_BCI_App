# -*- coding: utf-8 -*-
import socket
import pyttsx3
from collections import deque
import json
from datetime import datetime
import pickle
import struct
import importlib.util
import shutil
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

_camera_viewer_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "standalone_car_camera",
    "camera_viewer",
)
if _camera_viewer_dir not in sys.path:
    sys.path.append(_camera_viewer_dir)

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA
from models.OptimizedTDCA import ImprovedTDCA, TriBranchTDCA
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


def _load_car_video_panel_class(camera_viewer_dir):
    panel_path = os.path.join(camera_viewer_dir, "car_video_panel.py")
    if not os.path.exists(panel_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("car_video_panel", panel_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "CarVideoPanel", None)
    except Exception:
        return None


CarVideoPanel = _load_car_video_panel_class(_camera_viewer_dir)

try:
    import cv2
except Exception:
    cv2 = None

# ==========================================
# PC CLIENT FOR X1
# Run this on Windows
# ==========================================

ROBOT_IP = '10.186.179.92'  # CHANGE THIS IF NEEDED
PORT = 65432

CAMERA_ENDPOINTS = [
    ("127.0.0.1", 5001),
    (ROBOT_IP, 5000),
]
CAMERA_FLIP_CODE = 1

# SSVEP car control uses frame-locked square-wave stimuli.  On high-refresh
# displays (for example 165 Hz), a nominal frequency such as 6.67 Hz cannot be
# rendered exactly unless it is an integer divisor of the refresh rate.  The
# functions below choose integer frame periods first, then use the resulting
# actual frequencies for CCA/FBCCA/TDCA reference signals.  This keeps the
# visual stimulus and the decoder mathematically consistent.
LEGACY_CAR_FREQS = np.array([6.67, 7.50, 8.57, 12.00, 15.00], dtype=float)
SSVEP_TARGET_FREQS = np.array([8.250, 9.1667, 10.3125, 11.7857, 13.750], dtype=float)
SSVEP_FREQ_MIN_HZ = 6.0
SSVEP_FREQ_MAX_HZ = 15.5


def _detect_screen_refresh_rate(default_hz=60.0):
    """Return the primary screen refresh rate reported by Qt."""
    try:
        screen = QApplication.primaryScreen()
        if screen is None:
            return float(default_hz)
        rate = float(screen.refreshRate())
        if 30.0 <= rate <= 360.0 and np.isfinite(rate):
            return float(rate)
    except Exception:
        pass
    return float(default_hz)


def _common_refresh_periods(refresh_hz):
    """Hand-tuned periods for common monitor refresh rates."""
    r = float(refresh_hz)
    if 160.0 <= r <= 170.0:
        return [20, 18, 16, 14, 12]      # 165 Hz -> 8.25, 9.17, 10.31, 11.79, 13.75
    if 235.0 <= r <= 245.0:
        return [36, 32, 28, 20, 16]
    if 115.0 <= r <= 125.0:
        return [18, 16, 14, 10, 8]
    if 140.0 <= r <= 148.0:
        return [21, 19, 17, 12, 10]
    if 55.0 <= r <= 65.0:
        return [9, 8, 7, 5, 4]
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
            v1=61
            v2=0
            v3=64
            v4=5
            self.set_motor(v1, v2, v3, v4)
            time.sleep(2.68)
            self.set_motor(0,0,0,0)
        elif(ing==1):
            v1=-61
            v2=0
            v3=-64
            v4=-5
            self.set_motor(v1, v2, v3, v4)
            time.sleep(2.68)
            self.set_motor(0,0,0,0)
        elif(ing==2):
            v1=-65
            v2=-65
            v3=65
            v4=65
            self.set_motor(v1, v2, v3, v4)
        elif(ing==3):
            v1=0
            v2=0
            v3=0
            v4=0
            self.set_motor(v1, v2, v3, v4)

        elif(ing==4):
            v1=65
            v2=65
            v3=-65
            v4=-65
            self.set_motor(v1, v2, v3, v4)
        
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
        self.current_color = QColor(255, 255, 255, 255)
        self.default_color = QColor(255, 255, 255, 255)
        self.text_color = QColor(0, 0, 0)
        self.border_color = QColor(0, 0, 0)
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
        self.current_color = QColor(0, 0, 0, 255) if is_black else QColor(255, 255, 255, 255)
        self.text_color = QColor(255, 255, 255) if is_black else QColor(0, 0, 0)
        self.update()

    def setDefaultColor(self):
        self.current_color = self.default_color
        self.text_color = QColor(0, 0, 0)
        self.update()

    def paintEvent(self, a0):
        super().paintEvent(a0)

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHints(QPainter.Antialiasing)
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


class CameraStreamWorker(QThread):
    frameReady = pyqtSignal(QImage)
    statusChanged = pyqtSignal(str)

    def __init__(self, endpoints, flip_code, max_fps=20):
        super().__init__()
        self.endpoints = endpoints
        self.flip_code = flip_code
        self.max_fps = max_fps
        self.payload_size = struct.calcsize("Q")
        self._running = True
        self._socket = None

    def _connect_socket(self):
        for host, port in self.endpoints:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(2.0)
                sock.connect((host, port))
                sock.settimeout(None)
                self.statusChanged.emit(f"已连接视频流: {host}:{port}")
                return sock
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
        return None

    def run(self):
        if cv2 is None:
            self.statusChanged.emit("摄像头不可用：未检测到 opencv-python")
            return
        while self._running:
            self._socket = self._connect_socket()
            if self._socket is None:
                self.statusChanged.emit("Waiting for car video stream...")
                self.msleep(2000)
                continue

            buffer = b""
            try:
                while self._running:
                    while len(buffer) < self.payload_size and self._running:
                        chunk = self._socket.recv(self.payload_size - len(buffer))
                        if not chunk:
                            raise ConnectionError("socket closed")
                        buffer += chunk

                    if not self._running:
                        break

                    packed_msg_size = buffer[:self.payload_size]
                    buffer = buffer[self.payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]

                    while len(buffer) < msg_size and self._running:
                        needed = msg_size - len(buffer)
                        packet = self._socket.recv(min(needed, 65536))
                        if not packet:
                            raise ConnectionError("socket closed")
                        buffer += packet

                    if not self._running:
                        break

                    frame_data = buffer[:msg_size]
                    buffer = buffer[msg_size:]

                    try:
                        obj = pickle.loads(frame_data)
                        if hasattr(obj, "shape") and (len(obj.shape) == 1 or (len(obj.shape) == 2 and 1 in obj.shape)):
                            frame = cv2.imdecode(obj, cv2.IMREAD_COLOR)
                        else:
                            frame = obj

                        if frame is None:
                            continue

                        frame = cv2.flip(frame, self.flip_code)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame.shape
                        qimg = QImage(frame.data, w, h, c * w, QImage.Format_RGB888).copy()
                        self.frameReady.emit(qimg)
                        self.msleep(max(1, int(1000 / self.max_fps)))
                    except Exception:
                        continue

            except Exception as e:
                if self._running:
                    self.statusChanged.emit(f"视频流断开: {e}，重连中...")
                    self.msleep(1500)
            finally:
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None

        self.statusChanged.emit("摄像头已关闭")

    def stop(self):
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

class CarControlWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()
        
        refresh_hz = _detect_screen_refresh_rate(default_hz=60.0)
        self.stim_profile = build_stimulus_profile(refresh_hz, n_targets=5)
        self.stim_refresh_hz = float(self.stim_profile["refresh_hz"])
        self.stim_period_frames = [int(x) for x in self.stim_profile["period_frames"]]
        self.stim_duty_frames = [int(x) for x in self.stim_profile["duty_frames"]]
        self.sti_lst = [float(x) for x in self.stim_profile["actual_freqs_hz"]]
        
        self.commands = [
            "前进", "后退", "左转", "停止", "右转"
        ]
        
        self.cache_data = np.array([])
        self.cache_chunks = []
        self.cache_points = 0
        self.sample_buffer = EegSampleBuffer()
        self.start_flick = False
        self.finish = True
        self.start_cache = False
        self.continuous_mode = False

        self.sample_rate_hz = 250
        self.analysis_delay_sec = 0.14
        self.model_times_sec = 4.0
        self.analysis_window_sec = self.model_times_sec
        self.training_window_sec = self.model_times_sec
        self.training_sample_points = int(round(self.model_times_sec * self.sample_rate_hz))
        self.online_window_sec = self.model_times_sec
        self.online_sample_points = int(round(self.model_times_sec * self.sample_rate_hz))
        self.model_name = "TDCA"
        self.min_confidence = 0.02

        self.trial_cue_sec = 0.5
        self.trial_rest_sec = 0.5
        self.acquisition_tail_sec = 0.5

        self.stim_frame_interval_ms = max(1, int(round(1000.0 / self.stim_refresh_hz)))
        self.stim_poll_interval_ms = max(1, int(round(1000.0 / (self.stim_refresh_hz * 2.0))))

        self.fbcca = self._create_classifier(self.model_times_sec)
        self.quality_calc_timer = QTimer(self)
        self.quality_calc_timer.setInterval(30)
        self.quality_calc_timer.timeout.connect(self._process_quality_calc_batch)
        self.quality_calc_pending = []
        self.quality_calc_rows = []
        self.quality_calc_batch_size = 4
        
        self._initLayout()
        self._initItems()

        self.decision_buffer = deque(maxlen=3)
        self.vote_threshold = 2
        self.command_cooldown_sec = 1.0
        self.last_command_time = 0.0
        self.last_command_idx = None
        self.execution_mode = "vote"
        self.last_gate_reason = ""

        self.current_candidate = "-"
        self.current_confidence = 0.0
        self.current_votes = []
        self.online_eval_truth = "不统计"
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
        self.video_net_ok = False

        self.mode = "online"
        self.training_collecting = False
        self.training_collected = 0
        self.training_plan = []
        self.training_timeout_sec = 6.0
        self.training_target_label = ""
        self.training_collect_start_time = 0.0
        self.selected_weight_file = ""
        self.dataset_file_map = {}
        self.online_time_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.training_hint.setText(f"在线模式: 回车启动连续 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.online_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)")

        self.online_window_active = False
        self.online_window_start_time = 0.0
        self.online_phase = "idle"
        self.online_phase_start_ts = 0.0
        self.online_stim_frame_idx = 0
        self.online_last_render_frame_idx = -1
        self.online_stim_onset_mono = 0.0
        self.online_stim_onset_unix = 0.0
        self.online_trial_meta = {}

        self.training_phase = "idle"
        self.training_phase_start_ts = 0.0
        self.training_stim_frame_idx = 0
        self.training_last_render_frame_idx = -1
        self.training_stim_onset_mono = 0.0
        self.training_stim_onset_unix = 0.0
        self.training_pending_success = False
        self.training_trial_meta = {}

        self.online_timer = QTimer(self)
        self.online_timer.setTimerType(Qt.PreciseTimer)
        self.online_timer.setInterval(self.stim_poll_interval_ms)
        self.online_timer.timeout.connect(self._online_tick)

        self.training_timer = QTimer(self)
        self.training_timer.setTimerType(Qt.PreciseTimer)
        self.training_timer.setInterval(self.stim_poll_interval_ms)
        self.training_timer.timeout.connect(self._training_tick)

        self.camera_thread = None
        self.tunnel_retry_timer = None
        self.video_panel = None
        
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
        self.show_label.setText("视觉脑机接口控制系统")
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
        self.system_status = QLabel("系统状态: 运行中")
        self.system_status.setObjectName("statusTagOk")
        right_status.addWidget(self.system_status)
        self.top_network_status = QLabel("连接状态: 未连接")
        self.top_network_status.setObjectName("statusTagBad")
        right_status.addWidget(self.top_network_status)
        self.time_status = QLabel(time.strftime("时间: %Y-%m-%d %H:%M:%S"))
        self.time_status.setStyleSheet("font-size: 12px; color: #9FB0C9;")
        right_status.addWidget(self.time_status)
        top_title_row.addLayout(right_status)
        top_layout.addLayout(top_title_row)

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(lambda: self.time_status.setText(time.strftime("时间: %Y-%m-%d %H:%M:%S")))
        self.clock_timer.start()

        param_row = QHBoxLayout()
        param_row.setSpacing(10)

        def make_field_label(text):
            lb = CaptionLabel(text)
            lb.setStyleSheet("font-size: 13px; color: #E5E7EB; font-weight: 500;")
            return lb

        self.mode_bar = QGroupBox("模式设置")
        mode_layout = QGridLayout(self.mode_bar)
        mode_layout.setContentsMargins(10, 12, 10, 10)
        mode_layout.setSpacing(8)

        mode_layout.addWidget(make_field_label("模式:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["在线模式", "训练模式", "测试模式"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo, 0, 1)
        mode_layout.addWidget(make_field_label("样本数:"), 0, 2)
        self.train_count_spin = QSpinBox()
        self.train_count_spin.setRange(1, 500)
        self.train_count_spin.setValue(20)
        self.train_count_spin.setMinimumWidth(70)
        self.train_count_spin.valueChanged.connect(self._on_training_plan_params_changed)
        mode_layout.addWidget(self.train_count_spin, 0, 3)
        mode_layout.addWidget(make_field_label("在线测试时长:"), 0, 4)
        self.online_time_combo = QComboBox()
        self.online_time_combo.addItems(["1s", "2s", "3s", "4s"])
        self.online_time_combo.setCurrentText("4s")
        self.online_time_combo.currentTextChanged.connect(self._on_online_time_changed)
        self.online_time_combo.setMinimumWidth(84)
        mode_layout.addWidget(self.online_time_combo, 0, 5)

        mode_layout.addWidget(make_field_label("训练时长:"), 0, 6)
        self.training_time_combo = QComboBox()
        self.training_time_combo.addItems(["1s", "2s", "3s", "4s"])
        self.training_time_combo.setCurrentText("4s")
        self.training_time_combo.currentTextChanged.connect(self._on_training_time_changed)
        self.training_time_combo.setMinimumWidth(84)
        mode_layout.addWidget(self.training_time_combo, 0, 7)

        mode_layout.addWidget(make_field_label("标签(逗号分隔):"), 1, 0)
        self.train_labels_edit = QLineEdit("前进,后退,左转,停止,右转")
        self.train_labels_edit.setMinimumWidth(220)
        self.train_labels_edit.editingFinished.connect(self._on_training_plan_params_changed)
        mode_layout.addWidget(self.train_labels_edit, 1, 1, 1, 5)
        mode_layout.setColumnStretch(1, 2)
        mode_layout.setColumnStretch(5, 1)

        self.model_bar = QGroupBox("模型参数")
        model_layout = QHBoxLayout(self.model_bar)
        model_layout.setContentsMargins(10, 12, 10, 10)
        model_layout.setSpacing(8)
        model_layout.addWidget(make_field_label("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TriBranchTDCA", "ImprovedTDCA", "TDCA", "FBCCA", "CCA"])
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(make_field_label("采样率:"))
        self.sampling_rate_combo = QComboBox()
        self.sampling_rate_combo.addItems(["250Hz"])
        self.sampling_rate_combo.setCurrentText("250Hz")
        self.sampling_rate_combo.setEnabled(False)
        self.sampling_rate_combo.currentTextChanged.connect(self._on_sampling_rate_changed)
        model_layout.addWidget(self.sampling_rate_combo)
        model_layout.addWidget(make_field_label("置信度阈值:"))

        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setRange(0.00, 1.00)
        self.conf_threshold_spin.setSingleStep(0.01)
        self.conf_threshold_spin.setValue(self.min_confidence)
        self.conf_threshold_spin.valueChanged.connect(self._on_min_confidence_changed)
        model_layout.addWidget(self.conf_threshold_spin)

        exec_group = QGroupBox("执行策略")
        exec_layout = QHBoxLayout(exec_group)
        exec_layout.setContentsMargins(10, 12, 10, 10)
        exec_layout.setSpacing(8)
        exec_layout.addWidget(make_field_label("策略:"))
        self.exec_mode_combo = QComboBox()
        self.exec_mode_combo.addItems(["投票执行", "候选直发"])
        self.exec_mode_combo.setCurrentIndex(0)
        self.exec_mode_combo.currentIndexChanged.connect(self._on_execution_mode_changed)
        exec_layout.addWidget(self.exec_mode_combo)
        self.training_hint = CaptionLabel("训练/测试: 回车启动 trial(提示→刺激→间隔)")
        self.training_hint.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        self.training_hint.setWordWrap(True)
        self.training_hint.setMaximumWidth(280)
        self.training_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        exec_layout.addWidget(self.training_hint, stretch=1)
        exec_layout.addWidget(make_field_label("在线真值:"))
        self.online_truth_combo = QComboBox()
        self.online_truth_combo.addItem("不统计")
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

        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #1F3350;border-radius:10px;background:#050B12;}"
            "QTabBar::tab{background:#0B1220;color:#AFC4E8;padding:8px 18px;border:1px solid #1F3350;}"
            "QTabBar::tab:selected{background:#1E3355;color:#FFFFFF;font-weight:600;}"
        )
        self.content_layout.addWidget(self.workspace_tabs, stretch=1)

        acquisition_page = QWidget()
        acquisition_layout = QVBoxLayout(acquisition_page)
        acquisition_layout.setContentsMargins(8, 8, 8, 8)
        acquisition_layout.setSpacing(10)
        self.workspace_tabs.addTab(acquisition_page, "采集与在线测试")

        offline_page = QWidget()
        self.offline_layout = QVBoxLayout(offline_page)
        self.offline_layout.setContentsMargins(8, 8, 8, 8)
        self.offline_layout.setSpacing(10)
        self.workspace_tabs.addTab(offline_page, "离线数据处理")

        self.mainSplitter = QSplitter(Qt.Vertical)
        self.mainSplitter.setChildrenCollapsible(False)
        acquisition_layout.addWidget(self.mainSplitter, stretch=1)

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

        control_camera_panel = make_card()
        control_camera_layout = QVBoxLayout(control_camera_panel)
        control_camera_layout.setContentsMargins(10, 10, 10, 10)
        control_camera_layout.setSpacing(8)
        upper_layout.addWidget(control_camera_panel, stretch=1)
        control_camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        control_camera_panel.setMinimumHeight(620)

        panel_title = StrongBodyLabel("核心控制按钮区")
        panel_title.setObjectName("cardTitle")
        control_camera_layout.addWidget(panel_title)

        self.sti_rects = []
        scene_host = QWidget()
        scene_layout = QVBoxLayout(scene_host)
        scene_layout.setContentsMargins(4, 4, 4, 4)
        scene_layout.setSpacing(8)

        self.forward_rect = StiRect("↑\n前进", scene_host, self.sti_lst[0], fontSize=34)
        self.forward_rect.setMinimumSize(120, 120)
        self.forward_rect.setMaximumSize(156, 156)
        self.forward_rect.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sti_rects.append(self.forward_rect)

        self.backward_rect = StiRect("↓\n后退", scene_host, self.sti_lst[1], fontSize=34)
        self.backward_rect.setMinimumSize(120, 120)
        self.backward_rect.setMaximumSize(156, 156)
        self.backward_rect.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sti_rects.append(self.backward_rect)

        self.left_rect = StiRect("←\n左转", scene_host, self.sti_lst[2], fontSize=32)
        self.left_rect.setMinimumSize(112, 112)
        self.left_rect.setMaximumSize(146, 146)
        self.left_rect.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sti_rects.append(self.left_rect)

        self.stop_rect = StiRect("■\n停止", scene_host, self.sti_lst[3], fontSize=34)
        self.stop_rect.setMinimumSize(112, 112)
        self.stop_rect.setMaximumSize(146, 146)
        self.stop_rect.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sti_rects.append(self.stop_rect)

        self.right_rect = StiRect("→\n右转", scene_host, self.sti_lst[4], fontSize=32)
        self.right_rect.setMinimumSize(112, 112)
        self.right_rect.setMaximumSize(146, 146)
        self.right_rect.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sti_rects.append(self.right_rect)
        self._apply_stimulus_profile_to_rects()

        top_row = QHBoxLayout()
        top_row.setSpacing(0)
        top_row.addStretch(2)
        top_row.addWidget(self.forward_rect, 0, Qt.AlignCenter)
        top_row.addStretch(6)
        top_row.addWidget(self.backward_rect, 0, Qt.AlignCenter)
        top_row.addStretch(2)
        scene_layout.addLayout(top_row)

        self.camera_view = QLabel("摄像头未启用")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.hide()
        scene_layout.addStretch(2)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(0)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.left_rect, 0, Qt.AlignCenter)
        bottom_row.addStretch(2)
        bottom_row.addWidget(self.stop_rect, 0, Qt.AlignCenter)
        bottom_row.addStretch(2)
        bottom_row.addWidget(self.right_rect, 0, Qt.AlignCenter)
        bottom_row.addStretch(1)
        scene_layout.addLayout(bottom_row)

        control_camera_layout.addWidget(scene_host, 1)

        status_panel = make_card()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(10)
        lower_layout.addWidget(status_panel, stretch=3)
        status_title = StrongBodyLabel("状态信息可视化区")
        status_title.setObjectName("cardTitle")
        status_layout.addWidget(status_title)

        self.camera_status = CaptionLabel("摄像头状态: 初始化中")
        self.camera_status.setStyleSheet("font-size: 12px; color: #E5E7EB; font-weight: 500;")
        status_layout.addWidget(self.camera_status)

        self.network_status = CaptionLabel("网络状态: 控制未连接 | 视频未连接")
        self.network_status.setStyleSheet("font-size: 12px; color: #E5E7EB; font-weight: 600;")
        status_layout.addWidget(self.network_status)

        self.freq_status = CaptionLabel(self._online_status_text())
        self.freq_status.setStyleSheet("font-size: 12px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.freq_status)

        ctrl_row = QGridLayout()
        ctrl_row.setHorizontalSpacing(6)
        ctrl_row.setVerticalSpacing(6)
        ctrl_row.addWidget(make_field_label("控制IP:"), 0, 0)
        self.robot_ip_edit = QLineEdit(ROBOT_IP)
        self.robot_ip_edit.setMinimumWidth(120)
        ctrl_row.addWidget(self.robot_ip_edit, 0, 1)
        ctrl_row.addWidget(make_field_label("端口:"), 0, 2)
        self.robot_port_spin = QSpinBox()
        self.robot_port_spin.setRange(1, 65535)
        self.robot_port_spin.setValue(PORT)
        self.robot_port_spin.setMinimumWidth(70)
        ctrl_row.addWidget(self.robot_port_spin, 0, 3)
        self.test_conn_btn = PushButton("测试连接")
        self.test_conn_btn.setMinimumWidth(92)
        self.test_conn_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.test_conn_btn.clicked.connect(self._test_robot_connection)
        ctrl_row.addWidget(self.test_conn_btn, 0, 4)
        ctrl_row.setColumnStretch(1, 1)
        status_layout.addLayout(ctrl_row)

        self.decision_info = CaptionLabel("识别状态: 待开始")
        self.decision_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.decision_info)

        self.command_info = CaptionLabel("当前识别命令: -")
        self.command_info.setStyleSheet("font-size: 16px; color: #7CB0FF; font-weight: 600;")
        status_layout.addWidget(self.command_info)

        self.confidence_bar = ProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        status_layout.addWidget(self.confidence_bar)
        self.confidence_info = CaptionLabel("置信度: 0.0000")
        self.confidence_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.confidence_info)

        self.vote_info = CaptionLabel("最近投票: -")
        self.vote_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        self.vote_info.setWordWrap(True)
        status_layout.addWidget(self.vote_info)

        self.online_acc_info = CaptionLabel("在线准确率: -")
        self.online_acc_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.online_acc_info)

        self.train_info = CaptionLabel("训练进度: 0/0")
        self.train_info.setStyleSheet("font-size: 13px; color: #D1D5DB; font-weight: 500;")
        status_layout.addWidget(self.train_info)
        self.train_progress_bar = ProgressBar()
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        status_layout.addWidget(self.train_progress_bar)
        status_layout.addStretch()

        data_panel = make_card()
        data_layout = QVBoxLayout(data_panel)
        data_layout.setContentsMargins(12, 12, 12, 12)
        data_layout.setSpacing(10)
        data_panel.setMinimumHeight(540)
        self.offline_layout.addWidget(data_panel, stretch=1)

        data_title = StrongBodyLabel("离线数据处理与权重管理")
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
        self.data_tabs.addTab(train_tab, "训练数据")

        score_tab = QWidget()
        score_layout = QVBoxLayout(score_tab)
        score_layout.setContentsMargins(8, 8, 8, 8)
        score_layout.setSpacing(8)
        self.data_tabs.addTab(score_tab, "数据分析")

        train_op_row = QHBoxLayout()
        self.refresh_data_btn = PushButton("刷新数据")
        self.refresh_data_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.refresh_data_btn.clicked.connect(self.refresh_train_dataset_view)
        train_op_row.addWidget(self.refresh_data_btn)
        self.auto_pick_btn = PushButton("勾选当前时长全量样本")
        self.auto_pick_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.auto_pick_btn.clicked.connect(self.auto_check_high_quality_samples)
        train_op_row.addWidget(self.auto_pick_btn)
        self.score_pick_train_btn = PushButton("质量分析并勾选")
        self.score_pick_train_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.score_pick_train_btn.clicked.connect(self.score_and_check_train_samples)
        train_op_row.addWidget(self.score_pick_train_btn)
        self.train_weight_btn = PushButton("勾选数据训练权重")
        self.train_weight_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.train_weight_btn.clicked.connect(self.train_weight_from_checked_files)
        train_op_row.addWidget(self.train_weight_btn)
        
        self.evaluate_checked_btn = PushButton("离线测试勾选数据")
        self.evaluate_checked_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.evaluate_checked_btn.clicked.connect(self.evaluate_checked_data)
        train_op_row.addWidget(self.evaluate_checked_btn)
        
        train_tab_layout.addLayout(train_op_row)

        self.train_data_tree = QTreeWidget()
        self.train_data_tree.setHeaderLabels(["日期/文件", "标签", "采样点", "质量分"])
        self.train_data_tree.setMinimumHeight(170)
        self.train_data_tree.setAlternatingRowColors(True)
        train_tab_layout.addWidget(self.train_data_tree)

        weight_row = QHBoxLayout()
        weight_row.addWidget(make_field_label("在线权重:"))
        self.weight_combo = QComboBox()
        weight_row.addWidget(self.weight_combo, stretch=1)
        self.load_weight_btn = PushButton("加载权重")
        self.load_weight_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 14px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.load_weight_btn.clicked.connect(self.load_selected_weight_file)
        weight_row.addWidget(self.load_weight_btn)
        train_tab_layout.addLayout(weight_row)

        self.weight_status = CaptionLabel("当前权重: 默认(无)")
        self.weight_status.setStyleSheet("font-size: 13px; color: #9FB0C9;")
        train_tab_layout.addWidget(self.weight_status)
        train_tab_layout.addStretch()

        score_op_row = QHBoxLayout()
        self.score_refresh_btn = PushButton("分析勾选数据")
        self.score_refresh_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.score_refresh_btn.clicked.connect(self.refresh_score_dataset_view)
        score_op_row.addWidget(self.score_refresh_btn)
        score_op_row.addWidget(make_field_label("SNR阈值:"))
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(-20.0, 40.0)
        self.score_threshold_spin.setDecimals(1)
        self.score_threshold_spin.setSingleStep(0.5)
        self.score_threshold_spin.setValue(1.0)
        self.score_threshold_spin.setMinimumWidth(78)
        score_op_row.addWidget(self.score_threshold_spin)
        self.score_auto_btn = PushButton("勾选可用样本")
        self.score_auto_btn.setStyleSheet("QPushButton{background:#162640;color:#DDE7F7;border:1px solid #35507A;border-radius:10px;padding:6px 12px;}QPushButton:hover{background:#1E3355;}")
        self.score_auto_btn.clicked.connect(self.auto_check_scored_samples)
        score_op_row.addWidget(self.score_auto_btn)
        self.score_save_btn = PushButton("保存分析勾选")
        self.score_save_btn.setStyleSheet("QPushButton{background:#15803D;color:#F0FDF4;border:1px solid #34D399;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#16A34A;}")
        self.score_save_btn.clicked.connect(self.save_checked_scored_samples)
        score_op_row.addWidget(self.score_save_btn)
        self.score_sync_btn = PushButton("同步到训练页")
        self.score_sync_btn.setStyleSheet("QPushButton{background:#2F6FD6;color:#F7FAFF;border:1px solid #5E8EDC;border-radius:10px;padding:6px 12px;font-weight:600;}QPushButton:hover{background:#3E7CE2;}")
        self.score_sync_btn.clicked.connect(self.sync_score_checked_to_train_samples)
        score_op_row.addWidget(self.score_sync_btn)
        score_op_row.addStretch()
        score_layout.addLayout(score_op_row)

        self.score_tree = QTreeWidget()
        self.score_tree.setColumnCount(11)
        self.score_tree.setHeaderLabels([
            "文件", "标签", "点数", "RMS", "目标功率", "SNR(dB)", "峰值Hz", "FBCCA", "CCA", "TDCA", "结论"
        ])
        self.score_tree.setMinimumHeight(260)
        self.score_tree.setAlternatingRowColors(True)
        self.score_tree.setSortingEnabled(True)
        score_layout.addWidget(self.score_tree, stretch=1)

        self.score_status = CaptionLabel("数据分析: 请先在训练数据页勾选样本，再点击分析")
        self.score_status.setStyleSheet("font-size: 13px; color: #9FB0C9;")
        score_layout.addWidget(self.score_status)

        self.refresh_train_dataset_view()
        self.refresh_weight_file_list()
    
    def _stim_freq_summary(self):
        pairs = []
        for i, freq in enumerate(self.sti_lst):
            cmd = self.commands[i] if i < len(self.commands) else f"C{i + 1}"
            period = self.stim_period_frames[i] if i < len(self.stim_period_frames) else None
            if period is None:
                pairs.append(f"{cmd}:{freq:.2f}Hz")
            else:
                pairs.append(f"{cmd}:{freq:.2f}Hz/{period}帧")
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

    def _render_stim_by_frame(self, frame_index):
        for rect in self.sti_rects:
            rect.changeColorByFrame(rect.sti, frame_index, self.stim_refresh_hz)

    def _start_camera(self):
        if self.camera_thread is not None:
            return
        self.camera_thread = CameraStreamWorker(CAMERA_ENDPOINTS, CAMERA_FLIP_CODE, max_fps=20)
        self.camera_thread.frameReady.connect(self._on_camera_frame)
        self.camera_thread.statusChanged.connect(self._on_camera_status)
        self.camera_thread.start()

    def _init_tunnel_retry(self):
        self.tunnel_retry_timer = QTimer(self)
        self.tunnel_retry_timer.setInterval(5000)
        self.tunnel_retry_timer.timeout.connect(self._start_tunnel_async)
        QTimer.singleShot(0, self._start_tunnel_async)
        self.tunnel_retry_timer.start()

    def _start_tunnel_async(self):
        return

    def _stop_camera(self):
        if self.camera_thread is None:
            return
        self.camera_thread.stop()
        self.camera_thread.wait(800)
        self.camera_thread = None

    def _on_camera_frame(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.camera_view.setPixmap(
            pix.scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_camera_status(self, text):
        self.camera_status.setText(f"摄像头状态: {text}")
        self.video_net_ok = "已连接" in str(text)
        self._refresh_network_status()

    def _on_video_worker_status(self, text):
        self.camera_status.setText(f"摄像头状态: {text}")
        txt = str(text).lower()
        self.video_net_ok = ("connected" in txt) or ("已连接" in str(text))
        self._refresh_network_status()

    def _on_video_worker_frame(self, _):
        if not self.video_net_ok:
            self.video_net_ok = True
            self._refresh_network_status()

    def _refresh_network_status(self):
        ip, port = self._get_robot_endpoint()
        robot_text = "控制已连接" if self.robot_net_ok else "控制未连接"
        video_text = "视频已连接" if self.video_net_ok else "视频未连接"
        self.network_status.setText(f"网络状态: {robot_text}({ip}:{port}) | {video_text}")
        self.network_status.setStyleSheet(
            "font-size: 13px; color: #10B981; font-weight: 600;" if self.robot_net_ok
            else "font-size: 13px; color: #EF4444; font-weight: 600;"
        )
        if hasattr(self, "top_network_status"):
            self.top_network_status.setText(f"连接状态: {robot_text}")
            self.top_network_status.setObjectName("statusTagOk" if self.robot_net_ok else "statusTagBad")
            self.top_network_status.style().unpolish(self.top_network_status)
            self.top_network_status.style().polish(self.top_network_status)

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

    def _test_robot_connection(self):
        ip, port = self._get_robot_endpoint()
        try:
            client = RobotClient(ip, port)
            client.connect()
            self.robot_net_ok = bool(client.connected)
            if client.connected:
                self.decision_info.setText(f"识别状态: 控制连接成功 {ip}:{port}")
            else:
                self.decision_info.setText(f"识别状态: 控制连接失败 {ip}:{port}")
            client.close()
        except Exception:
            self.robot_net_ok = False
            self.decision_info.setText(f"识别状态: 控制连接失败 {ip}:{port}")
        self._refresh_network_status()

    def _update_live_panel(self):
        self.command_info.setText(f"当前识别命令: {self.current_candidate}")
        bar_val = int(max(0.0, min(1.0, self.current_confidence / 0.25)) * 100)
        self.confidence_bar.setValue(bar_val)
        self.confidence_info.setText(f"置信度: {self.current_confidence:.4f}")
        color_map = {
            "前进": "#3B82F6",
            "后退": "#8B5CF6",
            "左转": "#F59E0B",
            "右转": "#0EA5E9",
            "停止": "#EF4444",
        }
        if len(self.current_votes) > 0:
            tags = []
            for cmd in self.current_votes[-3:]:
                color = color_map.get(cmd, "#64748B")
                tags.append(
                    f"<span style='background:{color}; color:#FFFFFF; border-radius:8px; padding:2px 8px; margin-right:4px;'>"
                    f"{cmd}</span>"
                )
            self.vote_info.setText("最近投票: " + " ".join(tags))
        else:
            self.vote_info.setText("最近投票: -")

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
        self._clear_cache_buffer()
        self.setDefaultColor()
        self.progress.setValue(0)

    def _clear_cache_buffer(self):
        if not hasattr(self, "sample_buffer"):
            self.sample_buffer = EegSampleBuffer()
        self.sample_buffer.clear()
        self.cache_data = np.array([])
        self.cache_chunks = []
        self.cache_points = 0

    def _append_cache_chunk(self, data):
        if not hasattr(self, "sample_buffer"):
            self.sample_buffer = EegSampleBuffer()
        if self.sample_buffer.append(data):
            self.cache_chunks = []
            self.cache_points = self.sample_buffer.points

    def _cache_sample_count(self):
        if hasattr(self, "sample_buffer"):
            return int(self.sample_buffer.points)
        return int(self.cache_points)

    def _materialize_cache_data(self):
        if hasattr(self, "sample_buffer"):
            return self.sample_buffer.materialize()
        return np.array([])

    def _extract_aligned_window(self, full_data, recv_start_mono, stim_onset_mono, sample_points):
        return extract_aligned_window(
            full_data,
            recv_start_mono,
            stim_onset_mono,
            sample_points,
            AcquisitionConfig(self.sample_rate_hz, self.analysis_delay_sec),
        )

    def _aligned_window_indices(self, recv_start_mono, stim_onset_mono, sample_points):
        return aligned_window_indices(
            recv_start_mono,
            stim_onset_mono,
            sample_points,
            AcquisitionConfig(self.sample_rate_hz, self.analysis_delay_sec),
        )

    def _has_aligned_window_ready(self, full_data, recv_start_mono, stim_onset_mono, sample_points):
        used_data = self._extract_aligned_window(
            full_data=full_data,
            recv_start_mono=recv_start_mono,
            stim_onset_mono=stim_onset_mono,
            sample_points=int(sample_points),
        )
        return isinstance(used_data, np.ndarray) and used_data.ndim == 2 and used_data.shape[-1] >= int(sample_points)

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
        return preprocess_model_input(data, self.sample_rate_hz)

    def _on_mode_changed(self, index):
        if self.training_collecting or self.start_flick:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex({"online": 0, "train": 1, "test": 2}.get(self.mode, 0))
            self.mode_combo.blockSignals(False)
            return
        self.mode = "online" if index == 0 else ("train" if index == 1 else "test")
        self._reset_collection_state()
        if self.mode == "online":
            self.show_label.setText("脑控小车在线控制")
            self.online_time_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            if hasattr(self, "training_time_combo"):
                self.training_time_combo.setEnabled(True)
            if hasattr(self, "sampling_rate_combo"):
                self.sampling_rate_combo.setEnabled(False)
            self._reset_online_accuracy_stats()
            self.training_hint.setText(f"在线模式: 回车启动连续 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.online_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)")
            self.refresh_weight_file_list()
        elif self.mode == "train":
            self.show_label.setText("脑控小车训练模式")
            self.online_time_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            if hasattr(self, "training_time_combo"):
                self.training_time_combo.setEnabled(True)
            if hasattr(self, "sampling_rate_combo"):
                self.sampling_rate_combo.setEnabled(False)
            self._reset_online_accuracy_stats()
            self.training_hint.setText(
                f"训练模式: 回车启动 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.training_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)"
            )
            self.training_collected = 0
            self.training_plan = self._build_training_plan()
            self._update_training_progress()
            self.refresh_train_dataset_view()
        else:
            self.show_label.setText("脑控小车测试模式")
            self.online_time_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            if hasattr(self, "training_time_combo"):
                self.training_time_combo.setEnabled(True)
            if hasattr(self, "sampling_rate_combo"):
                self.sampling_rate_combo.setEnabled(False)
            self._reset_test_accuracy_stats(reset_plan=True)
            self.training_hint.setText(
                f"测试模式: 按标签顺序采集并分析(提示{self.trial_cue_sec:.1f}s→刺激{self.online_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)"
            )
            self.refresh_weight_file_list()

    def _create_classifier(self, times_sec):
        if self.model_name == "FBCCA":
            return FBCCA(3, times_sec, self.sti_lst, sample_rate=self.sample_rate_hz)
        if self.model_name == "CCA":
            return CCA(3, times_sec, self.sti_lst, sample_rate=self.sample_rate_hz)
        if self.model_name == "IMPROVEDTDCA":
            return ImprovedTDCA(3, times_sec, self.sti_lst, sample_rate=self.sample_rate_hz)
        if self.model_name == "TRIBRANCHTDCA":
            return TriBranchTDCA(3, times_sec, self.sti_lst, sample_rate=self.sample_rate_hz)
        return TDCA(3, times_sec, self.sti_lst, sample_rate=self.sample_rate_hz)

    def _is_supervised_model(self):
        return self.model_name in {"TDCA", "IMPROVEDTDCA", "TRIBRANCHTDCA"}

    def _update_sample_points(self):
        self.analysis_delay_sec = 0.14
        online_times = 4.0
        try:
            if hasattr(self, "online_time_combo"):
                online_times = float(str(self.online_time_combo.currentText()).replace("s", "").strip())
        except Exception:
            online_times = 4.0
        self.model_times_sec = float(np.clip(online_times, 1.0, 4.0))
        self.analysis_window_sec = self.model_times_sec
        training_times = 4.0
        try:
            if hasattr(self, "training_time_combo"):
                training_times = float(str(self.training_time_combo.currentText()).replace("s", "").strip())
        except Exception:
            training_times = 4.0
        self.training_window_sec = float(np.clip(training_times, 1.0, 4.0))
        self.online_window_sec = self.model_times_sec
        self.training_sample_points = max(1, int(round(self.training_window_sec * self.sample_rate_hz)))
        self.online_sample_points = max(1, int(round(self.model_times_sec * self.sample_rate_hz)))

    def _rebuild_classifier_with_adaptation(self):
        self._update_sample_points()
        self.fbcca = self._create_classifier(self.model_times_sec)
        self._sync_required_sample_points()

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
            f"{self.model_name}: 刺激{self.online_window_sec:.2f}s | 分析{self.analysis_window_sec:.2f}s"
            f"(+延迟{self.analysis_delay_sec:.2f}s) | EEG {self.sample_rate_hz}Hz | 屏幕{self.stim_refresh_hz:.1f}Hz | "
            f"实际刺激: {self._stim_freq_summary()}"
        )

    def _sync_required_sample_points(self):
        online_base = max(1, int(round(self.analysis_window_sec * self.sample_rate_hz)))
        train_base = max(1, int(round(self.training_window_sec * self.sample_rate_hz)))
        self.online_sample_points = online_base
        self.training_sample_points = train_base
        try:
            train_tdca = TDCA(3, self.training_window_sec, self.sti_lst, sample_rate=self.sample_rate_hz)
            self.training_sample_points = max(train_base, int(getattr(train_tdca, "required_points", train_base)))
        except Exception:
            self.training_sample_points = train_base
        if self._is_supervised_model():
            self.online_sample_points = max(online_base, int(getattr(self.fbcca, "required_points", online_base)))

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
                        "当前权重: 刺激频率不匹配，请用当前屏幕刷新率重新采集/训练"
                    )
                    if hasattr(self.fbcca, "reset_frequency_weights"):
                        self.fbcca.reset_frequency_weights()
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                    return
                if cfg_model and cfg_model != self.model_name:
                    self.weight_status.setText(
                        f"当前权重: 不兼容(权重模型={cfg_model}, 当前={self.model_name})"
                    )
                    if hasattr(self.fbcca, "reset_frequency_weights"):
                        self.fbcca.reset_frequency_weights()
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                    return
                if cfg_fs != self.sample_rate_hz:
                    self.weight_status.setText(
                        f"当前权重: 采样率不匹配(权重={cfg_fs}Hz, 当前={self.sample_rate_hz}Hz)"
                    )
                    if hasattr(self.fbcca, "reset_frequency_weights"):
                        self.fbcca.reset_frequency_weights()
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                    return
                weights = np.asarray(cfg.get("weights", []), dtype=float)
                self.fbcca.set_frequency_weights(weights)
                scale = np.asarray(cfg.get("class_score_scale", np.ones(self.fbcca.Nf)), dtype=float)
                if scale.shape[0] == self.fbcca.Nf:
                    self.class_score_scale = np.clip(scale, 0.5, 2.0)
                else:
                    self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                if self._is_supervised_model():
                    train_files = cfg.get("train_files", [])
                    ok, msg = self._fit_tdca_from_files(train_files)
                    if not ok:
                        self.weight_status.setText(f"当前权重: {self.model_name}未就绪({msg})")
            except Exception:
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)

    def _tdca_required_points(self):
        if not self._is_supervised_model():
            return int(self.training_sample_points)
        required_points = int(getattr(self.fbcca, "required_points", self.training_sample_points))
        return max(int(self.training_sample_points), required_points)

    def _fit_tdca_from_files(self, file_list):
        return self._training_framework().fit_tdca_from_files(file_list)

    def _ensure_tdca_ready(self):
        if not self._is_supervised_model():
            return True
        if getattr(self.fbcca, "is_fitted", False):
            return True
        self.decision_info.setText(f"识别状态: {self.model_name}未训练，请先训练权重或加载带训练文件的权重")
        self.show_label.setText(f"{self.model_name}未就绪：请先训练后再在线测试")
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
        if self.model_name not in ("TDCA", "IMPROVEDTDCA", "TRIBRANCHTDCA", "FBCCA", "CCA"):
            self.model_name = "TDCA"

        if getattr(self, "mode", "online") in ("online", "test") and self.start_flick:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("参数已变更，请按回车重新开始")
            self.decision_info.setText("识别状态: 已停止（模型已切换）")

        self._rebuild_classifier_with_adaptation()
        if getattr(self, "mode", "online") == "train":
            self._reset_training_plan_for_param_change()

        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if getattr(self, "mode", "online") in ("online", "test") and hasattr(self, "training_hint"):
            prefix = "测试模式" if getattr(self, "mode", "online") == "test" else "在线模式"
            self.training_hint.setText(f"{prefix}: 回车启动 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.online_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)")

    def _on_sampling_rate_changed(self, text):
        fs = 250
        self.sample_rate_hz = fs

        if getattr(self, "mode", "online") in ("online", "test") and self.start_flick:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("参数已变更，请按回车重新开始")
            self.decision_info.setText("识别状态: 已停止（采样率已切换）")

        self._rebuild_classifier_with_adaptation()
        if getattr(self, "mode", "online") == "train":
            self._reset_training_plan_for_param_change()

        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if getattr(self, "mode", "online") in ("online", "test") and hasattr(self, "training_hint"):
            prefix = "测试模式" if getattr(self, "mode", "online") == "test" else "在线模式"
            self.training_hint.setText(f"{prefix}: 回车启动 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.online_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)")

    def _on_online_time_changed(self, text):
        if getattr(self, "mode", "online") in ("online", "test") and self.start_flick:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("参数已变更，请按回车重新开始")
            self.decision_info.setText("识别状态: 已停止（测试时长已切换）")

        self._rebuild_classifier_with_adaptation()
        if getattr(self, "mode", "online") == "train":
            self._reset_training_plan_for_param_change()

        if hasattr(self, "freq_status"):
            self.freq_status.setText(self._online_status_text())
        if getattr(self, "mode", "online") in ("online", "test") and hasattr(self, "training_hint"):
            prefix = "测试模式" if getattr(self, "mode", "online") == "test" else "在线模式"
            self.training_hint.setText(f"{prefix}: 回车启动 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.online_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)")

    def _on_training_time_changed(self, text):
        self._rebuild_classifier_with_adaptation()
        self._reset_training_plan_for_param_change()
        if getattr(self, "mode", "online") == "train" and hasattr(self, "training_hint"):
            self.training_hint.setText(
                f"训练模式: 回车启动 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.training_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)"
            )

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
            self.show_label.setText("训练参数已更新，按回车开始")
            self.decision_info.setText("识别状态: 训练计划已重置")
            if hasattr(self, "training_hint"):
                self.training_hint.setText(
                    f"训练模式: 回车启动 trial(提示{self.trial_cue_sec:.1f}s→刺激{self.training_window_sec:.2f}s→间隔{self.trial_rest_sec:.1f}s)"
                )
        elif getattr(self, "mode", "online") == "test":
            self._reset_test_accuracy_stats(reset_plan=True)
            self.show_label.setText("测试标签已更新，按回车开始")
            self.decision_info.setText("识别状态: 测试计划已重置")
        if hasattr(self, "train_data_tree"):
            self.refresh_train_dataset_view()

    def _on_min_confidence_changed(self, value):
        try:
            self.min_confidence = float(value)
        except Exception:
            self.min_confidence = 0.02

    def _on_execution_mode_changed(self, index):
        self.execution_mode = "vote" if int(index) == 0 else "direct"

    def _reset_online_accuracy_stats(self):
        self.online_eval_total = 0
        self.online_eval_correct = 0
        if not hasattr(self, "online_acc_info"):
            return
        if self.online_eval_truth == "不统计":
            self.online_acc_info.setText("在线准确率: -")
        else:
            self.online_acc_info.setText(f"在线准确率: 0.00% (0/0) | 真值: {self.online_eval_truth}")

    def _on_online_truth_changed(self, text):
        self.online_eval_truth = str(text).strip() if str(text).strip() else "不统计"
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
            self.online_acc_info.setText(f"测试准确率: 0.00% (0/0) | 计划: {total}条")

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
                f"测试准确率: {acc:.2f}% ({self.test_eval_correct}/{self.test_eval_total}) | 当前: {truth}→{pred_text}"
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
        if self.online_eval_truth == "不统计":
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
            f"在线准确率: {acc:.2f}% ({self.online_eval_correct}/{self.online_eval_total}) | 真值: {self.online_eval_truth}"
        )

    def _cooldown_ok(self, idx):
        now = time.time()
        if (now - self.last_command_time) < self.command_cooldown_sec and idx == self.last_command_idx:
            self.last_gate_reason = "同指令冷却中"
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
        self.train_info.setText(f"训练进度: {self.training_collected}/{total}")
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

    def _training_framework(self):
        return CarTrainingFramework(
            subject=config.subjectName,
            commands=self.commands,
            sample_rate_hz=self.sample_rate_hz,
            model_name=self.model_name,
            classifier=self.fbcca,
            prepare_model_input=self._prepare_model_input,
            required_points_func=self._tdca_required_points if self._is_supervised_model() else lambda: self.training_sample_points,
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

    def _supervised_leave_one_out_scores(self, rows, model_times, model_name="TDCA"):
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
                name = str(model_name).upper()
                if name == "IMPROVEDTDCA":
                    model = ImprovedTDCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
                elif name == "TRIBRANCHTDCA":
                    model = TriBranchTDCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
                else:
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
            tdca_preds, tdca_margins = self._supervised_leave_one_out_scores(rows, model_times, "TDCA")
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
            self.weight_status.setText(f"当前权重: 正在分析 {len(files)} 条...")
        QApplication.processEvents()
        scored = self._score_current_bucket_rows(include_tdca=False, files=files, progress_label="train_check")
        if len(scored) == 0:
            self.weight_status.setText(f"当前权重: 当前{self.training_window_sec:.2f}s无可分析样本")
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
                    f"命中分 {row['accuracy_score']:.1f} | 置信分 {row['confidence_score']:.1f} | 总分: {row['total_score']:.1f}"
                ),
            )

        self.weight_status.setText(
            f"当前权重: 已按质量分析勾选 {selected}/{matched} 条，训练仍使用原始训练数据路径"
        )
        if hasattr(self, "score_status"):
            self.score_status.setText(f"数据分析: 已同步到训练数据表 {selected}/{matched} 条")
        print(
            f"[SCORE_PROGRESS] train_check done selected={selected}/{matched} "
            f"elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def _ssvep_signal_metrics(self, sample, label_idx):
        data = np.asarray(sample, dtype=float)
        if data.ndim != 2 or data.shape[-1] < 8:
            return {"rms": 0.0, "target_power": 0.0, "snr_db": -99.0, "peak_hz": 0.0}

        data = data - np.mean(data, axis=1, keepdims=True)
        rms = float(np.sqrt(np.mean(np.square(data))))
        fs = float(self.sample_rate_hz)
        nperseg = min(data.shape[-1], max(64, int(fs * 2)))
        try:
            freqs, psd = signal.welch(data, fs=fs, axis=-1, nperseg=nperseg)
            mean_psd = np.mean(psd, axis=0)
        except Exception:
            return {"rms": rms, "target_power": 0.0, "snr_db": -99.0, "peak_hz": 0.0}

        if 0 <= int(label_idx) < len(self.sti_lst):
            target_hz = float(self.sti_lst[int(label_idx)])
        else:
            target_hz = float(self.sti_lst[0])

        target_mask = np.abs(freqs - target_hz) <= 0.35
        noise_mask = (np.abs(freqs - target_hz) > 0.55) & (np.abs(freqs - target_hz) <= 2.5)
        if not np.any(target_mask):
            target_mask = np.abs(freqs - target_hz) <= 0.6
        target_power = float(np.mean(mean_psd[target_mask])) if np.any(target_mask) else 0.0
        noise_power = float(np.mean(mean_psd[noise_mask])) if np.any(noise_mask) else float(np.median(mean_psd) + 1e-12)
        snr_db = float(10.0 * np.log10((target_power + 1e-12) / (noise_power + 1e-12)))

        band_mask = (freqs >= 4.0) & (freqs <= 45.0)
        if np.any(band_mask):
            band_freqs = freqs[band_mask]
            band_psd = mean_psd[band_mask]
            peak_hz = float(band_freqs[int(np.argmax(band_psd))])
        else:
            peak_hz = 0.0

        return {
            "rms": rms,
            "target_power": target_power,
            "snr_db": snr_db,
            "peak_hz": peak_hz,
        }

    def refresh_score_dataset_view(self):
        if not hasattr(self, "score_tree"):
            return
        t0 = time.perf_counter()
        self.score_tree.setSortingEnabled(False)
        self.score_tree.clear()
        files = self._iter_checked_train_files()
        if len(files) == 0:
            self.score_status.setText("数据分析: 请先在训练数据页勾选要分析的样本")
            self.score_tree.setSortingEnabled(True)
            return
        rows = self._load_score_samples(files)
        print(f"[ANALYSIS_PROGRESS] analysis_tab requested files={len(files)} rows={len(rows)}", flush=True)
        if len(rows) == 0:
            self.score_status.setText(f"数据分析: 勾选样本中没有满足{self.training_window_sec:.2f}s时长的数据")
            self.score_tree.setSortingEnabled(True)
            print("[ANALYSIS_PROGRESS] analysis_tab no analyzable samples", flush=True)
            return

        model_times = float(self.analysis_window_sec + self.analysis_delay_sec)
        fbcca = FBCCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        cca = CCA(3, model_times, self.sti_lst, sample_rate=self.sample_rate_hz)
        print("[ANALYSIS_PROGRESS] analysis_tab tdca_leave_one_out start", flush=True)
        tdca_preds, tdca_margins = self._supervised_leave_one_out_scores(rows, model_times, "TDCA")
        print("[ANALYSIS_PROGRESS] analysis_tab tdca_leave_one_out done", flush=True)

        usable_count = 0
        snr_values = []
        rms_values = []
        for idx, row in enumerate(rows):
            sample = row["sample"]
            label_idx = int(row["label_idx"])
            fb_pred, fb_margin = self._score_model_once(fbcca, sample)
            cca_pred, cca_margin = self._score_model_once(cca, sample)
            tdca_pred = int(tdca_preds[idx])
            metrics = self._ssvep_signal_metrics(sample, label_idx)
            snr_values.append(float(metrics["snr_db"]))
            rms_values.append(float(metrics["rms"]))

            fb_hit = fb_pred == label_idx
            cca_hit = cca_pred == label_idx
            tdca_hit = tdca_pred == label_idx
            tdca_valid = 0 <= tdca_pred < len(self.commands)
            snr_ok = float(metrics["snr_db"]) >= float(self.score_threshold_spin.value())
            alg_votes = int(fb_hit) + int(cca_hit) + int(tdca_hit and tdca_valid)
            if snr_ok and alg_votes >= 2:
                conclusion = "频率清晰"
            elif snr_ok or alg_votes >= 2:
                conclusion = "可用"
            else:
                conclusion = "复查"
            if conclusion != "复查":
                usable_count += 1

            print(
                f"[ANALYSIS_PROGRESS] analysis_tab {idx + 1}/{len(rows)} "
                f"file={os.path.basename(row['fp'])} snr={metrics['snr_db']:.2f} "
                f"peak={metrics['peak_hz']:.2f} fb={'hit' if fb_hit else 'miss'} "
                f"cca={'hit' if cca_hit else 'miss'} tdca={'hit' if tdca_hit else 'miss'}",
                flush=True,
            )

            def name_of(pred):
                return self.commands[pred] if 0 <= int(pred) < len(self.commands) else "-"

            item = QTreeWidgetItem([
                os.path.basename(row["fp"]),
                self.commands[label_idx],
                str(row["points"]),
                f"{metrics['rms']:.3g}",
                f"{metrics['target_power']:.3g}",
                f"{metrics['snr_db']:.2f}",
                f"{metrics['peak_hz']:.2f}",
                name_of(fb_pred),
                name_of(cca_pred),
                name_of(tdca_pred),
                conclusion,
            ])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if conclusion != "复查" else Qt.Unchecked)
            item.setData(0, Qt.UserRole, row["fp"])
            item.setData(5, Qt.UserRole, float(metrics["snr_db"]))
            if conclusion != "复查":
                item.setForeground(10, QBrush(QColor(110, 231, 183)))
            else:
                item.setForeground(10, QBrush(QColor(252, 211, 77)))
            self.score_tree.addTopLevelItem(item)

        for col in range(self.score_tree.columnCount()):
            self.score_tree.resizeColumnToContents(col)
        self.score_tree.setSortingEnabled(True)
        mean_snr = float(np.mean(snr_values)) if len(snr_values) > 0 else 0.0
        mean_rms = float(np.mean(rms_values)) if len(rms_values) > 0 else 0.0
        self.score_status.setText(
            f"数据分析: 已分析 {len(rows)} 条 | 可用 {usable_count} 条 | 平均SNR {mean_snr:.2f}dB | 平均RMS {mean_rms:.3g} | 用时 {time.perf_counter() - t0:.2f}s"
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
            score = item.data(5, Qt.UserRole)
            try:
                score = float(score)
            except Exception:
                score = -99.0
            keep = item.text(10) != "复查" and score >= threshold
            item.setCheckState(0, Qt.Checked if keep else Qt.Unchecked)
            if keep:
                selected += 1
        self.score_status.setText(f"数据分析: 已按SNR阈值勾选 {selected}/{total} 条")

    def sync_score_checked_to_train_samples(self):
        checked_paths = set()
        for item in self._iter_score_items():
            fp = item.data(0, Qt.UserRole)
            if item.checkState(0) == Qt.Checked and isinstance(fp, str):
                checked_paths.add(os.path.normcase(os.path.abspath(fp)))
        if len(checked_paths) == 0:
            self.score_status.setText("数据分析: 没有勾选可同步样本")
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
        self.weight_status.setText(f"当前权重: 已从数据分析页同步勾选 {selected}/{matched} 条训练数据")
        self.score_status.setText(f"数据分析: 已同步 {selected}/{matched} 条到训练数据页")

    def save_checked_scored_samples(self):
        checked = []
        for item in self._iter_score_items():
            if item.checkState(0) == Qt.Checked:
                fp = item.data(0, Qt.UserRole)
                if isinstance(fp, str) and os.path.exists(fp):
                    checked.append((fp, item))
        if len(checked) == 0:
            self.score_status.setText("数据分析: 未勾选样本")
            return

        day_dir = datetime.now().strftime("%Y-%m-%d")
        bucket = f"{float(self.training_window_sec):.2f}s"
        out_dir = os.path.join(self._subject_root(), "curated", day_dir, bucket)
        os.makedirs(out_dir, exist_ok=True)
        copied = 0
        for fp, item in checked:
            score_text = item.text(5).replace(".", "p").replace("-", "m")
            base = os.path.basename(fp)
            dst = os.path.join(out_dir, f"snr_{score_text}_{base}")
            try:
                shutil.copy2(fp, dst)
                copied += 1
            except Exception:
                continue
        self.score_status.setText(f"数据分析: 已保存 {copied}/{len(checked)} 条到 {out_dir}")

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
        if hasattr(self, "quality_calc_timer") and self.quality_calc_timer.isActive():
            self.quality_calc_timer.stop()
        self.quality_calc_pending = []
        self.quality_calc_rows = []

    def _start_quality_calc(self):
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
                            stim_key = "未知时长"
                        saved_q = None
                        if 'instant_quality_score' in m:
                            saved_q = float(np.array(m.get('instant_quality_score')).reshape(-1)[0])
                        elif 'quality_score' in m:
                            saved_q = float(np.array(m.get('quality_score')).reshape(-1)[0])
                    except Exception:
                        label_text = "读取失败"
                        pts = "?"
                        stim_key = "未知时长"
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
                        label_text = "读取失败"
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
            f"当前权重: 已自动勾选{self.training_window_sec:.2f}s 全量样本 {selected}/{total}"
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
            self.weight_status.setText("当前权重: 未训练（未勾选数据）")
            return

        result = self._training_framework().train_weights(checked_files)
        if not result.ok:
            self.weight_status.setText("当前权重: " + result.message)
            if hasattr(self, "decision_info") and isinstance(result.point_hist, dict):
                top_pts = sorted(result.point_hist.items(), key=lambda x: x[1], reverse=True)
                if top_pts:
                    required_points = self._tdca_required_points() if self._is_supervised_model() else self.training_sample_points
                    self.decision_info.setText(
                        f"识别状态: 训练失败，当前要求采样点={required_points}，主流样本点={top_pts[0][0]}"
                    )
            return

        self.class_score_scale = np.asarray(result.class_score_scale, dtype=float)
        self.selected_weight_file = result.save_file
        self.weight_status.setText("当前权重: " + result.message)
        self.refresh_weight_file_list(select_file=result.save_file)

    def evaluate_checked_data(self):
        checked_files = self._iter_checked_train_files()
        if len(checked_files) == 0:
            QMessageBox.warning(self, "测评失败", "未勾选任何可用的数据样本。")
            return
        
        if hasattr(self, "weight_status"):
            self.weight_status.setText(f"当前权重: 正在评测 {len(checked_files)} 条勾选数据...")
        QApplication.processEvents()
        
        rows = self._load_score_samples(checked_files)
        if len(rows) == 0:
            QMessageBox.warning(self, "测评失败", "未能成功读取到满足要求的勾选数据样本。")
            if hasattr(self, "weight_status"):
                self.weight_status.setText("当前权重: 测评失败")
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
        tdca_preds, _ = self._supervised_leave_one_out_scores(rows, model_times, "TDCA")
        improved_preds, _ = self._supervised_leave_one_out_scores(rows, model_times, "IMPROVEDTDCA")
        tri_preds, _ = self._supervised_leave_one_out_scores(rows, model_times, "TRIBRANCHTDCA")
        
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
        improved_preds = np.array(improved_preds)
        tri_preds = np.array(tri_preds)

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
        improved_acc, improved_bacc, improved_f1, improved_ok = get_metrics(y_true, improved_preds)
        tri_acc, tri_bacc, tri_f1, tri_ok = get_metrics(y_true, tri_preds)
        
        dialog = MessageBoxBase(self)
        dialog.titleLabel = SubtitleLabel(f"基于 {len(rows)} 条勾选数据的测评结果", dialog)
        
        table = TableWidget(dialog)
        table.setColumnCount(4)
        table.setRowCount(5)
        table.setHorizontalHeaderLabels(["模型", "ACC (%)", "bACC (%)", "Macro F1 (%)"])
        
        models = ["FBCCA", "CCA", "TDCA (留一法)", "ImprovedTDCA (留一法)", "TriBranchTDCA (留一法)"]
        metrics = [
            (fb_acc, fb_bacc, fb_f1),
            (cca_acc, cca_bacc, cca_f1),
            (tdca_acc, tdca_bacc, tdca_f1),
            (improved_acc, improved_bacc, improved_f1),
            (tri_acc, tri_bacc, tri_f1),
        ]
        status_ok = [fb_ok, cca_ok, tdca_ok, improved_ok, tri_ok]
        
        def create_item(text):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            return item

        for row, (model, mets, ok) in enumerate(zip(models, metrics, status_ok)):
            table.setItem(row, 0, create_item(model))
            if row >= 2 and not ok:
                item = create_item("样本多样性不足 / 各类样本过少，无法交叉验证")
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
            no_sk_label = CaptionLabel("(提示: 环境未安装 scikit-learn，仅显示基本 ACC)")
            no_sk_label.setStyleSheet("color: #9FB0C9;")
            dialog.viewLayout.addWidget(no_sk_label)
        
        dialog.widget.setMinimumWidth(550)
        dialog.yesButton.setText("确定")
        dialog.cancelButton.hide()
        
        if hasattr(self, "weight_status"):
            self.weight_status.setText("当前权重: 测评完成")
            
        dialog.exec_()


    def refresh_weight_file_list(self, select_file=None):
        self.weight_combo.clear()
        self.weight_combo.addItem("默认(无)", "")

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
            if self._is_supervised_model() and hasattr(self.fbcca, "clear_fit"):
                self.fbcca.clear_fit()
            self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
            self.selected_weight_file = ""
            self.weight_status.setText("当前权重: 默认(无)")
            return

        try:
            with open(fp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_model = str(cfg.get("model_name", "")).upper()
            cfg_fs = int(cfg.get("sample_rate_hz", self.sample_rate_hz))
            cfg_freqs = cfg.get("stim_freqs_hz", None)
            if not self._freqs_compatible(cfg_freqs, allow_legacy_without_meta=True):
                self.fbcca.reset_frequency_weights()
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                self.selected_weight_file = ""
                self.weight_status.setText("当前权重: 刺激频率不匹配，请用当前屏幕刷新率重新采集/训练")
                return
            if cfg_model and cfg_model != self.model_name:
                self.fbcca.reset_frequency_weights()
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                self.selected_weight_file = ""
                self.weight_status.setText(f"当前权重: 模型不匹配({cfg_model} != {self.model_name})")
                return
            if cfg_fs != self.sample_rate_hz:
                self.fbcca.reset_frequency_weights()
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
                self.selected_weight_file = ""
                self.weight_status.setText(f"当前权重: 采样率不匹配({cfg_fs}Hz != {self.sample_rate_hz}Hz)")
                return
            weights = np.asarray(cfg.get("weights", []), dtype=float)
            self.fbcca.set_frequency_weights(weights)
            scale = np.asarray(cfg.get("class_score_scale", np.ones(self.fbcca.Nf)), dtype=float)
            if scale.shape[0] == self.fbcca.Nf:
                self.class_score_scale = np.clip(scale, 0.5, 2.0)
            else:
                self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
            if self._is_supervised_model():
                train_files = cfg.get("train_files", [])
                ok, msg = self._fit_tdca_from_files(train_files)
                if not ok:
                    self.selected_weight_file = ""
                    self.weight_status.setText(f"当前权重: {self.model_name}加载失败({msg})")
                    return
            self.selected_weight_file = fp
            self.weight_status.setText(f"当前权重: 已加载 {os.path.basename(fp)}")
        except Exception as e:
            self.class_score_scale = np.ones(self.fbcca.Nf, dtype=float)
            self.weight_status.setText(f"当前权重: 加载失败 {str(e)}")

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
        if self._is_supervised_model() and getattr(self.fbcca, "is_fitted", False):
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
            if self._is_supervised_model() and 0 <= int(fallback_label_idx) < len(self.commands):
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
            self.show_label.setText("训练已完成，请修改样本数/标签后继续")
            self.decision_info.setText("识别状态: 训练已完成")
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
        self.show_label.setText(f"训练提示: 准备注视 {self.training_target_label}")
        self.decision_info.setText("识别状态: 训练提示阶段")
        self.setDefaultColor()
        self.training_timer.start()

    def _training_tick(self):
        if not self.training_collecting:
            self.training_timer.stop()
            return

        if self.training_phase == "cue":
            elapsed = self._phase_elapsed(self.training_phase_start_ts)
            self.progress.setValue(int(min(1.0, elapsed / max(self.trial_cue_sec, 1e-6)) * 100))
            self.show_label.setText(f"训练提示: 请注视 {self.training_target_label}")
            if elapsed >= self.trial_cue_sec:
                self._enter_training_phase("stim")
                self.start_cache = True
                self.training_stim_frame_idx = 0
                self.training_last_render_frame_idx = -1
                self.training_stim_onset_mono = self._now_mono()
                self.training_stim_onset_unix = time.time()
                self.training_trial_meta["stim_onset_monotonic"] = float(self.training_stim_onset_mono)
                self.training_trial_meta["stim_onset_unix"] = float(self.training_stim_onset_unix)
                self.show_label.setText(f"训练刺激中: {self.training_target_label}")
                self.decision_info.setText("识别状态: 训练刺激阶段")
            return

        if self.training_phase == "stim":
            elapsed = self._phase_elapsed(self.training_phase_start_ts)
            self.progress.setValue(int(min(1.0, elapsed / max(self.training_window_sec, 1e-6)) * 100))
            if elapsed < self.training_window_sec:
                cur_frame = self._frame_index_from_onset(self.training_stim_onset_mono)
                if cur_frame != self.training_last_render_frame_idx:
                    self.training_stim_frame_idx = cur_frame
                    self.training_last_render_frame_idx = cur_frame
                    self._render_stim_by_frame(cur_frame)
                return
            if elapsed >= self.training_window_sec:
                self.setDefaultColor()
                self.training_pending_success = False
                self.training_trial_meta["stim_end_monotonic"] = float(self._now_mono())
                self.training_trial_meta["stim_end_unix"] = float(time.time())
                self._enter_training_phase("rest")
                self.show_label.setText(f"训练间隔: {self.training_target_label}")
                self.decision_info.setText("识别状态: 训练间隔阶段")
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

        self.training_trial_meta["trial_end_monotonic"] = float(self._now_mono())
        self.training_trial_meta["trial_end_unix"] = float(time.time())

        self.start_cache = False
        self.setDefaultColor()

        required_samples = int(self.training_trial_meta.get("expected_samples", self.training_sample_points))
        if not success:
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
            self.show_label.setText("训练采集超时：未收到足够EEG数据")
            self.decision_info.setText("识别状态: 训练采集超时")
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
            self.show_label.setText("训练采集失败：缓存数据异常")
            self.decision_info.setText("识别状态: 缓存数据异常")
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
            self.show_label.setText("训练采集失败：时序对齐窗口不足")
            self.decision_info.setText("识别状态: 时序对齐失败（请重试）")
            self._clear_cache_buffer()
            return
        model_data = self._prepare_model_input(used_data)
        if not isinstance(model_data, np.ndarray) or model_data.ndim != 2:
            self.progress.setValue(0)
            self.training_collecting = False
            self.training_phase = "idle"
            self.show_label.setText("训练采集失败：样本预处理异常")
            self.decision_info.setText("识别状态: 样本预处理失败")
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

        try:
            savemat(saveFile, {
                'data': model_data,
                'raw_data': used_data,
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
            self.show_label.setText("训练样本保存失败")
            self.decision_info.setText(f"识别状态: 保存失败 {str(e)}")
            return

        self.decision_info.setText(
            f"识别状态: 输入质量 ok={bool(input_ok)} | 实得/期望={actual_samples}/{expected_samples} | 有效Fs={effective_fs:.1f}Hz"
        )

        self.training_collected += 1
        self._update_training_progress()
        if self.training_collected >= len(self.training_plan):
            self.show_label.setText(f"训练完成: 已保存 {self.training_collected}/{len(self.training_plan)}")
            self.decision_info.setText("识别状态: 训练已完成")
            QTimer.singleShot(100, self.refresh_train_dataset_view)
        else:
            self.show_label.setText(
                f"训练已保存 {self.training_target_label} ({self.training_collected}/{len(self.training_plan)})"
            )
        self.training_collecting = False
        self.training_phase = "idle"
        self.training_target_label = ""

    def _vote_command(self, idx, confidence):
        if confidence < self.min_confidence:
            self.last_gate_reason = f"置信度不足({confidence:.3f}<{self.min_confidence:.3f})"
            return None

        self.decision_buffer.append(idx)
        self.current_votes = [self.commands[i] for i in self.decision_buffer]
        self._update_live_panel()
        if len(self.decision_buffer) < self.vote_threshold:
            self.last_gate_reason = f"投票样本不足({len(self.decision_buffer)}/{self.vote_threshold})"
            return None

        values, counts = np.unique(np.array(self.decision_buffer), return_counts=True)
        winner = int(values[np.argmax(counts)])
        agree = int(np.max(counts))
        if agree < self.vote_threshold:
            self.last_gate_reason = f"投票未达阈值({agree}/{self.vote_threshold})"
            return None

        if not self._cooldown_ok(winner):
            return None

        self.last_gate_reason = "投票通过"
        return winner

    def _send_robot_command_async(self, idx):
        th = threading.Thread(target=self._send_robot_command, args=(idx,), daemon=True)
        th.start()

    def _send_robot_command(self, idx):
        ip, port = self._get_robot_endpoint()
        try:
            client = RobotClient(ip, port)
            client.connect()
            if client.connected:
                self.robot_net_ok = True
                self._refresh_network_status()
                client.move(idx)
                print(f"Socket命令已发送: {self.commands[idx]}")
            else:
                self.robot_net_ok = False
                self._refresh_network_status()
                self.decision_info.setText(
                    f"识别状态: 控制连接失败 {ip}:{port}（拒绝连接）"
                )
                print("小车连接失败")
            client.close()
        except Exception as e:
            self.robot_net_ok = False
            self._refresh_network_status()
            self.decision_info.setText(
                f"识别状态: 控制发送失败 {ip}:{port}"
            )
            print(f"发送命令失败: {str(e)}")

    def start_sti_event(self):
        if not self.start_flick:
            if not self._ensure_tdca_ready():
                return
            if getattr(self, "mode", "online") == "test":
                self.test_eval_plan = self._build_training_plan()
                if len(self.test_eval_plan) == 0:
                    self.decision_info.setText("识别状态: 测试标签为空")
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
            if getattr(self, "mode", "online") == "test":
                self.show_label.setText(f"测试模式运行中（{self.online_window_sec:.2f}秒/次） - 再按回车键停止")
            else:
                self._reset_online_accuracy_stats()
                self.show_label.setText(f"在线模式运行中（{self.online_window_sec:.2f}秒/次） - 再按回车键停止")
            self._start_online_window()
        else:
            self._reset_collection_state()
            self.continuous_mode = False
            self.show_label.setText("已停止")
            self.decision_info.setText("识别状态: 已停止")

    def _start_online_window(self):
        if self.finish or not self.start_flick:
            return
        if getattr(self, "mode", "online") == "test":
            if self.test_eval_index >= len(getattr(self, "test_eval_plan", [])):
                acc = 100.0 * self.test_eval_correct / max(self.test_eval_total, 1)
                self._reset_collection_state()
                self.continuous_mode = False
                self.show_label.setText(f"测试完成: {acc:.2f}% ({self.test_eval_correct}/{self.test_eval_total})")
                self.decision_info.setText("识别状态: 测试完成")
                return
        self._clear_cache_buffer()
        self.start_cache = True
        self.online_trial_meta = {
            "expected_samples": int(round(self.online_window_sec * self.sample_rate_hz)),
            "analysis_samples": int(self.online_sample_points),
            "sample_rate_hz": int(self.sample_rate_hz),
        }
        self.online_window_active = True
        self.online_window_start_time = time.time()
        self.online_stim_frame_idx = 0
        self.online_last_render_frame_idx = -1
        self.online_stim_onset_mono = 0.0
        self.online_stim_onset_unix = 0.0
        self._enter_online_phase("cue")
        if getattr(self, "mode", "online") == "test":
            truth = self._current_test_truth()
            self.show_label.setText(f"测试提示: 请注视 {truth}（{self.test_eval_index + 1}/{len(self.test_eval_plan)}）")
            self.decision_info.setText("识别状态: 测试提示阶段")
        else:
            self.show_label.setText(f"在线提示: 请注视目标（{self.online_window_sec:.2f}s刺激）")
            self.decision_info.setText("识别状态: 在线提示阶段")
        self.setDefaultColor()
        if not self.online_timer.isActive():
            self.online_timer.start()

    def _online_tick(self):
        if self.finish or not self.start_flick:
            self.online_timer.stop()
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
                self.online_stim_onset_mono = self._now_mono()
                self.online_stim_onset_unix = time.time()
                self.online_trial_meta["stim_onset_monotonic"] = float(self.online_stim_onset_mono)
                self.online_trial_meta["stim_onset_unix"] = float(self.online_stim_onset_unix)
                if getattr(self, "mode", "online") == "test":
                    self.show_label.setText(f"测试刺激中: {self._current_test_truth()}（{self.online_window_sec:.2f}s）")
                    self.decision_info.setText("识别状态: 测试刺激阶段")
                else:
                    self.show_label.setText(f"在线刺激中（{self.online_window_sec:.2f}s）")
                    self.decision_info.setText("识别状态: 在线刺激阶段")
            return

        if self.online_phase == "stim":
            elapsed = self._phase_elapsed(self.online_phase_start_ts)
            self.progress.setValue(int(min(1.0, elapsed / max(self.online_window_sec, 1e-6)) * 100))
            if elapsed < self.online_window_sec:
                cur_frame = self._frame_index_from_onset(self.online_stim_onset_mono)
                if cur_frame != self.online_last_render_frame_idx:
                    self.online_stim_frame_idx = cur_frame
                    self.online_last_render_frame_idx = cur_frame
                    self._render_stim_by_frame(cur_frame)
                return

            self.setDefaultColor()
            self.progress.setValue(100)
            if "stim_end_monotonic" not in self.online_trial_meta:
                _ts_mono = float(self._now_mono())
                _ts_unix = float(time.time())
                self.online_trial_meta["stim_end_monotonic"] = _ts_mono
                self.online_trial_meta["stim_end_unix"] = _ts_unix
                self.online_trial_meta["rest_onset_monotonic"] = _ts_mono
                self.online_trial_meta["rest_onset_unix"] = _ts_unix
                self.online_trial_meta["trial_end_monotonic"] = _ts_mono + float(self.trial_rest_sec)
                self.online_trial_meta["trial_end_unix"] = _ts_unix + float(self.trial_rest_sec)

            full_data = self._materialize_cache_data()
            recv_start_for_wait = float(self.online_trial_meta.get("recv_start_monotonic", np.nan))
            aligned_ready = self._has_aligned_window_ready(
                full_data,
                recv_start_for_wait,
                float(self.online_stim_onset_mono),
                int(self.online_sample_points),
            )
            if not aligned_ready and elapsed < self.online_window_sec + self.acquisition_tail_sec:
                self.start_cache = True
                self.decision_info.setText("识别状态: 等待EEG尾包")
                return

            self.start_cache = False

            required_samples = int(self.online_trial_meta.get("expected_samples", self.online_sample_points))
            if not aligned_ready:
                self._debug_alignment_failure(
                    "online_precheck",
                    self._materialize_cache_data(),
                    float(self.online_trial_meta.get("recv_start_monotonic", np.nan)),
                    float(self.online_stim_onset_mono),
                    int(self.online_sample_points),
                    self.online_trial_meta,
                )
                self.decision_info.setText("识别状态: 在线样本不足，跳过本轮")
                self._enter_online_phase("rest")
                return

            raw_samples = int(self._cache_sample_count())
            full_data = self._materialize_cache_data()
            if not isinstance(full_data, np.ndarray) or full_data.ndim != 2 or full_data.shape[-1] < self.online_sample_points:
                self._debug_alignment_failure(
                    "online_buffer",
                    full_data,
                    float(self.online_trial_meta.get("recv_start_monotonic", np.nan)),
                    float(self.online_stim_onset_mono),
                    int(self.online_sample_points),
                    self.online_trial_meta,
                )
                self.decision_info.setText("识别状态: 在线缓存异常，跳过本轮")
                self._clear_cache_buffer()
                self._enter_online_phase("rest")
                return
            recv_start = float(self.online_trial_meta.get("recv_start_monotonic", np.nan))
            used_data = self._extract_aligned_window(
                full_data=full_data,
                recv_start_mono=recv_start,
                stim_onset_mono=float(self.online_stim_onset_mono),
                sample_points=int(self.online_sample_points),
            )
            if not isinstance(used_data, np.ndarray) or used_data.ndim != 2 or used_data.shape[-1] < self.online_sample_points:
                self._debug_alignment_failure(
                    "online",
                    full_data,
                    recv_start,
                    float(self.online_stim_onset_mono),
                    int(self.online_sample_points),
                    self.online_trial_meta,
                )
                self.decision_info.setText("识别状态: 在线时序对齐失败，跳过本轮")
                self._clear_cache_buffer()
                self._enter_online_phase("rest")
                return
            model_data = self._prepare_model_input(used_data)
            if not isinstance(model_data, np.ndarray) or model_data.ndim != 2:
                self.decision_info.setText("识别状态: 在线样本预处理失败，跳过本轮")
                self._clear_cache_buffer()
                self._enter_online_phase("rest")
                return

            expected_samples = int(self.online_trial_meta.get("expected_samples", self.online_sample_points))
            analysis_samples = int(used_data.shape[-1])
            cache_samples = int(raw_samples)
            # Quality statistics should be based on the aligned stimulus window,
            # not the full cache that also includes cue/rest samples.
            actual_samples = int(analysis_samples)
            stim_onset = float(self.online_trial_meta.get("stim_onset_monotonic", self.online_stim_onset_mono))
            stim_end = float(self.online_trial_meta.get("stim_end_monotonic", self._now_mono()))
            stim_dur = max(1e-6, stim_end - stim_onset)
            effective_fs = float(actual_samples / stim_dur)
            drop_ratio = max(0.0, float(expected_samples - actual_samples) / max(expected_samples, 1))
            recv_start = float(self.online_trial_meta.get("recv_start_monotonic", stim_onset))
            recv_end = float(self.online_trial_meta.get("recv_end_monotonic", stim_end))
            recv_dur = max(0.0, recv_end - recv_start)
            input_ok = int((actual_samples >= int(0.95 * expected_samples)) and (effective_fs >= 0.75 * self.sample_rate_hz))

            savePath = self._subject_root()
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            fileNums = len(glob(os.path.join(savePath, '*.mat')))
            saveFile = os.path.join(savePath, f'{fileNums + 1}.mat')
            align_idx = self._aligned_window_indices(
                recv_start_mono=recv_start,
                stim_onset_mono=float(stim_onset),
                sample_points=int(self.online_sample_points),
            )
            align_start = int(align_idx[0]) if isinstance(align_idx, tuple) else -1
            align_end = int(align_idx[1]) if isinstance(align_idx, tuple) else -1

            savemat(saveFile, {
                'data': model_data,
                'raw_data': used_data,
                'sample_rate_hz': int(self.sample_rate_hz),
                'model_name': self.model_name,
                'trial_cue_sec': float(self.trial_cue_sec),
                'trial_stim_sec': float(self.online_window_sec),
                'analysis_window_sec': float(self.analysis_window_sec),
                'analysis_delay_sec': float(self.analysis_delay_sec),
                'trial_rest_sec': float(self.trial_rest_sec),
                'cue_onset_monotonic': float(self.online_trial_meta.get('cue_onset_monotonic', 0.0)),
                'cue_onset_unix': float(self.online_trial_meta.get('cue_onset_unix', 0.0)),
                'stim_onset_monotonic': float(self.online_stim_onset_mono),
                'stim_onset_unix': float(self.online_stim_onset_unix),
                'stim_end_monotonic': float(self.online_trial_meta.get('stim_end_monotonic', 0.0)),
                'stim_end_unix': float(self.online_trial_meta.get('stim_end_unix', 0.0)),
                'rest_onset_monotonic': float(self.online_trial_meta.get('rest_onset_monotonic', 0.0)),
                'rest_onset_unix': float(self.online_trial_meta.get('rest_onset_unix', 0.0)),
                'trial_end_monotonic': float(self.online_trial_meta.get('trial_end_monotonic', 0.0)),
                'trial_end_unix': float(self.online_trial_meta.get('trial_end_unix', 0.0)),
                'expected_samples': int(expected_samples),
                'actual_samples': int(actual_samples),
                'cache_samples': int(cache_samples),
                'analysis_samples': int(analysis_samples),
                'drop_ratio': float(drop_ratio),
                'effective_sample_rate_hz': float(effective_fs),
                'recv_start_monotonic': float(self.online_trial_meta.get('recv_start_monotonic', 0.0)),
                'recv_end_monotonic': float(self.online_trial_meta.get('recv_end_monotonic', 0.0)),
                'recv_duration_sec': float(recv_dur),
                'aligned_start_idx': int(align_start),
                'aligned_end_idx': int(align_end),
                'recv_chunks': int(self.online_trial_meta.get('recv_chunks', 0)),
                'recv_points': int(self.online_trial_meta.get('recv_points', 0)),
                'input_quality_ok': int(input_ok),
                'online_window_sec': float(self.online_window_sec),
                'stim_freqs_hz': np.array(self.sti_lst, dtype=float),
                'stim_frame_periods': np.array(self.stim_period_frames, dtype=int),
                'stim_duty_frames': np.array(self.stim_duty_frames, dtype=int),
                'display_refresh_hz': float(self.stim_refresh_hz),
                'selected_weight_file': os.path.basename(str(getattr(self, 'selected_weight_file', ''))),
            })

            self._clear_cache_buffer()
            _, scores, _ = self.fbcca.classify_with_scores(model_data)
            balanced_scores = self._apply_class_score_balance(scores)
            if getattr(self, "mode", "online") == "test":
                online_scores = np.asarray(balanced_scores, dtype=float)
            else:
                online_scores = self._normalize_online_scores(balanced_scores)
            result = int(np.argmax(online_scores))
            confidence = self._confidence_from_scores(online_scores)

            if getattr(self, "mode", "online") != "test" and self.online_score_warmup < self.online_warmup_windows:
                self.current_candidate = "-"
                self.current_confidence = 0.0
                self.current_votes = []
                self._update_live_panel()
                self.decision_info.setText(
                    f"识别状态: 在线基线预热 {self.online_score_warmup}/{self.online_warmup_windows}"
                )
                self._enter_online_phase("rest")
                return

            self.current_candidate = self.commands[result]
            self.current_confidence = float(confidence)
            self._update_live_panel()

            if getattr(self, "mode", "online") == "test":
                truth = self._current_test_truth()
                hit, acc = self._update_test_accuracy(result, confidence)
                self.decision_info.setText(
                    f"识别状态: 测试 {truth}→{self.commands[result]} | {'正确' if hit else '错误'} | 当前准确率 {acc:.2f}%"
                )
            else:
                self._update_online_accuracy(result)
                self.decision_info.setText(
                    f"识别状态: 候选 {self.commands[result]} | 置信度 {confidence:.4f}"
                )

                stable_idx = self._vote_command(result, confidence)
                if stable_idx is not None:
                    self.set_result(stable_idx, confidence)
                elif self.execution_mode == "direct":
                    if confidence >= self.min_confidence and self._cooldown_ok(result):
                        self.last_gate_reason = "直发通过"
                        self.set_result(result, confidence)
                    else:
                        self.decision_info.setText(
                            f"识别状态: 候选 {self.commands[result]} | 置信度 {confidence:.4f} | 未执行: {self.last_gate_reason}"
                        )
                else:
                    self.decision_info.setText(
                        f"识别状态: 候选 {self.commands[result]} | 置信度 {confidence:.4f} | 未执行: {self.last_gate_reason}"
                    )

            self._enter_online_phase("rest")
            if getattr(self, "mode", "online") != "test":
                self.decision_info.setText(
                    f"识别状态: 输入质量 ok={bool(input_ok)} | 实得/期望={actual_samples}/{expected_samples} | 有效Fs={effective_fs:.1f}Hz"
                )
            return

        if self.online_phase == "rest":
            elapsed = self._phase_elapsed(self.online_phase_start_ts)
            self.progress.setValue(int(max(0.0, 100.0 * (1.0 - min(1.0, elapsed / max(self.trial_rest_sec, 1e-6))))))
            if elapsed >= self.trial_rest_sec:
                self.online_trial_meta["trial_end_monotonic"] = float(self._now_mono())
                self.online_trial_meta["trial_end_unix"] = float(time.time())
                self._start_online_window()
            return

    def set_result(self, idx, confidence=None):
        """
        处理识别结果。

        参数:
            idx: 命令索引，范围 0-4。
                0: 前进 (6.67Hz)
                1: 后退 (7.5Hz)
                2: 左转 (8.57Hz)
                3: 停止 (12.0Hz)
                4: 右转 (15.0Hz)
        """
        if idx < len(self.commands):
            command = self.commands[idx]
            self.show_label.setText(f"执行命令: {command}")
            if confidence is None:
                self.decision_info.setText(f"识别状态: 已执行 {command}")
            else:
                self.decision_info.setText(f"识别状态: 已执行 {command} | 置信度 {confidence:.4f}")
            self.current_candidate = command
            if confidence is not None:
                self.current_confidence = float(confidence)
            self._update_live_panel()

            speak_async_safe(command)
            
            print(f"=" * 50)
            print(f"识别结果: {command}")
            print(f"命令索引: {idx}")
            print(f"刺激频率: {self.sti_lst[idx]} Hz")
            print(f"=" * 50)
            self._send_robot_command_async(idx)
            
        else:
            print(f"识别结果索引超出范围: {idx}")

    def getData(self, data):
        if self.finish and not self.training_collecting:
            return

        if self.start_cache:
            try:
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
        if self.video_panel is not None:
            try:
                self.video_panel.close()
            except Exception:
                pass
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
    commands = {0: "前进", 1: "后退", 2: "左转", 3: "停止", 4: "右转"}
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
