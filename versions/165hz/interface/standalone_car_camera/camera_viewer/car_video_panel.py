import pickle
import socket
import struct
import threading

import cv2
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QFrame, QSizePolicy

from start_car_camera_ssh import ensure_camera_tunnel_started

ROBOT_IP = "10.186.179.92"
CAMERA_ENDPOINTS = [
    (ROBOT_IP, 5000),
    ("127.0.0.1", 5001),
]
CAMERA_FLIP_CODE = 1


class CameraStreamWorker(QThread):
    frameReady = pyqtSignal(QImage)
    statusChanged = pyqtSignal(str)

    def __init__(self, endpoints, flip_code, parent=None):
        super().__init__(parent)
        self.endpoints = endpoints
        self.flip_code = flip_code
        self.payload_size = struct.calcsize("Q")
        self._running = True
        self._socket = None

    def stop(self):
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self.wait(1000)

    def _connect_socket(self):
        for host, port in self.endpoints:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(2.0)
                sock.connect((host, port))
                sock.settimeout(5.0)
                self.statusChanged.emit(f"Connected stream: {host}:{port}")
                return sock
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
        return None

    def run(self):
        while self._running:
            self._socket = self._connect_socket()
            if self._socket is None:
                self.statusChanged.emit(
                    "Waiting camera stream...\n"
                    "SSH tunnel auto-start tried.\n"
                    "Enter robot password if prompted."
                )
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

                    packed_msg_size = buffer[: self.payload_size]
                    buffer = buffer[self.payload_size :]
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

                    obj = pickle.loads(frame_data)
                    if hasattr(obj, "shape") and (len(obj.shape) == 1 or (len(obj.shape) == 2 and 1 in obj.shape)):
                        frame = cv2.imdecode(obj, cv2.IMREAD_COLOR)
                    else:
                        frame = obj

                    frame = cv2.flip(frame, self.flip_code)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
                    self.frameReady.emit(image)

            except Exception as e:
                if self._running:
                    self.statusChanged.emit(f"Stream disconnected: {e}\nReconnecting...")
                    self.msleep(1500)
            finally:
                if self._socket:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None


class CarVideoPanel(QWidget):
    def __init__(self, width=640, height=480, parent=None):
        super().__init__(parent)
        self._init_ui(width, height)
        self._init_tunnel_retry()
        self._init_camera()

    def _init_ui(self, width, height):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.frame = QFrame()
        self.frame.setStyleSheet(
            "QFrame {border: 1px solid #00E5FF; border-radius: 4px; background-color: #000000;}"
        )
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(1, 1, 1, 1)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(max(480, int(width * 0.75)), max(320, int(height * 0.9)))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; color: #404040;")
        self.video_label.setText("NO SIGNAL")
        self.video_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.last_image = None

        frame_layout.addWidget(self.video_label)
        layout.addWidget(self.frame)

    def _init_tunnel_retry(self):
        pass

    def _start_tunnel_async(self):
        pass

    def _init_camera(self):
        self.camera_worker = CameraStreamWorker(CAMERA_ENDPOINTS, CAMERA_FLIP_CODE, self)
        self.camera_worker.statusChanged.connect(self.video_label.setText)
        self.camera_worker.frameReady.connect(self._update_camera_frame)
        self.camera_worker.start()

    def _update_camera_frame(self, image):
        self.last_image = image
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self.last_image is None:
            return
        pixmap = QPixmap.fromImage(self.last_image).scaled(
            max(1, self.video_label.width()),
            max(1, self.video_label.height()),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def close(self):
        if hasattr(self, "camera_worker") and self.camera_worker is not None:
            self.camera_worker.stop()
        super().close()
