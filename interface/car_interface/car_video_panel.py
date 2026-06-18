import pickle
import socket
import struct
import threading
import select
import sys
import os

# Ensure the root directory is in sys.path so we can import start_car_camera_ssh
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) # Two levels up from car_interface -> interface -> Visual_BCI_App
if root_dir not in sys.path:
    sys.path.append(root_dir)

import cv2
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QFrame

from start_car_camera_ssh import ensure_camera_tunnel_started


ROBOT_IP = '192.168.1.11'
# 优先尝试直连，减少 SSH 隧道转发 (TCP over TCP) 带来的严重延迟
# 如果直连失败（如防火墙阻挡），再尝试走 SSH 隧道
CAMERA_ENDPOINTS = [
    (ROBOT_IP, 5000),       # Direct
    ('127.0.0.1', 5001),    # SSH Tunnel
]
CAMERA_FLIP_CODE = 1


class CameraStreamWorker(QThread):
    frameReady = pyqtSignal(QImage)
    statusChanged = pyqtSignal(str)

    def __init__(self, endpoints, flip_code, parent=None):
        super().__init__(parent)
        self.endpoints = endpoints
        self.flip_code = flip_code
        self.payload_size = struct.calcsize('Q')
        self._running = True
        self._socket = None
        self._skipped_frames = 0 # 丢帧计数器

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
                # Disable Nagle's algorithm for lower latency
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                sock.settimeout(2.0)
                sock.connect((host, port))
                sock.settimeout(5.0)
                self.statusChanged.emit(f'已连接视频流: {host}:{port}')
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
                self.statusChanged.emit('正在等待小车视频流...\n程序已自动尝试启动 SSH 隧道\n若弹出密码窗口请输入小车密码')
                self.msleep(2000)
                continue

            buffer = b''
            try:
                while self._running:
                    # 1. 读取数据包长度 (8 bytes)
                    while len(buffer) < self.payload_size and self._running:
                        chunk = self._socket.recv(self.payload_size - len(buffer))
                        if not chunk:
                            raise ConnectionError('socket closed')
                        buffer += chunk
                    
                    if not self._running:
                        break
                        
                    packed_msg_size = buffer[:self.payload_size]
                    buffer = buffer[self.payload_size:]
                    msg_size = struct.unpack('Q', packed_msg_size)[0]
                    
                    # 2. 读取完整的帧数据
                    while len(buffer) < msg_size and self._running:
                        needed = msg_size - len(buffer)
                        chunk_size = min(needed, 65536)
                        packet = self._socket.recv(chunk_size)
                        if not packet:
                            raise ConnectionError('socket closed')
                        buffer += packet

                    if not self._running:
                        break

                    frame_data = buffer[:msg_size]
                    buffer = buffer[msg_size:]
                    
                    # 3. 帧解码与显示
                    try:
                        # 恢复全速解码，移除所有丢帧逻辑，解决画面卡死问题
                        # 兼容处理
                        obj = pickle.loads(frame_data)
                        if hasattr(obj, 'shape') and (len(obj.shape) == 1 or (len(obj.shape) == 2 and 1 in obj.shape)):
                            # Compressed stream (JPEG)
                            frame = cv2.imdecode(obj, cv2.IMREAD_COLOR)
                        else:
                            # Raw stream
                            frame = obj
                        
                        frame = cv2.flip(frame, self.flip_code)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame.shape
                        bytes_per_line = ch * w
                        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                        
                        # 发送信号更新UI
                        self.frameReady.emit(image)
                        
                    except Exception as e:
                        print(f"Frame decode error: {e}")

            except Exception as e:
                if self._running:
                    self.statusChanged.emit(f'视频流已断开: {e}\n正在自动重连...')
                    self.msleep(1500)
            finally:
                if self._socket:
                    try:
                        self._socket.close()
                    except:
                        pass
                    self._socket = None


class CarVideoPanel(QWidget):
    def __init__(self, width=480, height=360, parent=None):
        super().__init__(parent)
        self._init_ui(width, height)
        self._init_tunnel_retry()
        self._init_camera()

    def _init_ui(self, width, height):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10) # 增加内边距
        layout.setSpacing(0)
        
        # 装饰性外框 - 极简科技风
        self.frame = QFrame()
        self.frame.setStyleSheet("""
            QFrame {
                border: 1px solid #00E5FF; 
                border-radius: 4px;
                background-color: #000000;
            }
        """)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(1, 1, 1, 1)

        self.video_label = QLabel()
        self.video_label.setFixedSize(width, height)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('background-color: #000000; color: #404040; border-radius: 0px;')
        self.video_label.setText('NO SIGNAL')
        self.video_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
        
        frame_layout.addWidget(self.video_label)
        layout.addWidget(self.frame)

        # 添加底部状态条 (极简文字装饰)
        self.status_bar = QLabel("LINK / ACTIVE   LATENCY / LOW   MODE / JPEG")
        self.status_bar.setStyleSheet("color: #00E5FF; font-family: 'Segoe UI', Consolas; font-size: 9px; margin-top: 5px; opacity: 0.8;")
        self.status_bar.setAlignment(Qt.AlignRight)
        layout.addWidget(self.status_bar)

    def _init_tunnel_retry(self):
        self.tunnel_retry_timer = QTimer(self)
        self.tunnel_retry_timer.setInterval(5000)
        self.tunnel_retry_timer.timeout.connect(self._start_tunnel_async)
        QTimer.singleShot(0, self._start_tunnel_async)
        self.tunnel_retry_timer.start()

    def _start_tunnel_async(self):
        def runner():
            try:
                ensure_camera_tunnel_started()
            except Exception:
                pass

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

    def _init_camera(self):
        self.camera_worker = CameraStreamWorker(CAMERA_ENDPOINTS, CAMERA_FLIP_CODE, self)
        self.camera_worker.statusChanged.connect(self._handle_status)
        self.camera_worker.frameReady.connect(self._update_camera_frame)
        self.camera_worker.start()

    def _handle_status(self, text):
        self.video_label.setText(text)

    def _update_camera_frame(self, image):
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def close(self):
        if hasattr(self, 'camera_worker') and self.camera_worker is not None:
            self.camera_worker.stop()
        super().close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CarVideoPanel()
    window.show()
    sys.exit(app.exec_())
