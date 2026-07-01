"""BrainVision Recorder RDA 数据接收器。

通过 Recorder 内置的 RDA (Remote Data Access) TCP 服务
直接读取 32 导 EEG 数据，无需 LSL 桥接。
"""

import socket
import struct
import time
import threading
from collections import deque

import numpy as np
from scipy import signal


class RdaReceiver:
    """通过 TCP 连接 BrainVision Recorder 的 RDA 端口 (默认 51234)。

    对外接口兼容 NdDevice / LslReceiver：
        receiver = RdaReceiver(selected_channels=list(range(32)))
        receiver.start()
        data = receiver.read_latest_eeg_data()  # (N_ch, N_samples) 或 None
        receiver.close()
    """

    def __init__(
        self,
        host="127.0.0.1",
        port=51234,
        selected_channels=None,
        target_sample_rate=250,
    ):
        self._host = host
        self._port = port
        self._selected_channels = (
            list(selected_channels)
            if selected_channels is not None
            else list(range(32))
        )
        self._target_fs = int(target_sample_rate)
        self._source_fs = None
        self._n_total_channels = None

        self._sock = None
        self._running = False
        self._recv_thread = None

        # 环形缓冲区: (samples_2d_float32) 元组
        self._ring = deque(maxlen=600)
        self._ring_lock = threading.Lock()
        self._last_returned_index = 0
        self._total_chunks_received = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def start(self):
        """连接 RDA 端口并启动后台接收线程。"""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((self._host, self._port))
        self._sock.settimeout(None)  # 后续 recv 不超时

        print(
            f"RdaReceiver: 已连接 {self._host}:{self._port}"
        )

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        # 等待收到第一块数据以获取通道数和采样率
        waited = 0.0
        while self._source_fs is None and waited < 5.0:
            time.sleep(0.1)
            waited += 0.1

        if self._source_fs is None:
            self.close()
            raise RuntimeError("RDA: 5 秒内未收到数据，请确认 Recorder 正在采集")

        print(
            f"RdaReceiver: {self._n_total_channels} 导, "
            f"{self._source_fs:.0f} Hz → {self._target_fs} Hz, "
            f"选中 {len(self._selected_channels)} 导"
        )

    def read_latest_eeg_data(self, target_freq=250):
        """读取自上次调用以来的全部新 EEG 数据。

        Returns
        -------
        np.ndarray, shape (N_selected, N_samples), or None
        """
        if not self._running:
            return None

        with self._ring_lock:
            total = len(self._ring)
            if total == 0:
                return None
            new_start = self._last_returned_index
            if new_start >= total:
                return None
            chunks = list(self._ring)[new_start:]
            self._last_returned_index = total

        if len(chunks) == 0:
            return None

        # 拼接
        data = np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]

        # 选通道
        if self._selected_channels:
            n_total = data.shape[0]
            valid = [i for i in self._selected_channels if 0 <= i < n_total]
            data = data[valid, :]

        # 下采样
        data = self._downsample(data)

        return np.asarray(data, dtype=float)

    def close(self):
        """断开 RDA 连接。"""
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        with self._ring_lock:
            self._ring.clear()
            self._last_returned_index = 0

    @property
    def is_connected(self):
        return self._sock is not None and self._running

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _recv_loop(self):
        """后台线程：阻塞读取 RDA 数据块，放入环形缓冲区。"""
        buf = b""
        header_fmt = "<iiidd"  # nSize, nChannels, nSamples, dSamplingInterval, dReserved
        header_len = struct.calcsize(header_fmt)

        while self._running and self._sock is not None:
            try:
                # 确保收到完整头部
                while len(buf) < header_len:
                    chunk = self._sock.recv(header_len - len(buf) + 4096)
                    if not chunk:
                        raise ConnectionError("RDA 连接已断开")
                    buf += chunk

                # 解析头部
                n_size, n_channels, n_samples, d_sampling_interval, _ = struct.unpack(
                    header_fmt, buf[:header_len]
                )
                n_size = int(n_size)
                n_channels = int(n_channels)
                n_samples = int(n_samples)

                if self._source_fs is None and d_sampling_interval > 0:
                    self._source_fs = 1e6 / d_sampling_interval
                    self._n_total_channels = n_channels

                # 确保收到完整数据体
                data_bytes = n_channels * n_samples * 4
                while len(buf) < header_len + data_bytes:
                    chunk = self._sock.recv(
                        header_len + data_bytes - len(buf) + 4096
                    )
                    if not chunk:
                        raise ConnectionError("RDA 连接已断开")
                    buf += chunk

                # 解析数据：float32, multiplexed
                raw = np.frombuffer(
                    buf[header_len : header_len + data_bytes], dtype=np.float32
                )
                # reshape: (n_channels, n_samples) — 数据是 interleaved 格式
                data = raw.reshape(n_samples, n_channels).T.astype(np.float32)

                with self._ring_lock:
                    self._ring.append(data)
                    self._total_chunks_received += 1

                # 丢弃已消费的头部+数据
                buf = buf[header_len + data_bytes:]

            except (ConnectionError, OSError, struct.error) as e:
                if self._running:
                    print(f"RdaReceiver: recv 异常: {e}")
                break
            except Exception as e:
                print(f"RdaReceiver: 未预期的错误: {e}")
                break

        self._running = False
        print("RdaReceiver: 接收线程已退出")

    def _downsample(self, data):
        """将 data 下采样到 self._target_fs。"""
        if self._source_fs is None or data.shape[1] <= 1:
            return data

        src = self._source_fs
        dst = self._target_fs

        if abs(src - dst) < 0.5:
            return data

        ratio = dst / src
        if ratio > 1.0:
            return data

        if src / dst == int(src / dst):
            step = int(round(src / dst))
            return data[:, ::step]

        try:
            from fractions import Fraction

            frac = Fraction(dst, src).limit_denominator(100)
            up = frac.numerator
            down = frac.denominator
            n_out = int(data.shape[1] * up / down)
            if n_out < 1:
                return data
            return signal.resample_poly(data, up, down, axis=1)[:, :n_out]
        except Exception:
            step = max(1, int(round(src / dst)))
            return data[:, ::step]
