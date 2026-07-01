"""BrainVision Recorder LSL 数据接收器。

替代 NdDevice，通过 pylsl 从 BrainVision Recorder 的 LSL 流中
读取 32 导 EEG 数据，选取指定通道，下采样到目标采样率。
"""

import time
from collections import deque

import numpy as np
from scipy import signal

try:
    import pylsl
except ImportError:
    pylsl = None


class LslReceiver:
    """从 BrainVision Recorder 的 LSL 流中拉取 EEG 数据。

    对外接口兼容 NdDevice 的使用方式：
        receiver = LslReceiver(stream_name, stream_type, selected_channels)
        receiver.start()                        # no-op，与 NdDevice API 兼容
        data = receiver.read_latest_eeg_data()  # (N_ch, N_samples) 或 None
        receiver.close()
    """

    def __init__(
        self,
        stream_name="BrainVision",
        stream_type="EEG",
        selected_channels=None,
        target_sample_rate=250,
    ):
        if pylsl is None:
            raise ImportError(
                "pylsl 未安装。请运行: pip install pylsl"
            )

        self._stream_name = stream_name
        self._stream_type = stream_type
        self._selected_channels = (
            list(selected_channels)
            if selected_channels is not None
            else list(range(22, 31))  # 9 个默认枕叶通道
        )
        self._target_fs = int(target_sample_rate)
        self._source_fs = None
        self._last_read_ts = 0.0  # LSL timestamp of the last sample returned
        self._inlet = None
        self._running = False
        self._n_total_channels = None

        # 环形缓冲区: (samples_2d, timestamps_1d) 元组，最多保留 30 秒
        self._ring = deque(maxlen=300)

        self._resolve()

    # ------------------------------------------------------------------
    # 与 NdDevice 兼容的公开接口
    # ------------------------------------------------------------------

    def start(self):
        """启动接收（LSL stream 在 __init__ 已连接，这里做 no-op）。"""
        self._running = True

    def read_latest_eeg_data(self, target_freq=250):
        """读取自上次调用以来的全部新 EEG 数据。

        Returns
        -------
        np.ndarray, shape (N_selected, N_samples), or None
        """
        if not self._running or self._inlet is None:
            return None

        # 1. 从 LSL 拉取所有可用数据块
        self._pull_all()

        if len(self._ring) == 0:
            return None

        # 2. 丢弃已读过的旧样本
        new_chunks = []
        latest_ts = self._last_read_ts
        for samples, timestamps in self._ring:
            # timestamps 末尾的时间戳为这一块的最新时间
            chunk_last_ts = float(timestamps[-1])
            if chunk_last_ts > self._last_read_ts + 1e-9:
                # 找出本块中 new 部分的起始索引
                start_idx = 0
                if len(timestamps) > 0:
                    # 二分或线性扫描首个 > last_read_ts 的样本
                    for i, t in enumerate(timestamps):
                        if t > self._last_read_ts + 1e-9:
                            start_idx = i
                            break
                new_part = samples[:, start_idx:]
                if new_part.shape[1] > 0:
                    new_chunks.append(new_part)
                if chunk_last_ts > latest_ts:
                    latest_ts = chunk_last_ts

        if len(new_chunks) == 0:
            return None

        # 3. 拼接并选通道
        data = (
            np.concatenate(new_chunks, axis=1)
            if len(new_chunks) > 1
            else new_chunks[0]
        )

        if self._selected_channels:
            n_total = data.shape[0]
            valid = [i for i in self._selected_channels if 0 <= i < n_total]
            if len(valid) != len(self._selected_channels):
                print(
                    f"LslReceiver: 部分通道索引超出范围 (共 {n_total} 导)，"
                    f"实际选中 {len(valid)} 个"
                )
            data = data[valid, :] if valid else data

        # 4. 下采样到目标频率
        data = self._downsample(data)

        # 5. 更新读取时间戳
        self._last_read_ts = latest_ts

        return np.asarray(data, dtype=float)

    def close(self):
        """关闭 LSL 流连接（线程安全：先置标志再关 inlet）。"""
        self._running = False
        self._ring.clear()
        # 原子交换，防止后台线程在 close_stream 期间继续读写 inlet
        inlet = self._inlet
        self._inlet = None
        if inlet is not None:
            try:
                inlet.close_stream()
            except Exception:
                pass

    @property
    def is_connected(self):
        return self._inlet is not None and self._running

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve(self):
        """解析并连接 LSL 流。"""
        print(
            f"LslReceiver: 正在解析 LSL 流 name='{self._stream_name}' "
            f"type='{self._stream_type}' ..."
        )
        streams = pylsl.resolve_stream(
            "name", self._stream_name, "type", self._stream_type, timeout=3.0
        )
        if not streams:
            raise RuntimeError(
                f"未找到 LSL 流: name='{self._stream_name}', "
                f"type='{self._stream_type}'。\n"
                f"请确认 BrainVision Recorder 已启动且 LSL 输出已开启。"
            )

        stream_info = streams[0]
        self._inlet = pylsl.StreamInlet(stream_info, max_buflen=360)

        # 读取流信息
        info = stream_info
        self._source_fs = float(info.nominal_srate())
        self._n_total_channels = int(info.channel_count())
        print(
            f"LslReceiver: 已连接 → {self._n_total_channels} 导, "
            f"{self._source_fs:.0f} Hz, "
            f"选中 {len(self._selected_channels)} 导"
        )

        # 初始化读取时间戳，避免回读启动前的旧数据
        self._last_read_ts = pylsl.local_clock()

    def _pull_all(self):
        """拉取 LSL inlet 中当前可用的所有数据块。"""
        if self._inlet is None or not self._running:
            return
        try:
            while True:
                samples, timestamps = self._inlet.pull_chunk(
                    timeout=0.0, max_samples=8192
                )
                if samples is None or len(samples) == 0:
                    break
                # samples: list of lists; timestamps: list of floats
                arr = np.array(samples, dtype=float).T  # (n_ch, n_samples)
                ts_arr = np.array(timestamps, dtype=float)
                self._ring.append((arr, ts_arr))
        except Exception as e:
            print(f"LslReceiver: pull_chunk 异常: {e}")

    def _downsample(self, data):
        """将 data 下采样到 self._target_fs。

        data: (n_ch, n_samples) 在 self._source_fs 采样率下。
        """
        if self._source_fs is None or data.shape[1] <= 1:
            return data

        src = self._source_fs
        dst = self._target_fs

        if abs(src - dst) < 0.5:
            return data

        ratio = dst / src
        if ratio > 1.0:
            # 上采样（一般不发生，保留原样）
            return data

        # 整数倍下采样：直接切片（高效，零延迟）
        if src / dst == int(src / dst):
            step = int(round(src / dst))
            return data[:, ::step]

        # 非整数倍：用 polyphase 重采样
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
            # 退化：简单切片
            step = max(1, int(round(src / dst)))
            return data[:, ::step]
