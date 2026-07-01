import numpy as np
import math
from scipy import signal
from PyQt5.QtCore import *

class FBCCA(QObject):
    """
    滤波器组典型相关分析(Filter Bank Canonical Correlation Analysis)类
    用于SSVEP(稳态视觉诱发电位)信号的频率识别
    """
    # 定义信号，用于发送识别结果
    sendResultSignal = pyqtSignal(object)
    
    def __init__(self, num_harmonics, times, targets, Nh=8, sample_rate=250, delay_sec=0.14, phases=None):
        """
        初始化FBCCA算法

        参数:
            num_harmonics: 谐波数量，用于参考信号生成
            times: 时间窗口长度(秒)，包含视觉延迟
            targets: 目标频率列表 (Hz)
            Nh: 通道数量，默认为8
            sample_rate: 采样率(Hz)
            delay_sec: 视觉延迟(秒)，默认0.14s
            phases: 目标相位列表 (弧度), 可选。
                    传入后启用频率-相位联合编码参考信号
        """
        super(FBCCA, self).__init__()
        self.Nh = Nh                  # 通道数量
        self.Fs = int(sample_rate)    # 采样率(Hz)
        self.targets = [float(x) for x in targets]
        self.Nf = len(self.targets)        # 目标频率数量
        self.delay_sec = float(delay_sec)
        self.ws = max(times - delay_sec, 0.2)
        self.Nc = 10                  # 滤波器组数量
        self.Nm = 8 if self.Fs <= 300 else 10
        self.T = int(round(self.Fs * self.ws))  # 时间窗口内的采样点数

        # 相位信息
        self.phases = [float(p) for p in phases] if phases is not None else None

        # 生成参考信号（如果提供了 phases，使用频率-相位联合编码）
        self.reference_signals = self.get_reference_signal(
            num_harmonics, self.targets, self.phases
        )

        # 频率权重设置 - 统一权重
        self.frequency_weights = np.ones(self.Nf)  # 所有频率权重统一为1

    def get_reference_signal(self, num_harmonics, targets, phases=None):
        """
        为每个目标频率生成参考信号（支持频率-相位联合编码）。

        参数:
            num_harmonics: 谐波数量
            targets: 目标频率列表 (Hz)
            phases: 目标相位列表 (弧度), 可选。
                    若提供, 参考信号 = sin(2π·h·f·t + h·φ)
                    传入 None 则退化为标准版本 sin/cos

        返回:
            reference_signals: 形状为[频率数量, 2*谐波数量, 采样点数]的参考信号数组
        """
        reference_signals = []
        t = np.arange(0, (self.T / self.Fs), step=1.0 / self.Fs)

        has_phase = phases is not None and len(phases) == len(targets)

        for i, f in enumerate(targets):
            reference_f = []
            phi = float(phases[i]) if has_phase else None

            for h in range(1, num_harmonics + 1):
                if has_phase:
                    # 频率-相位联合编码: 参考信号带相位偏移
                    # 谐波相位 = h × 基频相位 (符合 SSVEP 谐波特性)
                    h_phi = float(h) * phi
                    reference_f.append(
                        np.sin(2 * np.pi * h * f * t + h_phi)[0:self.T]
                    )
                    reference_f.append(
                        np.cos(2 * np.pi * h * f * t + h_phi)[0:self.T]
                    )
                else:
                    # 标准版本: 仅频率编码
                    reference_f.append(np.sin(2 * np.pi * h * f * t)[0:self.T])
                    reference_f.append(np.cos(2 * np.pi * h * f * t)[0:self.T])

            reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        return reference_signals

    @staticmethod
    def _fast_cca_corr(X, Y):
        """快速CCA: 使用QR+SVD计算典型相关系数 (比sklearn PLS快~20x).

        X: (n_channels, n_samples), Y: (n_components, n_samples)
        返回最大平方相关系数.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        eps = 1e-10

        # 去均值
        X = X - X.mean(axis=-1, keepdims=True)
        Y = Y - Y.mean(axis=-1, keepdims=True)

        # QR分解以获得正交基 (处理可能的共线性)
        Qx, _ = np.linalg.qr(X.T)
        Qy, _ = np.linalg.qr(Y.T)

        # 通常保留 min(n_samples, n_features) 个分量
        k = min(Qx.shape[1], Qy.shape[1])
        Qx, Qy = Qx[:, :k], Qy[:, :k]

        # 交叉协方差的正交基
        C = Qx.T @ Qy
        try:
            _, S, _ = np.linalg.svd(C, full_matrices=False)
            rho = float(S[0]) if len(S) > 0 else 0.0
        except np.linalg.LinAlgError:
            rho = 0.0

        return max(0.0, min(1.0, rho ** 2))

    def find_correlation(self, n_components, x, y):
        """计算测试数据与所有频率参考信号的CCA相关系数。

        x: [channels, samples]
        y: [n_freqs, n_components, samples]
        返回: [n_freqs] 每个频率的最大平方相关系数
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        num_freq = y.shape[0]
        result = np.zeros(num_freq, dtype=float)
        for freq_idx in range(num_freq):
            result[freq_idx] = self._fast_cca_corr(x, y[freq_idx])
        return result

    def filter_bank(self, eeg):
        """
        应用滤波器组处理脑电信号
        
        参数:
            eeg: 原始脑电信号，形状为[通道数, 采样点数]
            
        返回:
            result: 滤波后的信号，形状为[滤波器数量, 通道数, 采样点数]
        """
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim != 2:
            raise ValueError("EEG input must be [channels, samples]")
        eeg = eeg[:, -self.T:]
        n_channels = int(eeg.shape[0])
        result = np.zeros((self.Nm, n_channels, self.T), dtype=float)

        nyq = self.Fs / 2  # 奈奎斯特频率

        # 定义各个滤波器的通带和阻带
        pass_band = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]  # 通带起始频率
        stop_band = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]  # 阻带起始频率
        high_cut_pass = min(80, int(self.Fs * 0.45))
        high_cut_stop = min(90, int(self.Fs * 0.48))

        # 滤波器设计参数 (matching TDCA/reference standard)
        gpass, gstop, rp = 3, 40, 0.5

        # 应用每个滤波器
        for i in range(self.Nm):
            # 计算归一化通带和阻带边界
            wp = np.array([pass_band[i] / nyq, high_cut_pass / nyq])
            ws = np.array([stop_band[i] / nyq, high_cut_stop / nyq])
            
            # 设计切比雪夫I型滤波器
            [n, wn] = signal.cheb1ord(wp, ws, gpass, gstop)
            [b, a] = signal.cheby1(n, rp, wn, 'bandpass')
            
            # 应用滤波器(零相位滤波)，沿时间轴处理。padlen 根据窗口长度裁剪，避免短窗口报错。
            padlen = min(3 * (max(len(b), len(a)) - 1), max(1, eeg.shape[-1] - 1))
            data = signal.filtfilt(b, a, eeg, axis=-1, padlen=padlen).copy()
            result[i, :, :] = data

        return result

    def classify(self, test_data):
        """
        对测试数据进行SSVEP频率识别
        
        参数:
            test_data: 测试脑电数据，形状为[通道数, 采样点数]
            
        返回:
            result: 识别结果的索引，对应目标频率列表中的位置
        """
        scores = self.score_vector(test_data)
        result = np.argmax(scores)
        # self.sendResultSignal.emit(result)  # 发送结果信号(当前已注释)
        return result

    def classify_4_class(self, test_data):
        """
        对测试数据进行4分类SSVEP频率识别（仅识别前4个频率）
        用于M键模式：前进、后退、左移、右移
        
        参数:
            test_data: 测试脑电数据，形状为[通道数, 采样点数]
            
        返回:
            result: 识别结果的索引（0-3），对应前4个频率
        """
        # 只使用前4个频率的参考信号
        reference_signals = self.reference_signals[:4]
        
        # 取最后T个采样点
        test_data = test_data[:, -self.T:]
        
        # 应用滤波器组
        test_data = self.filter_bank(test_data)

        # 计算滤波器权重(按照1/f^1.25+0.25的规则)
        fb_weight = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        
        result = np.zeros(4)  # 只计算前4个频率
        
        # 对每个滤波器的输出计算相关系数并加权
        for fb_i in range(self.Nm):
            x = test_data[fb_i]
            y = reference_signals
            w = fb_weight[fb_i]
            # 计算加权平方相关系数
            result += (w * (self.find_correlation(3, x, y) ** 2))
        
        # 应用频率权重调整（只对前4个频率，减小特定频率的权重）
        result = result * self.frequency_weights[:4]
        
        # 返回最大相关系数对应的索引（0-3）
        result = np.argmax(result)
        return result

    def set_frequency_weight(self, freq_index, weight):
        """
        设置特定频率的权重
        
        参数:
            freq_index: 频率索引（0-based）
            weight: 权重值，通常在0.1-1.0之间，1.0为正常权重
        """
        if 0 <= freq_index < self.Nf:
            self.frequency_weights[freq_index] = weight
            print(f"频率索引 {freq_index} 的权重已设置为 {weight}")
        else:
            print(f"无效的频率索引: {freq_index}，有效范围: 0-{self.Nf-1}")

    def set_frequency_weights(self, weights):
        """
        批量设置频率权重

        参数:
            weights: 与频率数量一致的一维权重数组
        """
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.shape[0] != self.Nf:
            raise ValueError(f"weights length mismatch: expected {self.Nf}, got {arr.shape[0]}")
        self.frequency_weights = arr.copy()

    def score_vector(self, test_data):
        """
        返回每个目标频率的打分向量

        参数:
            test_data: 测试脑电数据，形状为[通道数, 采样点数]

        返回:
            scores: 形状为[频率数量]的分数向量
        """
        reference_signals = self.reference_signals
        test_data = test_data[:, -self.T:]
        test_data = self.filter_bank(test_data)

        fb_weight = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        scores = np.zeros(self.Nf)

        for fb_i in range(self.Nm):
            x = test_data[fb_i]
            y = reference_signals
            w = fb_weight[fb_i]
            scores += (w * (self.find_correlation(3, x, y) ** 2))

        return scores * self.frequency_weights

    def classify_with_scores(self, test_data):
        """
        返回分类结果、分数向量与置信度（Top1-Top2差值）
        """
        scores = self.score_vector(test_data)
        result = int(np.argmax(scores))
        if scores.size >= 2:
            top2 = np.partition(scores, -2)[-2:]
            confidence = float(top2[1] - top2[0])
        elif scores.size == 1:
            confidence = float(scores[0])
        else:
            confidence = 0.0
        return result, scores, confidence
    
    def get_frequency_weights(self):
        """
        获取当前的频率权重设置
        
        返回:
            frequency_weights: 频率权重数组
        """
        return self.frequency_weights.copy()
    
    def reset_frequency_weights(self):
        """
        重置所有频率权重为1.0（默认值）
        """
        self.frequency_weights = np.ones(self.Nf)
        print("所有频率权重已重置为默认值1.0")
