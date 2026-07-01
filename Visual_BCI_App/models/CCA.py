import numpy as np
from scipy import signal


class CCA:
    """
    经典CCA（典型相关分析）模型 —— 用于SSVEP脑电信号的频率识别。
    原理：计算测试EEG信号与各频率参考信号之间的典型相关系数，
    取最大相关系数对应的频率作为分类结果。
    与FBCCA不同，本类使用单个宽带滤波器而非滤波器组。
    """

    def __init__(self, num_harmonics, times, targets, Nh=8, sample_rate=250, delay_sec=0.14):
        """
        初始化CCA分类器。

        参数:
            num_harmonics: 谐波数量，用于构造参考信号
            times: 数据窗口总长度（秒），包含视觉延迟
            targets: 目标刺激频率列表（Hz）
            Nh: 保留参数，与TDCA接口保持一致（此处未实际使用）
            sample_rate: 采样率（Hz）
            delay_sec: 视觉延迟（秒），默认0.14s
        """
        self.Nh = Nh  # 子带数量（保留接口兼容，未实际使用）
        self.Fs = int(sample_rate)  # 采样率（Hz）
        self.targets = [float(x) for x in targets]
        self.Nf = len(self.targets)  # 目标频率数量（类别数）
        self.delay_sec = float(delay_sec)
        self.ws = max(times - delay_sec, 0.2)  # 有效窗口长度 = 总长度 - 视觉延迟，最小0.2秒
        self.T = int(round(self.Fs * self.ws))  # 有效窗口对应的采样点数
        self.reference_signals = self.get_reference_signal(num_harmonics, self.targets)  # 构造所有频率的参考信号矩阵
        self.frequency_weights = np.ones(self.Nf, dtype=float)  # 频率权重向量（全1表示等权重，可在线调整）

        # 设计单个宽带带通滤波器，覆盖SSVEP相关频段（与TDCA频段范围保持一致）
        nyq = self.Fs / 2.0
        wp = np.array([6.0 / nyq, min(80.0, self.Fs * 0.45) / nyq])  # [6Hz, 80Hz]
        ws = np.array([4.0 / nyq, min(90.0, self.Fs * 0.48) / nyq])  # [4Hz, 90Hz]
        n, wn = signal.cheb1ord(wp, ws, 3, 40)
        b, a = signal.cheby1(n, 0.5, wn, "bandpass")
        self._sos = signal.tf2sos(b, a)  # 将传递函数系数转为二阶节（SOS）格式，提高数值稳定性

    def get_reference_signal(self, num_harmonics, targets):
        """
        构造各频率的参考信号矩阵。
        对每个目标频率，生成其基频和各次谐波的正弦、余弦信号。
        返回形状 (Nf, 2*num_harmonics, T) 的数组。
        """
        t = np.arange(0, self.T / self.Fs, step=1.0 / self.Fs)  # 时间轴（秒），从0到T/Fs，步长1/Fs
        refs = []  # 存储所有频率的参考信号
        for f in targets:  # 遍历每个目标频率
            rf = []  # 存储当前频率的各次谐波分量
            for h in range(1, num_harmonics + 1):  # 从基频(h=1)到指定次谐波
                rf.append(np.sin(2 * np.pi * h * f * t)[: self.T])  # 第h次谐波的正弦分量
                rf.append(np.cos(2 * np.pi * h * f * t)[: self.T])  # 第h次谐波的余弦分量
            refs.append(rf)  # 一个频率包含 2*num_harmonics 个参考信号分量
        return np.asarray(refs, dtype=float)  # 转为numpy数组并返回

    def _filter(self, eeg):
        """
        对输入EEG数据进+
        截取最后T个时间点（对应有效窗口），然后用宽带带通滤波器做零相位滤波。
        """
        eeg = np.asarray(eeg, dtype=float)  # 确保输入为浮点数numpy数组
        if eeg.ndim != 2:
            raise ValueError("EEG input must be [channels, samples]")
        eeg = eeg[:, -self.T:]  # 截取最后T个采样点（有效数据窗口）
        padlen = min(3 * (2 * self._sos.shape[0] + 1), max(1, eeg.shape[-1] - 1))
        return signal.sosfiltfilt(self._sos, eeg, axis=-1, padlen=padlen)  # 零相位滤波，避免相位偏移影响相关性

    @staticmethod
    def _fast_cca_corr(X, Y):
        """快速CCA: 使用QR+SVD计算最大平方典型相关系数.

        X: (n_channels, n_samples), Y: (n_ref_components, n_samples)
        返回最大平方相关系数.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        X = X - X.mean(axis=-1, keepdims=True)
        Y = Y - Y.mean(axis=-1, keepdims=True)
        Qx, _ = np.linalg.qr(X.T)
        Qy, _ = np.linalg.qr(Y.T)
        k = min(Qx.shape[1], Qy.shape[1])
        Qx, Qy = Qx[:, :k], Qy[:, :k]
        C = Qx.T @ Qy
        try:
            _, S, _ = np.linalg.svd(C, full_matrices=False)
            rho = float(S[0]) if len(S) > 0 else 0.0
        except np.linalg.LinAlgError:
            rho = 0.0
        return max(0.0, min(1.0, rho ** 2))

    def _find_corr(self, x, refs):
        """计算滤波后EEG x与各频率参考信号refs的CCA得分。

        x: [channels, samples], refs: [n_freqs, 2*Nh, samples]
        返回长度为Nf的平方相关系数向量。
        """
        x = np.asarray(x, dtype=float)
        scores = np.zeros(refs.shape[0], dtype=float)
        for i in range(refs.shape[0]):
            scores[i] = self._fast_cca_corr(x, refs[i])
        return scores

    def score_vector(self, test_data):
        """
        对外接口：返回加权后的得分向量。
        先滤波，再计算CCA相关系数，最后乘以频率权重。
        """
        x = self._filter(test_data)  # 对测试数据做滤波预处理
        scores = self._find_corr(x, self.reference_signals)  # 计算与各频率参考信号的CCA相关系数得分
        return scores * self.frequency_weights  # 乘以频率权重后返回（实现频率偏好的在线调整）

    def classify(self, test_data):
        """
        简化分类接口：返回预测的类别索引。
        取得分最高的频率作为分类结果。
        """
        scores = self.score_vector(test_data)  # 获取加权得分向量
        return int(np.argmax(scores))  # 返回最大得分对应的索引（类别编号）

    def classify_with_scores(self, test_data):
        """
        分类并返回详细信息：分类结果、得分向量和置信度。
        置信度 = 最高得分 - 次高得分（差值越大越可信）。
        """
        scores = self.score_vector(test_data)  # 获取加权得分向量
        result = int(np.argmax(scores))  # 得分最高的索引即为分类结果
        if scores.size >= 2:  # 至少2个类别才能计算差值
            top2 = np.partition(scores, -2)[-2:]  # 用分区算法高效取出最大的两个得分
            confidence = float(top2[1] - top2[0])  # 置信度 = 最高分 - 次高分
        elif scores.size == 1:  # 只有一个类别的情况
            confidence = float(scores[0])  # 直接返回该得分作为置信度
        else:  # 没有得分（异常情况）
            confidence = 0.0  # 置信度设为0
        return result, scores, confidence  # 返回分类结果、得分向量、置信度

    def set_frequency_weights(self, weights):
        """
        批量设置所有频率的权重。
        用于在线自适应调整不同频率的偏好。
        """
        arr = np.asarray(weights, dtype=float).reshape(-1)  # 确保权重为浮点数一维数组
        if arr.shape[0] != self.Nf:  # 检查权重数量是否与频率数量匹配
            raise ValueError(f"weights length mismatch: expected {self.Nf}, got {arr.shape[0]}")
        self.frequency_weights = arr.copy()  # 复制并保存权重数组

    def set_frequency_weight(self, freq_index, weight):
        """
        设置单个频率的权重。
        freq_index: 频率索引
        weight: 权重值
        """
        if 0 <= freq_index < self.Nf:  # 检查索引是否在有效范围内
            self.frequency_weights[freq_index] = float(weight)  # 设置指定频率的权重

    def get_frequency_weights(self):
        """获取当前所有频率的权重（返回副本，防止外部修改）。"""
        return self.frequency_weights.copy()

    def reset_frequency_weights(self):
        """重置所有频率权重为1.0（等权重，不做偏向）。"""
        self.frequency_weights = np.ones(self.Nf, dtype=float)
