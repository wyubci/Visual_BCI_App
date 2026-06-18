from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat


def main() -> None:
    mat_path = Path("saveCarData/hc33/468.mat")
    out_path = Path("saveCarData/hc33/468_mne_plot.png")

    mat = loadmat(mat_path)
    if "data" not in mat:
        raise KeyError("MAT 文件中未找到变量 data")

    data = np.asarray(mat["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data 维度应为 2，当前为 {data.ndim}")

    # 兼容 (T, C) 或 (C, T) 两种常见排布，优先转为 (C, T)
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_channels, n_times = data.shape
    sfreq = 250.0
    ch_names = [f"EEG{i + 1}" for i in range(n_channels)]

    # 该数据存在较大直流偏置，先做每通道去均值再转为 V
    data_uv = data - data.mean(axis=1, keepdims=True)
    data_v = data_uv * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")

    # 使用 matplotlib 后端以便保存静态图
    mne.viz.set_browser_backend("matplotlib")
    # 用稳健分位数估计幅值，避免显示过扁或过爆
    amp = np.percentile(np.abs(data_v), 95)
    eeg_scaling = float(max(20e-6, min(500e-6, 2.5 * amp)))

    fig = raw.plot(
        duration=min(5.0, n_times / sfreq),
        n_channels=n_channels,
        scalings={"eeg": eeg_scaling},
        remove_dc=True,
        show=False,
        title="EEG from 468.mat",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"输入文件: {mat_path.resolve()}")
    print(f"数据形状: {data.shape} (channels, samples)")
    print(f"显示缩放: {eeg_scaling * 1e6:.1f} uV")
    print(f"输出图片: {out_path.resolve()}")


if __name__ == "__main__":
    main()
