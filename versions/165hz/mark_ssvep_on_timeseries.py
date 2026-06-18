from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch


def bandpass(data: np.ndarray, fs: float, low: float = 5.0, high: float = 45.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def find_ssvep_intervals(
    data: np.ndarray,
    fs: float,
    peak_freq: float,
    win_sec: float = 0.30,
    step_sec: float = 0.05,
    bw: float = 1.0,
) -> list[tuple[float, float]]:
    n_ch, n_times = data.shape
    win = max(16, int(win_sec * fs))
    step = max(1, int(step_sec * fs))

    scores = []
    centers = []

    for s in range(0, n_times - win + 1, step):
        e = s + win
        seg = data[:, s:e]
        f = np.fft.rfftfreq(win, d=1.0 / fs)
        spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        p = spec.mean(axis=0)

        total_mask = (f >= 5.0) & (f <= 45.0)
        f1_mask = (f >= peak_freq - bw) & (f <= peak_freq + bw)
        f2_mask = (f >= 2 * peak_freq - bw) & (f <= 2 * peak_freq + bw)

        total = p[total_mask].sum() + 1e-12
        target = p[f1_mask].sum()
        if np.any(f2_mask):
            target += 0.5 * p[f2_mask].sum()

        score = target / total
        scores.append(score)
        centers.append((s + e) / 2.0 / fs)

    scores = np.asarray(scores)
    centers = np.asarray(centers)

    if len(scores) == 0:
        return []

    # 自适应阈值：取中高分段，兼顾不同被试和增益
    thr = float(np.quantile(scores, 0.75))
    active = scores >= thr

    intervals = []
    if np.any(active):
        idx = np.where(active)[0]
        start_i = idx[0]
        prev_i = idx[0]
        for i in idx[1:]:
            if i == prev_i + 1:
                prev_i = i
                continue
            intervals.append((centers[start_i] - win_sec / 2, centers[prev_i] + win_sec / 2))
            start_i = i
            prev_i = i
        intervals.append((centers[start_i] - win_sec / 2, centers[prev_i] + win_sec / 2))

    # 裁剪到合法时间范围
    t_max = n_times / fs
    clipped = [(max(0.0, a), min(t_max, b)) for a, b in intervals if b > a]
    return clipped


def process_file(mat_path: Path, out_path: Path) -> None:

    mat = loadmat(mat_path)
    if "data" not in mat:
        raise KeyError("MAT file does not contain key: data")

    data = np.asarray(mat["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")
    if data.shape[0] > data.shape[1]:
        data = data.T

    fs = 250.0
    n_ch, n_times = data.shape
    t = np.arange(n_times) / fs

    # 去直流并滤波
    data = data - data.mean(axis=1, keepdims=True)
    data = bandpass(data, fs, 5, 45)

    # 先找主峰频率（6-20Hz）
    f, p = welch(data, fs=fs, nperseg=min(n_times, int(fs)), axis=1)
    mean_p = p.mean(axis=0)
    mask = (f >= 6.0) & (f <= 20.0)
    peak_freq = float(f[mask][np.argmax(mean_p[mask])])

    intervals = find_ssvep_intervals(data, fs, peak_freq=peak_freq, win_sec=0.30, step_sec=0.05, bw=1.0)

    # 堆叠绘图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    spacing = np.percentile(np.abs(data), 95) * 3.5
    if spacing < 1.0:
        spacing = 1.0

    y_offsets = np.arange(n_ch)[::-1] * spacing
    colors = ["#2f2f2f"] * n_ch

    for ch in range(n_ch):
        y = data[ch] + y_offsets[ch]
        ax.plot(t, y, color=colors[ch], linewidth=0.8)

    # 给每一行通道加红框（类似你给的示意图）
    for a, b in intervals:
        width = b - a
        for ch in range(n_ch):
            y_mid = y_offsets[ch]
            rect = Rectangle(
                (a, y_mid - spacing * 0.30),
                width,
                spacing * 0.60,
                fill=False,
                edgecolor="red",
                linewidth=1.2,
                alpha=0.85,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, t[-1])
    ax.set_yticks(y_offsets)
    ax.set_yticklabels([f"EEG{idx + 1}" for idx in range(n_ch)])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Time-domain EEG with SSVEP band boxes (peak ~ {peak_freq:.1f} Hz)")
    ax.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(out_path)

    print(f"Input: {mat_path.resolve()}")
    print(f"Detected peak: {peak_freq:.2f} Hz")
    print(f"Intervals: {intervals}")
    print(f"Output: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark SSVEP-active intervals with red boxes on time-domain EEG.")
    parser.add_argument(
        "mat_path",
        nargs="?",
        default="saveCarData/hc33/468.mat",
        help="Path to input .mat file containing variable 'data'",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: <mat_stem>_timeseries_ssvep_marked.png)",
    )
    args = parser.parse_args()

    mat_path = Path(args.mat_path)
    if args.out is None:
        out_path = mat_path.with_name(f"{mat_path.stem}_timeseries_ssvep_marked.png")
    else:
        out_path = Path(args.out)

    process_file(mat_path, out_path)


if __name__ == "__main__":
    main()
