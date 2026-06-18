from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Rectangle
from scipy.io import loadmat


def main() -> None:
    mat_path = Path("saveCarData/hc33/468.mat")
    out_path = Path("saveCarData/hc33/468_ssvep_band_marked.png")

    mat = loadmat(mat_path)
    if "data" not in mat:
        raise KeyError("MAT file does not contain key: data")

    data = np.asarray(mat["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")
    if data.shape[0] > data.shape[1]:
        data = data.T

    # Remove per-channel DC offset, then convert uV to V for MNE
    data_uv = data - data.mean(axis=1, keepdims=True)
    data_v = data_uv * 1e-6

    fs = 250.0
    n_channels = data_v.shape[0]
    ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")

    psd = raw.compute_psd(method="welch", fmin=1, fmax=45, n_fft=250, verbose="ERROR")
    freqs = psd.freqs
    psd_data = psd.get_data()  # shape: (n_channels, n_freqs)
    mean_psd = psd_data.mean(axis=0)

    # Find strongest peak in typical SSVEP range
    ssvep_search = (freqs >= 6) & (freqs <= 20)
    peak_freq = float(freqs[ssvep_search][np.argmax(mean_psd[ssvep_search])])

    # Mark +/-1 Hz around peak as SSVEP fundamental band
    bw = 1.0
    f1_lo, f1_hi = peak_freq - bw, peak_freq + bw
    f2_center = 2 * peak_freq
    mark_h2 = f2_center + bw <= freqs.max()
    if mark_h2:
        f2_lo, f2_hi = f2_center - bw, f2_center + bw

    y_max = float(mean_psd.max())

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i in range(n_channels):
        ax.plot(freqs, psd_data[i], color="gray", alpha=0.35, linewidth=0.8)
    ax.plot(freqs, mean_psd, color="black", linewidth=2.0, label="Mean PSD")

    rect1 = Rectangle((f1_lo, 0), 2 * bw, y_max * 1.05, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect1)
    ax.text(f1_lo + 0.05, y_max * 0.92, f"SSVEP: {peak_freq:.1f} Hz", color="red", fontsize=10)

    if mark_h2:
        rect2 = Rectangle((f2_lo, 0), 2 * bw, y_max * 1.05, fill=False, edgecolor="red", linewidth=2, linestyle="--")
        ax.add_patch(rect2)
        ax.text(f2_lo + 0.05, y_max * 0.78, f"2nd harmonic: {f2_center:.1f} Hz", color="red", fontsize=9)

    ax.set_xlim(1, 45)
    ax.set_ylim(0, y_max * 1.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title("468.mat SSVEP Band Marked")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)

    print(f"Input: {mat_path.resolve()}")
    print(f"Detected SSVEP peak: {peak_freq:.2f} Hz")
    if mark_h2:
        print(f"Marked bands: [{f1_lo:.1f}, {f1_hi:.1f}] Hz and [{f2_lo:.1f}, {f2_hi:.1f}] Hz")
    else:
        print(f"Marked band: [{f1_lo:.1f}, {f1_hi:.1f}] Hz")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
