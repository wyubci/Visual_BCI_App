from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import loadmat


INPUT_DIR = Path(r"C:\Users\adam\Desktop\Visual_BCI_App\saveCarData\hc33\train\2026-05-30")
OUTPUT_DIR = Path(r"C:\Users\adam\Desktop\脑热图")


def _ensure_ct(data: np.ndarray) -> np.ndarray:
    if data.shape[0] > data.shape[1]:
        return data.T
    return data


def _as_scalar(value, default=None):
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(INPUT_DIR.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {INPUT_DIR}")

    processed = 0
    for mat_path in mat_files:
        mat = loadmat(mat_path)
        if "data" not in mat:
            continue

        data = np.asarray(mat["data"], dtype=float)
        if data.ndim != 2:
            continue
        data = _ensure_ct(data)

        sfreq = float(_as_scalar(mat.get("sample_rate_hz"), 250.0))
        freqs = np.asarray(mat.get("stim_freqs_hz", []), dtype=float).reshape(-1)
        if freqs.size == 0:
            freqs = np.array([8.0, 9.0, 9.5, 10.0, 10.5], dtype=float)

        # Remove per-channel DC and convert uV -> V.
        data_uv = data - data.mean(axis=1, keepdims=True)
        data_v = data_uv * 1e-6

        n_channels, n_times = data_v.shape
        ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data_v, info, verbose="ERROR")

        # Time-domain EEG image.
        duration_sec = min(4.0, n_times / sfreq)
        t_max = int(duration_sec * sfreq)
        t = np.arange(t_max) / sfreq
        segment = data_uv[:, :t_max]

        trial_id = mat_path.stem.split("_", 1)[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=150)

        offset = np.nanmax(np.abs(segment))
        if not np.isfinite(offset) or offset == 0:
            offset = 1.0
        offset *= 1.35

        for idx in range(n_channels):
            ax1.plot(t, segment[idx] + idx * offset, linewidth=0.8, color="black")

        ax1.set_title(f"Raw EEG: trial {trial_id}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Channel offset")
        ax1.set_yticks(np.arange(n_channels) * offset)
        ax1.set_yticklabels(ch_names)
        ax1.grid(alpha=0.2)

        # PSD with SSVEP frequency markers.
        psd = raw.compute_psd(method="welch", fmin=1, fmax=45, n_fft=min(1024, n_times), verbose="ERROR")
        psd_freqs = psd.freqs
        psd_data = psd.get_data()
        mean_psd = psd_data.mean(axis=0) * 1e12  # uV^2/Hz

        for idx in range(n_channels):
            ax2.plot(psd_freqs, psd_data[idx] * 1e12, color="gray", alpha=0.28, linewidth=0.7)
        ax2.plot(psd_freqs, mean_psd, color="black", linewidth=2.0, label="Mean PSD")

        y_max = float(np.nanmax(mean_psd))
        if not np.isfinite(y_max) or y_max <= 0:
            y_max = 1.0

        used_freqs = []
        for f in freqs:
            if f < psd_freqs.min() or f > psd_freqs.max():
                continue
            idx = int(np.argmin(np.abs(psd_freqs - f)))
            height = float(mean_psd[idx])
            used_freqs.append((f, height))
            ax2.axvline(f, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
            ax2.scatter([f], [height], color="red", s=18, zorder=3)
            ax2.text(
                f,
                min(height + y_max * 0.04, y_max * 1.02),
                f"{f:.1f}Hz\n{height:.2f}",
                color="red",
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax2.set_xlim(1, 45)
        ax2.set_ylim(0, y_max * 1.18)
        ax2.set_title("SSVEP PSD with frequency heights")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD (uV^2/Hz)")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="upper right")

        freq_text = ", ".join(f"{f:.1f}Hz" for f, _ in used_freqs)
        fig.suptitle(f"Trial {trial_id} | Target frequencies: {freq_text}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        out_path = OUTPUT_DIR / f"{mat_path.stem}_ssvep_eeg.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        processed += 1

    print(f"Processed {processed} files.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
