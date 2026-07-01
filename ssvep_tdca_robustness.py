from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ssvep_tdca_experiment as exp


OUT_DIR = Path("ssvep_results") / "robustness_s1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def top_margin(scores: np.ndarray) -> np.ndarray:
    scores = exp.normalize_scores(scores)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def add_noise_snr(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(x ** 2, axis=-1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(size=x.shape) * np.sqrt(np.maximum(noise_power, 1e-12))
    return x + noise


def evaluate_window(subject_id: int, mat_path: Path, window_sec: float, noise_snrs: tuple[float, ...]):
    exp.configure_window(window_sec)
    data = exp.load_subject_data(mat_path)
    x, y, block_ids = exp.extract_trials(data, extra_points=exp.MAX_DELAY_POINTS)
    rng = np.random.default_rng(20260528 + int(window_sec * 100))

    clean_rows = []
    noise_rows = []
    trial_rows = []

    for test_block in np.unique(block_ids):
        train_idx = block_ids != test_block
        test_idx = block_ids == test_block
        x_train, y_train, block_train = x[train_idx], y[train_idx], block_ids[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        models = {
            "TDCA": exp.TDCA().fit(x_train, y_train),
            "ImprovedTDCA": exp.ImprovedTDCA().fit(x_train, y_train, block_train),
        }

        for name, model in models.items():
            pred, scores = model.predict(x_test)
            margins = top_margin(scores)
            acc = float(np.mean(pred == y_test))
            clean_rows.append({
                "subject": subject_id,
                "window_sec": window_sec,
                "block": int(test_block),
                "method": name,
                "accuracy": acc,
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "min_margin": float(np.min(margins)),
                "itr_bits_min": exp.itr_bits_per_min(acc),
            })
            for yt, yp, margin in zip(y_test, pred, margins):
                trial_rows.append({
                    "subject": subject_id,
                    "window_sec": window_sec,
                    "block": int(test_block),
                    "method": name,
                    "true_label": int(yt),
                    "pred_label": int(yp),
                    "correct": int(yt == yp),
                    "margin": float(margin),
                })

            for snr in noise_snrs:
                noisy = add_noise_snr(x_test, snr, rng)
                pred_n, scores_n = model.predict(noisy)
                margins_n = top_margin(scores_n)
                acc_n = float(np.mean(pred_n == y_test))
                noise_rows.append({
                    "subject": subject_id,
                    "window_sec": window_sec,
                    "block": int(test_block),
                    "method": name,
                    "snr_db": snr,
                    "accuracy": acc_n,
                    "mean_margin": float(np.mean(margins_n)),
                    "itr_bits_min": exp.itr_bits_per_min(acc_n),
                })

        print(f"window={window_sec:g}s block={int(test_block)+1}/6 done", flush=True)

    return clean_rows, noise_rows, trial_rows


def summarize_block_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["window_sec", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std_across_blocks=("accuracy", "std"),
            mean_margin=("mean_margin", "mean"),
            median_margin=("median_margin", "mean"),
            min_margin=("min_margin", "mean"),
            itr_bits_min=("itr_bits_min", "mean"),
        )
        .sort_values(["window_sec", "method"])
    )


def make_plots(short_summary: pd.DataFrame, noise_summary: pd.DataFrame) -> None:
    colors = {"TDCA": "#54A24B", "ImprovedTDCA": "#B279A2"}

    for metric, ylabel, out_name in [
        ("accuracy", "Accuracy", "short_window_accuracy.png"),
        ("mean_margin", "Mean normalized top-2 margin", "short_window_margin.png"),
        ("itr_bits_min", "ITR (bits/min)", "short_window_itr.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
        for method, sub in short_summary.groupby("method"):
            ax.plot(sub["window_sec"], sub[metric], marker="o", label=method, color=colors[method])
            for _, row in sub.iterrows():
                label = f"{row[metric]:.3f}" if metric != "itr_bits_min" else f"{row[metric]:.1f}"
                ax.text(row["window_sec"], row[metric], label, fontsize=8, ha="center", va="bottom")
        ax.set_xlabel("Analysis window (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"S1 TDCA vs ImprovedTDCA - {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / out_name)

    for window_sec, win_df in noise_summary.groupby("window_sec"):
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
        for method, sub in win_df.groupby("method"):
            sub = sub.sort_values("snr_db")
            ax.plot(sub["snr_db"], sub["accuracy"], marker="o", label=method, color=colors[method])
            for _, row in sub.iterrows():
                ax.text(row["snr_db"], row["accuracy"], f"{row['accuracy']:.3f}", fontsize=8, ha="center", va="bottom")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"S1 noise robustness, window={window_sec:g}s")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"noise_accuracy_{str(window_sec).replace('.', '_')}s.png")


def main() -> None:
    subject_id, mat_path = exp.discover_subject_files(Path("benchmark"), limit=1)[0]
    windows = (0.3, 0.4, 0.5, 0.8, 1.0)
    noise_windows = {0.5, 1.0}
    noise_snrs = (20.0, 10.0, 5.0, 0.0)

    clean_rows, noise_rows, trial_rows = [], [], []
    for window_sec in windows:
        n_snrs = noise_snrs if window_sec in noise_windows else ()
        c, n, t = evaluate_window(subject_id, mat_path, window_sec, n_snrs)
        clean_rows.extend(c)
        noise_rows.extend(n)
        trial_rows.extend(t)

    clean_df = pd.DataFrame(clean_rows)
    noise_df = pd.DataFrame(noise_rows)
    trial_df = pd.DataFrame(trial_rows)
    short_summary = summarize_block_rows(clean_df)
    noise_summary = (
        noise_df.groupby(["window_sec", "snr_db", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std_across_blocks=("accuracy", "std"),
            mean_margin=("mean_margin", "mean"),
            itr_bits_min=("itr_bits_min", "mean"),
        )
        .sort_values(["window_sec", "snr_db", "method"])
    )

    clean_df.to_csv(OUT_DIR / "block_metrics.csv", index=False, encoding="utf-8-sig")
    trial_df.to_csv(OUT_DIR / "trial_margins.csv", index=False, encoding="utf-8-sig")
    short_summary.to_csv(OUT_DIR / "short_window_summary.csv", index=False, encoding="utf-8-sig")
    noise_summary.to_csv(OUT_DIR / "noise_robustness_summary.csv", index=False, encoding="utf-8-sig")
    make_plots(short_summary, noise_summary)

    print("\nShort-window summary")
    print(short_summary.to_string(index=False))
    print("\nNoise robustness summary")
    print(noise_summary.to_string(index=False))
    print("\nSaved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
