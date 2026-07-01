from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from scipy.linalg import eigh, qr, svdvals


FS = 250
PRE_STIM_SEC = 0.5
VISUAL_LATENCY_SEC = 0.14
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(round(WINDOW_SEC * FS))
NUM_HARMONICS = 5
NUM_FILTER_BANKS = 5
DELAY_POINTS = 5
MAX_DELAY_POINTS = 8
N_COMPONENTS = 1
N_CLASSES = 40

TARGET_FREQS = np.asarray(
    [base + offset for offset in np.arange(0.0, 1.0, 0.2) for base in np.arange(8.0, 16.0, 1.0)],
    dtype=np.float64,
)

CHANNEL_NAMES_64 = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
USE_CHANNELS = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
CHANNEL_IDXS = [CHANNEL_NAMES_64.index(ch) for ch in USE_CHANNELS]

FB_WEIGHTS = np.asarray([(m + 1) ** (-1.25) + 0.25 for m in range(NUM_FILTER_BANKS)], dtype=np.float64)


def discover_subject_files(data_dir: Path, limit: int | None) -> list[tuple[int, Path]]:
    files = []
    for subject_dir in sorted(data_dir.glob("S*.mat"), key=lambda p: int(p.stem[1:])):
        sid = int(subject_dir.stem[1:])
        mat_path = subject_dir / f"S{sid}.mat"
        if mat_path.exists():
            files.append((sid, mat_path))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No subject files found in {data_dir}")
    return files


def load_subject_data(mat_path: Path) -> np.ndarray:
    data = np.asarray(loadmat(mat_path)["data"], dtype=np.float64)
    if data.shape[0] != 64 or data.shape[2] != N_CLASSES:
        raise ValueError(f"Unexpected Benchmark data shape: {data.shape}")
    return data


def extract_trials(data: np.ndarray, extra_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(round((PRE_STIM_SEC + VISUAL_LATENCY_SEC) * FS))
    stop = start + WINDOW_SAMPLES + extra_points
    if stop > data.shape[1]:
        raise ValueError(f"Requested samples [{start}:{stop}] from shape {data.shape}")

    xs, ys, blocks = [], [], []
    for block in range(data.shape[3]):
        for target in range(N_CLASSES):
            xs.append(data[CHANNEL_IDXS, start:stop, target, block])
            ys.append(target)
            blocks.append(block)
    return np.stack(xs), np.asarray(ys, dtype=int), np.asarray(blocks, dtype=int)


def zscore_epoch(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - np.mean(x, axis=-1, keepdims=True)


def make_refs() -> np.ndarray:
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64) / FS
    refs = []
    for freq in TARGET_FREQS:
        rows = []
        for harmonic in range(1, NUM_HARMONICS + 1):
            rows.append(np.sin(2 * np.pi * harmonic * freq * t))
            rows.append(np.cos(2 * np.pi * harmonic * freq * t))
        refs.append(np.asarray(rows, dtype=np.float64))
    return np.asarray(refs, dtype=np.float64)


REFS = make_refs()
REF_QS = [qr(ref.T, mode="economic")[0].astype(np.float64) for ref in REFS]


def configure_window(window_sec: float) -> None:
    global WINDOW_SEC, WINDOW_SAMPLES, REFS, REF_QS
    WINDOW_SEC = float(window_sec)
    WINDOW_SAMPLES = int(round(WINDOW_SEC * FS))
    REFS = make_refs()
    REF_QS = [qr(ref.T, mode="economic")[0].astype(np.float64) for ref in REFS]


def cca_score_one(epoch: np.ndarray) -> np.ndarray:
    x = zscore_epoch(epoch[:, :WINDOW_SAMPLES])
    qx, _ = qr(x.T, mode="economic")
    scores = np.empty(N_CLASSES, dtype=np.float64)
    for label, qy in enumerate(REF_QS):
        values = svdvals(qx.T @ qy)
        scores[label] = values[0] if values.size else 0.0
    return scores


def predict_cca(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray([cca_score_one(epoch) for epoch in x_test], dtype=np.float64)
    return np.argmax(scores, axis=1).astype(int), scores


def design_filter_bank(n_bands: int = NUM_FILTER_BANKS) -> list[np.ndarray]:
    pass_low = np.asarray([6, 14, 22, 30, 38, 46, 54, 62, 70, 78], dtype=float)
    stop_low = np.asarray([4, 10, 16, 24, 32, 40, 48, 56, 64, 72], dtype=float)
    nyq = FS / 2.0
    high_pass = min(90.0, nyq - 5.0)
    high_stop = min(100.0, nyq - 2.0)
    filters = []
    for band in range(n_bands):
        wp = [pass_low[band] / nyq, high_pass / nyq]
        ws = [stop_low[band] / nyq, high_stop / nyq]
        order, wn = signal.cheb1ord(wp, ws, gpass=3, gstop=40)
        filters.append(signal.cheby1(order, rp=0.5, Wn=wn, btype="bandpass", output="sos"))
    return filters


FILTER_BANK = design_filter_bank()


def fbcca_scores(x_test: np.ndarray) -> np.ndarray:
    scores = np.zeros((x_test.shape[0], N_CLASSES), dtype=np.float64)
    for weight, sos in zip(FB_WEIGHTS, FILTER_BANK):
        xf = signal.sosfiltfilt(sos, x_test[:, :, :WINDOW_SAMPLES], axis=-1)
        band_scores = np.asarray([cca_score_one(epoch) for epoch in xf], dtype=np.float64)
        scores += weight * (band_scores ** 2)
    return scores


def predict_fbcca(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = fbcca_scores(x_test)
    return np.argmax(scores, axis=1).astype(int), scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def regularized_eigh(sb: np.ndarray, sw: np.ndarray) -> np.ndarray:
    sb = (sb + sb.T) / 2.0
    sw = (sw + sw.T) / 2.0
    reg = 1e-6 * float(np.trace(sw)) / max(sw.shape[0], 1)
    if not np.isfinite(reg) or reg <= 0:
        reg = 1e-6
    values, vectors = eigh(sb, sw + reg * np.eye(sw.shape[0]), check_finite=False)
    return vectors[:, np.argsort(values)[::-1]]


def tdca_augment_q(trials: np.ndarray, ref_q: np.ndarray, delay_points: int = DELAY_POINTS) -> np.ndarray:
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim == 2:
        trials = trials[np.newaxis, ...]
    n_trials, n_channels, n_points = trials.shape
    if n_points < WINDOW_SAMPLES + delay_points:
        raise ValueError(f"TDCA needs {WINDOW_SAMPLES + delay_points} points, got {n_points}")

    delayed = np.empty((n_trials, n_channels * (delay_points + 1), WINDOW_SAMPLES), dtype=np.float64)
    for lag in range(delay_points + 1):
        delayed[:, lag * n_channels:(lag + 1) * n_channels, :] = trials[:, :, lag:lag + WINDOW_SAMPLES]
    delayed -= np.mean(delayed, axis=-1, keepdims=True)

    projected = np.matmul(np.matmul(delayed, ref_q), ref_q.T)
    return np.concatenate([delayed, projected], axis=-1)


def dsp_fit(aug_x: np.ndarray, y: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    aug_x = aug_x - np.mean(aug_x, axis=-1, keepdims=True)
    grand_mean = np.mean(aug_x, axis=0)
    n_features = aug_x.shape[1]
    sw = np.zeros((n_features, n_features), dtype=np.float64)
    sb = np.zeros((n_features, n_features), dtype=np.float64)

    for label in range(N_CLASSES):
        xi = aug_x[y == label]
        class_mean = np.mean(xi, axis=0)
        centered = xi - class_mean
        sw += np.einsum("nct,ndt->cd", centered, centered, optimize=True)
        diff = class_mean - grand_mean
        sb += xi.shape[0] * (diff @ diff.T)

    w = regularized_eigh(sb, sw)[:, :n_components]
    features = np.einsum("fc,nct->nft", w.T, aug_x, optimize=True)
    templates = np.zeros((N_CLASSES, n_components, aug_x.shape[-1]), dtype=np.float64)
    for label in range(N_CLASSES):
        templates[label] = np.mean(features[y == label], axis=0)
    return w, templates


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a).astype(np.float64)
    b = np.ravel(b).astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return 0.0 if denom <= 1e-12 else float((a @ b) / denom)


@dataclass
class TDCA:
    n_bands: int = NUM_FILTER_BANKS
    n_components: int = N_COMPONENTS
    delay_points: int = DELAY_POINTS

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.models_: list[tuple[np.ndarray, np.ndarray]] = []
        self.filters_ = FILTER_BANK[: self.n_bands]
        self.weights_ = FB_WEIGHTS[: self.n_bands]

        for sos in self.filters_:
            xf = signal.sosfiltfilt(sos, x_train, axis=-1)
            aug_blocks, aug_labels = [], []
            for label in range(N_CLASSES):
                xi = xf[y_train == label]
                aug_blocks.append(tdca_augment_q(xi, REF_QS[label], self.delay_points))
                aug_labels.append(np.full(xi.shape[0], label, dtype=int))
            aug_x = np.concatenate(aug_blocks, axis=0)
            aug_y = np.concatenate(aug_labels, axis=0)
            self.models_.append(dsp_fit(aug_x, aug_y, self.n_components))
        return self

    def decision_function(self, x_test: np.ndarray) -> np.ndarray:
        x_test = np.asarray(x_test, dtype=np.float64)
        if x_test.ndim == 2:
            x_test = x_test[np.newaxis, ...]

        scores = np.zeros((x_test.shape[0], N_CLASSES), dtype=np.float64)
        for weight, sos, (w, templates) in zip(self.weights_, self.filters_, self.models_):
            xf = signal.sosfiltfilt(sos, x_test, axis=-1)
            for trial_i in range(xf.shape[0]):
                trial_scores = np.zeros(N_CLASSES, dtype=np.float64)
                for label in range(N_CLASSES):
                    aug = tdca_augment_q(xf[trial_i], REF_QS[label], self.delay_points)[0]
                    feat = w.T @ aug
                    trial_scores[label] = corr_flat(feat, templates[label])
                scores[trial_i] += weight * trial_scores
        return scores

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.decision_function(x_test)
        return np.argmax(scores, axis=1).astype(int), scores


@dataclass
class ImprovedTDCA:
    """TDCA plus calibrated FBCCA score fusion and a slightly richer component space."""

    n_bands: int = NUM_FILTER_BANKS
    n_components: int = 2
    delay_points: int = DELAY_POINTS
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, block_train: np.ndarray):
        self.tdca_ = TDCA(self.n_bands, self.n_components, self.delay_points).fit(x_train, y_train)
        tdca_scores = self.tdca_.decision_function(x_train)
        fb_scores = fbcca_scores(x_train)
        best_score, best_alpha = (-1.0, -1.0), 0.0
        for alpha in self.alpha_grid:
            fused = normalize_scores(tdca_scores) + alpha * normalize_scores(fb_scores)
            pred = np.argmax(fused, axis=1)
            acc = float(np.mean(pred == y_train))
            margins = np.sort(fused, axis=1)[:, -1] - np.sort(fused, axis=1)[:, -2]
            score = (acc, float(np.mean(margins)))
            if score > best_score:
                best_score, best_alpha = score, alpha
        self.alpha_ = best_alpha
        return self

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tdca_scores = self.tdca_.decision_function(x_test)
        fb_scores = fbcca_scores(x_test)
        scores = normalize_scores(tdca_scores) + self.alpha_ * normalize_scores(fb_scores)
        return np.argmax(scores, axis=1).astype(int), scores


@dataclass
class EnsembleImprovedTDCA:
    """Training-calibrated ensemble over several TDCA variants plus FBCCA fusion."""

    variants: tuple[tuple[int, int, int], ...] = ((5, 1, 5), (5, 2, 5), (5, 2, 8), (3, 2, 5))
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, block_train: np.ndarray):
        self.models_ = []
        train_score_parts = []
        for n_bands, n_components, delay_points in self.variants:
            model = TDCA(n_bands=n_bands, n_components=n_components, delay_points=delay_points).fit(x_train, y_train)
            scores = model.decision_function(x_train)
            self.models_.append(model)
            train_score_parts.append(normalize_scores(scores))

        tdca_ensemble = np.mean(np.stack(train_score_parts, axis=0), axis=0)
        fb_scores = normalize_scores(fbcca_scores(x_train))
        best_score, best_alpha = (-1.0, -1.0), 0.0
        for alpha in self.alpha_grid:
            fused = tdca_ensemble + alpha * fb_scores
            pred = np.argmax(fused, axis=1)
            acc = float(np.mean(pred == y_train))
            margins = np.sort(fused, axis=1)[:, -1] - np.sort(fused, axis=1)[:, -2]
            score = (acc, float(np.mean(margins)))
            if score > best_score:
                best_score, best_alpha = score, alpha
        self.alpha_ = best_alpha
        return self

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_parts = [normalize_scores(model.decision_function(x_test)) for model in self.models_]
        tdca_ensemble = np.mean(np.stack(score_parts, axis=0), axis=0)
        scores = tdca_ensemble + self.alpha_ * normalize_scores(fbcca_scores(x_test))
        return np.argmax(scores, axis=1).astype(int), scores


def itr_bits_per_min(acc: float, n_classes: int = N_CLASSES, trial_time_sec: float = WINDOW_SEC) -> float:
    trial_time_sec = WINDOW_SEC
    p = float(np.clip(acc, 1e-12, 1 - 1e-12))
    bits = math.log2(n_classes) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n_classes - 1))
    return max(0.0, bits) * 60.0 / trial_time_sec


def evaluate_subject(sid: int, mat_path: Path, methods: set[str]) -> tuple[list[dict], list[dict]]:
    data = load_subject_data(mat_path)
    x, y, block_ids = extract_trials(data, extra_points=MAX_DELAY_POINTS)
    block_rows: list[dict] = []
    trial_rows: list[dict] = []

    print(f"S{sid:02d}: x={x.shape}", flush=True)
    for test_block in np.unique(block_ids):
        train_idx = block_ids != test_block
        test_idx = block_ids == test_block
        x_train, y_train, blocks_train = x[train_idx], y[train_idx], block_ids[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        predictions: list[tuple[str, np.ndarray]] = []
        if "CCA" in methods:
            pred, _ = predict_cca(x_test)
            predictions.append(("CCA", pred))
        if "FBCCA" in methods:
            pred, _ = predict_fbcca(x_test)
            predictions.append(("FBCCA", pred))
        if "TDCA" in methods:
            pred, _ = TDCA().fit(x_train, y_train).predict(x_test)
            predictions.append(("TDCA", pred))
        if "ImprovedTDCA" in methods:
            model = ImprovedTDCA().fit(x_train, y_train, blocks_train)
            pred, _ = model.predict(x_test)
            predictions.append((f"ImprovedTDCA(alpha={model.alpha_:.2f})", pred))
        if "EnsembleImprovedTDCA" in methods:
            model = EnsembleImprovedTDCA().fit(x_train, y_train, blocks_train)
            pred, _ = model.predict(x_test)
            predictions.append((f"EnsembleImprovedTDCA(alpha={model.alpha_:.2f})", pred))

        for method, pred in predictions:
            acc = float(np.mean(pred == y_test))
            block_rows.append({
                "subject": sid,
                "block": int(test_block),
                "method": method,
                "accuracy": acc,
                "itr_bits_min": itr_bits_per_min(acc),
            })
            for yt, yp in zip(y_test, pred):
                trial_rows.append({
                    "subject": sid,
                    "block": int(test_block),
                    "method": method,
                    "true_label": int(yt),
                    "pred_label": int(yp),
                    "true_freq": float(TARGET_FREQS[int(yt)]),
                    "pred_freq": float(TARGET_FREQS[int(yp)]),
                    "correct": int(yt == yp),
                })
            print(f"  block {int(test_block) + 1}/6 {method:24s} acc={acc:.3f} itr={itr_bits_per_min(acc):.2f}", flush=True)
    return block_rows, trial_rows


def parse_methods(value: str) -> set[str]:
    names = {name.strip() for name in value.split(",") if name.strip()}
    valid = {"CCA", "FBCCA", "TDCA", "ImprovedTDCA", "EnsembleImprovedTDCA"}
    unknown = names - valid
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path, default=Path("ssvep_results"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--methods", default="CCA,FBCCA,TDCA,ImprovedTDCA")
    parser.add_argument("--window-sec", type=float, default=4.0)
    args = parser.parse_args()

    configure_window(args.window_sec)
    args.out.mkdir(exist_ok=True)
    methods = parse_methods(args.methods)
    subject_files = discover_subject_files(args.benchmark, args.limit if args.limit > 0 else None)

    print("Benchmark:", args.benchmark.resolve(), flush=True)
    print("Output:", args.out.resolve(), flush=True)
    print("Subjects:", [sid for sid, _ in subject_files], flush=True)
    print("Methods:", sorted(methods), flush=True)
    print("Window:", WINDOW_SEC, "sec", WINDOW_SAMPLES, "samples", flush=True)
    print("Channels:", list(zip(USE_CHANNELS, CHANNEL_IDXS)), flush=True)
    print("Frequencies:", TARGET_FREQS[:8], "...", TARGET_FREQS[-8:], flush=True)

    start = time.time()
    all_blocks, all_trials = [], []
    for sid, path in subject_files:
        block_rows, trial_rows = evaluate_subject(sid, path, methods)
        all_blocks.extend(block_rows)
        all_trials.extend(trial_rows)

    block_df = pd.DataFrame(all_blocks)
    trial_df = pd.DataFrame(all_trials)
    subject_df = (
        block_df.groupby(["subject", "method"], as_index=False)[["accuracy", "itr_bits_min"]]
        .mean()
        .sort_values(["subject", "method"])
    )
    method_df = (
        subject_df.groupby("method", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_itr_bits_min=("itr_bits_min", "mean"),
            std_itr_bits_min=("itr_bits_min", "std"),
            subjects=("subject", "count"),
        )
        .sort_values("mean_accuracy", ascending=False)
    )

    block_df.to_csv(args.out / "block_summary.csv", index=False, encoding="utf-8-sig")
    trial_df.to_csv(args.out / "trial_predictions.csv", index=False, encoding="utf-8-sig")
    subject_df.to_csv(args.out / "subject_summary.csv", index=False, encoding="utf-8-sig")
    method_df.to_csv(args.out / "method_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.bar(method_df["method"], method_df["mean_accuracy"], color=["#54A24B", "#4C78A8", "#F58518", "#B279A2"][: len(method_df)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean accuracy")
    ax.set_title(f"Benchmark SSVEP {WINDOW_SEC:.1f}s window")
    ax.grid(axis="y", alpha=0.25)
    for i, row in method_df.reset_index(drop=True).iterrows():
        ax.text(i, row["mean_accuracy"] + 0.015, f"{row['mean_accuracy']:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(args.out / "algorithm_accuracy.png")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.bar(method_df["method"], method_df["mean_itr_bits_min"], color=["#54A24B", "#4C78A8", "#F58518", "#B279A2"][: len(method_df)])
    ax.set_ylabel("Mean ITR (bits/min)")
    ax.set_title(f"Benchmark SSVEP ITR, trial={WINDOW_SEC:.1f}s")
    ax.grid(axis="y", alpha=0.25)
    for i, row in method_df.reset_index(drop=True).iterrows():
        ax.text(i, row["mean_itr_bits_min"] + 0.5, f"{row['mean_itr_bits_min']:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(args.out / "algorithm_itr.png")

    elapsed = time.time() - start
    print("\nMethod summary:", flush=True)
    print(method_df.to_string(index=False), flush=True)
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed / 60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
