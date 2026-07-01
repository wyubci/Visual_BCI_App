from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import eigh

import ssvep_tdca_experiment as exp


OUT_DIR = Path("ssvep_results") / "more_models_s1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(scores: np.ndarray) -> np.ndarray:
    return exp.normalize_scores(scores)


def top_margin(scores: np.ndarray) -> np.ndarray:
    scores = normalize(scores)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a).astype(np.float64)
    b = np.ravel(b).astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    den = np.sqrt(float(a @ a) * float(b @ b))
    return 0.0 if den <= 1e-12 else float((a @ b) / den)


def trca_filter(class_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one TRCA spatial filter and the class template."""
    x = class_trials[:, :, :exp.WINDOW_SAMPLES].copy()
    x -= np.mean(x, axis=-1, keepdims=True)
    template = np.mean(x, axis=0)
    n_trials, n_channels, _ = x.shape
    s = np.zeros((n_channels, n_channels), dtype=np.float64)
    q = np.zeros_like(s)
    for i in range(n_trials):
        q += x[i] @ x[i].T
        for j in range(i + 1, n_trials):
            s += x[i] @ x[j].T + x[j] @ x[i].T
    reg = 1e-6 * np.trace(q) / max(1, n_channels)
    values, vectors = eigh((s + s.T) / 2.0, (q + q.T) / 2.0 + reg * np.eye(n_channels), check_finite=False)
    w = vectors[:, np.argmax(values)]
    return w, template


class TRCA:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.filters_ = []
        self.templates_ = []
        for label in range(exp.N_CLASSES):
            w, template = trca_filter(x_train[y_train == label])
            self.filters_.append(w)
            self.templates_.append(template)
        self.filters_ = np.asarray(self.filters_)
        self.templates_ = np.asarray(self.templates_)
        return self

    def decision_function(self, x_test: np.ndarray) -> np.ndarray:
        x = x_test[:, :, :exp.WINDOW_SAMPLES].copy()
        x -= np.mean(x, axis=-1, keepdims=True)
        scores = np.zeros((x.shape[0], exp.N_CLASSES), dtype=np.float64)
        for i, trial in enumerate(x):
            for label in range(exp.N_CLASSES):
                w = self.filters_[label]
                scores[i, label] = corr(w @ trial, w @ self.templates_[label])
        return scores

    def predict(self, x_test: np.ndarray):
        scores = self.decision_function(x_test)
        return np.argmax(scores, axis=1).astype(int), scores


class FBTRCA:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.models_ = []
        self.filters_ = exp.FILTER_BANK
        self.weights_ = exp.FB_WEIGHTS
        for sos in self.filters_:
            xf = signal.sosfiltfilt(sos, x_train, axis=-1)
            self.models_.append(TRCA().fit(xf, y_train))
        return self

    def decision_function(self, x_test: np.ndarray) -> np.ndarray:
        scores = np.zeros((x_test.shape[0], exp.N_CLASSES), dtype=np.float64)
        for weight, sos, model in zip(self.weights_, self.filters_, self.models_):
            xf = signal.sosfiltfilt(sos, x_test, axis=-1)
            scores += weight * normalize(model.decision_function(xf))
        return scores

    def predict(self, x_test: np.ndarray):
        scores = self.decision_function(x_test)
        return np.argmax(scores, axis=1).astype(int), scores


class TDCAFBTRCAFusion:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.tdca_ = exp.TDCA(n_components=2).fit(x_train, y_train)
        self.fbtrca_ = FBTRCA().fit(x_train, y_train)
        tdca = normalize(self.tdca_.decision_function(x_train))
        trca = normalize(self.fbtrca_.decision_function(x_train))
        best = (-1.0, -1.0, 0.0)
        for alpha in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            scores = tdca + alpha * trca
            pred = np.argmax(scores, axis=1)
            acc = float(np.mean(pred == y_train))
            margin = float(np.mean(top_margin(scores)))
            if (acc, margin) > best[:2]:
                best = (acc, margin, alpha)
        self.alpha_ = best[2]
        return self

    def predict(self, x_test: np.ndarray):
        scores = normalize(self.tdca_.decision_function(x_test)) + self.alpha_ * normalize(self.fbtrca_.decision_function(x_test))
        return np.argmax(scores, axis=1).astype(int), scores


class TriBranchTDCA:
    """TDCA + FBCCA + FBTRCA, calibrated on training data."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.tdca_ = exp.TDCA(n_components=2).fit(x_train, y_train)
        self.fbtrca_ = FBTRCA().fit(x_train, y_train)
        tdca = normalize(self.tdca_.decision_function(x_train))
        fbcca = normalize(exp.fbcca_scores(x_train))
        trca = normalize(self.fbtrca_.decision_function(x_train))
        best = (-1.0, -1.0, 0.0, 0.0)
        for a in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75):
            for b in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75):
                scores = tdca + a * fbcca + b * trca
                pred = np.argmax(scores, axis=1)
                acc = float(np.mean(pred == y_train))
                margin = float(np.mean(top_margin(scores)))
                if (acc, margin) > best[:2]:
                    best = (acc, margin, a, b)
        self.alpha_fbcca_ = best[2]
        self.beta_trca_ = best[3]
        return self

    def predict(self, x_test: np.ndarray):
        scores = (
            normalize(self.tdca_.decision_function(x_test))
            + self.alpha_fbcca_ * normalize(exp.fbcca_scores(x_test))
            + self.beta_trca_ * normalize(self.fbtrca_.decision_function(x_test))
        )
        return np.argmax(scores, axis=1).astype(int), scores


def make_model(name: str):
    if name == "TDCA":
        return exp.TDCA()
    if name == "ImprovedTDCA":
        return None
    if name == "TRCA":
        return TRCA()
    if name == "FBTRCA":
        return FBTRCA()
    if name == "TDCA-FBTRCA":
        return TDCAFBTRCAFusion()
    if name == "TriBranchTDCA":
        return TriBranchTDCA()
    raise KeyError(name)


def evaluate(window_sec: float, model_names: list[str]) -> list[dict]:
    exp.configure_window(window_sec)
    sid, mat_path = exp.discover_subject_files(Path("benchmark"), limit=1)[0]
    data = exp.load_subject_data(mat_path)
    x, y, block_ids = exp.extract_trials(data, extra_points=exp.MAX_DELAY_POINTS)
    rows = []
    for block in np.unique(block_ids):
        train = block_ids != block
        test = block_ids == block
        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]
        for name in model_names:
            if name == "ImprovedTDCA":
                model = exp.ImprovedTDCA().fit(x_train, y_train, block_ids[train])
            else:
                model = make_model(name).fit(x_train, y_train)
            pred, scores = model.predict(x_test)
            acc = float(np.mean(pred == y_test))
            rows.append({
                "window_sec": window_sec,
                "block": int(block),
                "method": name,
                "accuracy": acc,
                "mean_margin": float(np.mean(top_margin(scores))),
                "itr_bits_min": exp.itr_bits_per_min(acc),
            })
            print(f"window={window_sec:g}s block={int(block)+1}/6 {name} acc={acc:.3f}", flush=True)
    return rows


def main() -> None:
    models = ["TDCA", "ImprovedTDCA", "TRCA", "FBTRCA", "TDCA-FBTRCA", "TriBranchTDCA"]
    windows = [0.3, 0.5, 1.0]
    rows = []
    for window in windows:
        rows.extend(evaluate(window, models))
    block_df = pd.DataFrame(rows)
    summary = (
        block_df.groupby(["window_sec", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            mean_margin=("mean_margin", "mean"),
            itr_bits_min=("itr_bits_min", "mean"),
        )
        .sort_values(["window_sec", "accuracy", "mean_margin"], ascending=[True, False, False])
    )
    block_df.to_csv(OUT_DIR / "block_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "more_models_summary.csv", index=False, encoding="utf-8-sig")
    for metric, ylabel, out_name in [
        ("accuracy", "Accuracy", "more_models_accuracy.png"),
        ("mean_margin", "Mean top-2 margin", "more_models_margin.png"),
        ("itr_bits_min", "ITR (bits/min)", "more_models_itr.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
        for method, sub in summary.groupby("method"):
            sub = sub.sort_values("window_sec")
            ax.plot(sub["window_sec"], sub[metric], marker="o", label=method)
        ax.set_xlabel("Analysis window (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"S1 additional model search - {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / out_name)
    print("\nAdditional model summary")
    print(summary.to_string(index=False))
    print("\nSaved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
