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


OUT_DIR = Path("ssvep_results") / "model_search_s1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(scores: np.ndarray) -> np.ndarray:
    return exp.normalize_scores(scores)


def top_margin(scores: np.ndarray) -> np.ndarray:
    scores = normalize(scores)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def spd_inv_sqrt(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    vals, vecs = eigh(cov + eps * np.eye(cov.shape[0]), check_finite=False)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def logm_spd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    vals, vecs = eigh(cov + eps * np.eye(cov.shape[0]), check_finite=False)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def cov_trials(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=-1, keepdims=True)
    cov = np.matmul(x, np.swapaxes(x, -1, -2)) / max(1, x.shape[-1] - 1)
    tr = np.trace(cov, axis1=1, axis2=2)[:, None, None]
    return cov / np.maximum(tr, 1e-12)


class RiemannMDM:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        cov = cov_trials(x_train[:, :, :exp.WINDOW_SAMPLES])
        ref = np.mean(cov, axis=0)
        self.align_ = spd_inv_sqrt(ref)
        aligned = self.align_[None, :, :] @ cov @ self.align_[None, :, :]
        logs = np.asarray([logm_spd(c) for c in aligned])
        self.templates_ = []
        for label in range(exp.N_CLASSES):
            self.templates_.append(np.mean(logs[y_train == label], axis=0))
        self.templates_ = np.asarray(self.templates_)
        return self

    def decision_function(self, x_test: np.ndarray) -> np.ndarray:
        cov = cov_trials(x_test[:, :, :exp.WINDOW_SAMPLES])
        aligned = self.align_[None, :, :] @ cov @ self.align_[None, :, :]
        logs = np.asarray([logm_spd(c) for c in aligned])
        scores = np.empty((x_test.shape[0], exp.N_CLASSES), dtype=np.float64)
        for i, logc in enumerate(logs):
            dist = np.asarray([np.linalg.norm(logc - tmpl, ord="fro") for tmpl in self.templates_])
            scores[i] = -dist
        return scores

    def predict(self, x_test: np.ndarray):
        scores = self.decision_function(x_test)
        return np.argmax(scores, axis=1).astype(int), scores


class RATDCA:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        cov = cov_trials(x_train[:, :, :exp.WINDOW_SAMPLES])
        self.align_ = spd_inv_sqrt(np.mean(cov, axis=0))
        x_aligned = self.align_[None, :, :] @ x_train
        self.model_ = exp.TDCA().fit(x_aligned, y_train)
        return self

    def predict(self, x_test: np.ndarray):
        return self.model_.predict(self.align_[None, :, :] @ x_test)


class MultiDelayTDCA:
    def __init__(self, delays=(3, 5, 8), n_components=1):
        self.delays = delays
        self.n_components = n_components

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.models_ = [exp.TDCA(delay_points=d, n_components=self.n_components).fit(x_train, y_train) for d in self.delays]
        return self

    def predict(self, x_test: np.ndarray):
        parts = [normalize(model.decision_function(x_test)) for model in self.models_]
        scores = np.mean(np.stack(parts, axis=0), axis=0)
        return np.argmax(scores, axis=1).astype(int), scores


class AdaptiveFBTDCA:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model_ = exp.TDCA().fit(x_train, y_train)
        band_scores = []
        band_weights = []
        for fb_i, (sos, (w, templates)) in enumerate(zip(self.model_.filters_, self.model_.models_)):
            xf = signal.sosfiltfilt(sos, x_train, axis=-1)
            scores = np.zeros((x_train.shape[0], exp.N_CLASSES), dtype=np.float64)
            for trial_i in range(xf.shape[0]):
                for label in range(exp.N_CLASSES):
                    aug = exp.tdca_augment_q(xf[trial_i], exp.REF_QS[label], self.model_.delay_points)[0]
                    feat = w.T @ aug
                    scores[trial_i, label] = exp.corr_flat(feat, templates[label])
            pred = np.argmax(scores, axis=1)
            acc = float(np.mean(pred == y_train))
            margin = float(np.mean(top_margin(scores)))
            band_scores.append(scores)
            band_weights.append(max(1e-3, acc * max(margin, 1e-3)))
        weights = np.asarray(band_weights, dtype=np.float64)
        self.learned_weights_ = weights / np.sum(weights)
        return self

    def predict(self, x_test: np.ndarray):
        scores = np.zeros((x_test.shape[0], exp.N_CLASSES), dtype=np.float64)
        for weight, sos, (w, templates) in zip(self.learned_weights_, self.model_.filters_, self.model_.models_):
            xf = signal.sosfiltfilt(sos, x_test, axis=-1)
            band = np.zeros_like(scores)
            for trial_i in range(xf.shape[0]):
                for label in range(exp.N_CLASSES):
                    aug = exp.tdca_augment_q(xf[trial_i], exp.REF_QS[label], self.model_.delay_points)[0]
                    feat = w.T @ aug
                    band[trial_i, label] = exp.corr_flat(feat, templates[label])
            scores += weight * normalize(band)
        return np.argmax(scores, axis=1).astype(int), scores


class RiemannTDCAFusion:
    def __init__(self, alpha=0.25):
        self.alpha = alpha

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.tdca_ = exp.TDCA(n_components=2).fit(x_train, y_train)
        self.riem_ = RiemannMDM().fit(x_train, y_train)
        tdca_scores = self.tdca_.decision_function(x_train)
        riem_scores = self.riem_.decision_function(x_train)
        best = (-1.0, -1.0, 0.0)
        for alpha in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            scores = normalize(tdca_scores) + alpha * normalize(riem_scores)
            pred = np.argmax(scores, axis=1)
            acc = float(np.mean(pred == y_train))
            margin = float(np.mean(top_margin(scores)))
            if (acc, margin) > best[:2]:
                best = (acc, margin, alpha)
        self.alpha = best[2]
        return self

    def predict(self, x_test: np.ndarray):
        scores = normalize(self.tdca_.decision_function(x_test)) + self.alpha * normalize(self.riem_.decision_function(x_test))
        return np.argmax(scores, axis=1).astype(int), scores


def make_model(name: str):
    if name == "TDCA":
        return exp.TDCA()
    if name == "ImprovedTDCA":
        # Existing best simple model from the previous run.
        return None
    if name == "RA-TDCA":
        return RATDCA()
    if name == "MultiDelay-TDCA":
        return MultiDelayTDCA()
    if name == "AdaptiveFB-TDCA":
        return AdaptiveFBTDCA()
    if name == "Riemann-TDCA-Fusion":
        return RiemannTDCAFusion()
    if name == "RiemannMDM":
        return RiemannMDM()
    raise KeyError(name)


def evaluate(window_sec: float, model_names: list[str]):
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
    model_names = [
        "TDCA",
        "ImprovedTDCA",
        "AdaptiveFB-TDCA",
        "MultiDelay-TDCA",
        "RA-TDCA",
        "Riemann-TDCA-Fusion",
        "RiemannMDM",
    ]
    windows = [0.3, 0.5, 1.0]
    rows = []
    for window in windows:
        rows.extend(evaluate(window, model_names))
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
    summary.to_csv(OUT_DIR / "model_search_summary.csv", index=False, encoding="utf-8-sig")

    for metric, ylabel, out_name in [
        ("accuracy", "Accuracy", "model_search_accuracy.png"),
        ("mean_margin", "Mean top-2 margin", "model_search_margin.png"),
        ("itr_bits_min", "ITR (bits/min)", "model_search_itr.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
        for method, sub in summary.groupby("method"):
            sub = sub.sort_values("window_sec")
            ax.plot(sub["window_sec"], sub[metric], marker="o", label=method)
        ax.set_xlabel("Analysis window (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"S1 TDCA architecture search - {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / out_name)

    print("\nModel search summary")
    print(summary.to_string(index=False))
    print("\nSaved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
