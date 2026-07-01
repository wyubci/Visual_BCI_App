# -*- coding: utf-8 -*-
"""Optimized SSVEP decoders for the 165 Hz self-collected-data system.

The classes in this file keep the same public interface as CCA/FBCCA/TDCA:

- fit(X, y)
- score_vector(test_data)
- classify_with_scores(test_data)
- classify(test_data)
- set_frequency_weights / reset_frequency_weights

They are designed for the local self-collected `.mat` samples used by the car
system, not for the Benchmark 4-D data layout.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.linalg import eigh

from models.FBCCA import FBCCA
from models.TDCA import TDCA


def _normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def _top2_margin(scores):
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size < 2:
        return float(arr[0]) if arr.size == 1 else 0.0
    top2 = np.partition(arr, -2)[-2:]
    return float(top2[1] - top2[0])


def _corr(a, b):
    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    a -= np.mean(a)
    b -= np.mean(b)
    denom = math.sqrt(float(a @ a) * float(b @ b))
    if denom <= 1e-12:
        return 0.0
    return float((a @ b) / denom)


def _band_weights(n_bands):
    return np.asarray([(i + 1) ** (-1.25) + 0.25 for i in range(int(n_bands))], dtype=float)


class _BaseOptimizedDecoder:
    def __init__(self, num_harmonics, times, targets, Nh=8, sample_rate=250):
        self.num_harmonics = int(num_harmonics)
        self.Nh = int(Nh)
        self.Fs = int(sample_rate)
        self.targets = [float(x) for x in targets]
        self.Nf = len(self.targets)
        self.ws = max(float(times), 0.2)
        self.T = int(round(self.Fs * self.ws))
        self.frequency_weights = np.ones(self.Nf, dtype=float)
        self._is_fitted = False

    @property
    def is_fitted(self):
        return bool(self._is_fitted)

    def set_frequency_weight(self, freq_index, weight):
        if 0 <= int(freq_index) < self.Nf:
            self.frequency_weights[int(freq_index)] = float(weight)

    def set_frequency_weights(self, weights):
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.shape[0] != self.Nf:
            raise ValueError(f"weights length mismatch: expected {self.Nf}, got {arr.shape[0]}")
        self.frequency_weights = arr.copy()

    def get_frequency_weights(self):
        return self.frequency_weights.copy()

    def reset_frequency_weights(self):
        self.frequency_weights = np.ones(self.Nf, dtype=float)

    def clear_fit(self):
        self._is_fitted = False

    def classify_with_scores(self, test_data):
        scores = self.score_vector(test_data)
        result = int(np.argmax(scores))
        return result, scores, _top2_margin(scores)

    def classify(self, test_data):
        result, _, _ = self.classify_with_scores(test_data)
        return int(result)

    def classify_4_class_with_scores(self, test_data):
        scores = np.asarray(self.score_vector(test_data)[:4], dtype=float)
        result = int(np.argmax(scores))
        return result, scores, _top2_margin(scores)

    def classify_4_class(self, test_data):
        result, _, _ = self.classify_4_class_with_scores(test_data)
        return int(result)


class ImprovedTDCA(_BaseOptimizedDecoder):
    """TDCA plus train-calibrated FBCCA score fusion.

    This is the compact model that performed best on the full Benchmark for
    0.3 s / 0.5 s / 4 s windows.  It keeps TDCA as the main supervised branch
    and uses FBCCA as an auxiliary reference-correlation branch.
    """

    def __init__(self, num_harmonics, times, targets, Nh=8, sample_rate=250):
        super().__init__(num_harmonics, times, targets, Nh=Nh, sample_rate=sample_rate)
        self.tdca = TDCA(num_harmonics, times, targets, Nh=Nh, sample_rate=sample_rate)
        self.tdca.n_components = 2
        self.fbcca = FBCCA(num_harmonics, times, targets, Nh=Nh, sample_rate=sample_rate)
        self.required_points = max(
            int(getattr(self.tdca, "required_points", self.T)),
            int(getattr(self.fbcca, "T", self.T)),
        )
        self.alpha = 0.3

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.tdca.fit(x, y)
        tdca_scores = np.asarray([self.tdca.score_vector(sample) for sample in x], dtype=float)
        fbcca_scores = np.asarray([self.fbcca.score_vector(sample) for sample in x], dtype=float)

        best = (-1.0, -1.0, 0.0)
        for alpha in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            fused = _normalize_scores(tdca_scores) + alpha * _normalize_scores(fbcca_scores)
            pred = np.argmax(fused, axis=1)
            acc = float(np.mean(pred == y))
            margins = np.sort(fused, axis=1)[:, -1] - np.sort(fused, axis=1)[:, -2]
            margin = float(np.mean(margins))
            if (acc, margin) > best[:2]:
                best = (acc, margin, float(alpha))
        self.alpha = best[2]
        self._is_fitted = True
        return self

    def score_vector(self, test_data):
        if not self._is_fitted:
            raise RuntimeError("ImprovedTDCA is not fitted. Please train weights first.")
        tdca_scores = np.asarray(self.tdca.score_vector(test_data), dtype=float)
        fbcca_scores = np.asarray(self.fbcca.score_vector(test_data), dtype=float)
        fused = _normalize_scores(tdca_scores)[0] + self.alpha * _normalize_scores(fbcca_scores)[0]
        return fused * self.frequency_weights


def _trca_filter(class_trials, n_samples):
    x = np.asarray(class_trials, dtype=float)[..., :n_samples].copy()
    x -= np.mean(x, axis=-1, keepdims=True)
    template = np.mean(x, axis=0)
    n_trials, n_channels, _ = x.shape
    s = np.zeros((n_channels, n_channels), dtype=float)
    q = np.zeros_like(s)
    for i in range(n_trials):
        q += x[i] @ x[i].T
        for j in range(i + 1, n_trials):
            s += x[i] @ x[j].T + x[j] @ x[i].T
    reg = 1e-6 * float(np.trace(q)) / max(n_channels, 1)
    if not np.isfinite(reg) or reg <= 0:
        reg = 1e-6
    values, vectors = eigh((s + s.T) / 2.0, (q + q.T) / 2.0 + reg * np.eye(n_channels), check_finite=False)
    w = vectors[:, int(np.argmax(values))]
    return w, template


class _FBTRCA:
    def __init__(self, sample_rate, n_classes, n_samples, n_bands=5):
        self.Fs = int(sample_rate)
        self.Nf = int(n_classes)
        self.T = int(n_samples)
        self.Nm = int(n_bands)
        self.filters = self._design_filter_bank()
        self.weights = _band_weights(self.Nm)

    def _design_filter_bank(self):
        nyq = self.Fs / 2.0
        pass_band = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stop_band = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        high_cut_pass = min(80, int(self.Fs * 0.45))
        high_cut_stop = min(90, int(self.Fs * 0.48))
        filters = []
        for i in range(self.Nm):
            wp = np.array([pass_band[i] / nyq, high_cut_pass / nyq], dtype=float)
            ws = np.array([stop_band[i] / nyq, high_cut_stop / nyq], dtype=float)
            order, wn = signal.cheb1ord(wp, ws, 3, 40)
            b, a = signal.cheby1(order, 0.5, wn, "bandpass")
            filters.append((b, a))
        return filters

    def _filter(self, sos_or_ba, x):
        b, a = sos_or_ba
        x = np.asarray(x, dtype=float)[..., :self.T]
        padlen = min(3 * (max(len(b), len(a)) - 1), max(1, x.shape[-1] - 1))
        return signal.filtfilt(b, a, x, axis=-1, padlen=padlen)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.models = []
        for filt in self.filters:
            xf = self._filter(filt, x)
            filters, templates = [], []
            for label in range(self.Nf):
                class_x = xf[y == label]
                if class_x.shape[0] == 0:
                    raise ValueError(f"missing class samples for label {label}")
                w, template = _trca_filter(class_x, self.T)
                filters.append(w)
                templates.append(template)
            self.models.append((np.asarray(filters), np.asarray(templates)))
        return self

    def score_vector(self, sample):
        sample = np.asarray(sample, dtype=float)
        total = np.zeros(self.Nf, dtype=float)
        for weight, filt, (filters, templates) in zip(self.weights, self.filters, self.models):
            xf = self._filter(filt, sample)
            scores = np.zeros(self.Nf, dtype=float)
            for label in range(self.Nf):
                w = filters[label]
                scores[label] = _corr(w @ xf, w @ templates[label])
            total += float(weight) * _normalize_scores(scores)[0]
        return total


class TriBranchTDCA(_BaseOptimizedDecoder):
    """TDCA + FBCCA + FBTRCA train-calibrated fusion.

    This was the strongest unified model on the full Benchmark for 1 s and 2 s
    windows.  It combines supervised TDCA templates, sinusoidal reference
    evidence, and cross-trial TRCA consistency.
    """

    def __init__(self, num_harmonics, times, targets, Nh=8, sample_rate=250):
        super().__init__(num_harmonics, times, targets, Nh=Nh, sample_rate=sample_rate)
        self.tdca = TDCA(num_harmonics, times, targets, Nh=Nh, sample_rate=sample_rate)
        self.tdca.n_components = 2
        self.fbcca = FBCCA(num_harmonics, times, targets, Nh=Nh, sample_rate=sample_rate)
        self.fbtrca = _FBTRCA(sample_rate=sample_rate, n_classes=self.Nf, n_samples=self.T, n_bands=5)
        self.required_points = max(
            int(getattr(self.tdca, "required_points", self.T)),
            int(getattr(self.fbcca, "T", self.T)),
            int(self.T),
        )
        self.alpha_fbcca = 0.2
        self.beta_fbtrca = 0.2

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.tdca.fit(x, y)
        self.fbtrca.fit(x, y)

        tdca_scores = np.asarray([self.tdca.score_vector(sample) for sample in x], dtype=float)
        fbcca_scores = np.asarray([self.fbcca.score_vector(sample) for sample in x], dtype=float)
        fbtrca_scores = np.asarray([self.fbtrca.score_vector(sample) for sample in x], dtype=float)

        best = (-1.0, -1.0, 0.0, 0.0)
        for alpha in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75):
            for beta in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75):
                fused = (
                    _normalize_scores(tdca_scores)
                    + alpha * _normalize_scores(fbcca_scores)
                    + beta * _normalize_scores(fbtrca_scores)
                )
                pred = np.argmax(fused, axis=1)
                acc = float(np.mean(pred == y))
                margins = np.sort(fused, axis=1)[:, -1] - np.sort(fused, axis=1)[:, -2]
                margin = float(np.mean(margins))
                if (acc, margin) > best[:2]:
                    best = (acc, margin, float(alpha), float(beta))
        self.alpha_fbcca = best[2]
        self.beta_fbtrca = best[3]
        self._is_fitted = True
        return self

    def score_vector(self, test_data):
        if not self._is_fitted:
            raise RuntimeError("TriBranchTDCA is not fitted. Please train weights first.")
        tdca_scores = np.asarray(self.tdca.score_vector(test_data), dtype=float)
        fbcca_scores = np.asarray(self.fbcca.score_vector(test_data), dtype=float)
        fbtrca_scores = np.asarray(self.fbtrca.score_vector(test_data), dtype=float)
        fused = (
            _normalize_scores(tdca_scores)[0]
            + self.alpha_fbcca * _normalize_scores(fbcca_scores)[0]
            + self.beta_fbtrca * _normalize_scores(fbtrca_scores)[0]
        )
        return fused * self.frequency_weights

