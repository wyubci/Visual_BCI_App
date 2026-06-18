# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class AcquisitionConfig:
    sample_rate_hz: int = 250
    analysis_delay_sec: float = 0.0


class EegSampleBuffer:
    """Chunk based EEG buffer used by online and training trials."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._chunks = []
        self._points = 0

    def append(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            return False
        self._chunks.append(np.asarray(data, dtype=float))
        self._points += int(data.shape[-1])
        return True

    @property
    def points(self):
        return int(self._points)

    def materialize(self):
        if len(self._chunks) == 0:
            return np.array([])
        if len(self._chunks) == 1:
            return self._chunks[0]
        try:
            return np.concatenate(self._chunks, axis=-1)
        except Exception:
            return np.array([])


def preprocess_model_input(data, sample_rate_hz=250):
    """Prepare one EEG trial using the benchmark TDCA/FBCCA data convention.

    The benchmark scripts in this project keep acquisition preprocessing very
    conservative: trial windows are passed to the classifier after finite-value
    cleanup and per-channel DC removal.  Filter-bank decomposition, sinusoidal
    reference projection, temporal-delay augmentation, and discriminant spatial
    filtering are classifier-side steps, so they are intentionally not repeated
    here.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        return None
    arr = np.asarray(data, dtype=float)
    if arr.shape[-1] <= 1 or arr.shape[0] <= 0:
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - np.mean(arr, axis=-1, keepdims=True)
    return np.asarray(arr, dtype=float)


def aligned_window_indices(
    recv_start_mono,
    stim_onset_mono,
    sample_points,
    config: AcquisitionConfig,
) -> Optional[Tuple[int, int]]:
    try:
        req_points = int(sample_points)
        target_start = float(stim_onset_mono) + float(config.analysis_delay_sec)
        recv_start = float(recv_start_mono)
        if req_points <= 0 or not np.isfinite(target_start) or not np.isfinite(recv_start):
            return None
        start_idx = int(round((target_start - recv_start) * float(config.sample_rate_hz)))
        return start_idx, start_idx + req_points
    except Exception:
        return None


def extract_aligned_window(
    full_data,
    recv_start_mono,
    stim_onset_mono,
    sample_points,
    config: AcquisitionConfig,
):
    """Slice a trial window aligned to stimulus onset plus analysis delay."""
    if not isinstance(full_data, np.ndarray) or full_data.ndim != 2:
        return None
    n_points = int(full_data.shape[-1])
    if n_points < int(sample_points):
        return None

    idx = aligned_window_indices(recv_start_mono, stim_onset_mono, sample_points, config)
    if idx is None:
        return None
    start_idx, end_idx = idx
    if start_idx < 0 or end_idx > n_points:
        return None
    return np.asarray(full_data[:, start_idx:end_idx], dtype=float)


def update_receive_metadata(
    meta: Dict,
    data,
    now_mono: float,
    now_unix: float,
    sample_rate_hz: int,
):
    """Track receive timing from incoming chunks without touching UI state."""
    if not isinstance(meta, dict):
        return
    try:
        chunk_points = int(data.shape[-1])
    except Exception:
        chunk_points = 0

    chunk_dur = float(chunk_points) / max(float(sample_rate_hz), 1e-6)
    chunk_start_mono = float(now_mono - chunk_dur)
    chunk_start_unix = float(now_unix - chunk_dur)

    if "recv_start_monotonic" not in meta:
        meta["recv_start_monotonic"] = float(chunk_start_mono)
        meta["recv_start_unix"] = float(chunk_start_unix)
        meta["recv_chunks"] = 0
        meta["recv_points"] = 0
    else:
        meta["recv_start_monotonic"] = min(
            float(meta.get("recv_start_monotonic", chunk_start_mono)),
            float(chunk_start_mono),
        )
        meta["recv_start_unix"] = min(
            float(meta.get("recv_start_unix", chunk_start_unix)),
            float(chunk_start_unix),
        )

    meta["recv_chunks"] = int(meta.get("recv_chunks", 0)) + 1
    meta["recv_points"] = int(meta.get("recv_points", 0)) + int(chunk_points)

    recv_points = int(meta.get("recv_points", 0))
    recv_start_m = float(meta.get("recv_start_monotonic", chunk_start_mono))
    recv_start_u = float(meta.get("recv_start_unix", chunk_start_unix))
    recv_dur_est = float(recv_points) / max(float(sample_rate_hz), 1e-6)
    meta["recv_end_monotonic"] = float(recv_start_m + recv_dur_est)
    meta["recv_end_unix"] = float(recv_start_u + recv_dur_est)


def trial_quality_metrics(meta, raw_samples, analysis_samples, stim_onset, stim_end, sample_rate_hz):
    expected_samples = int(meta.get("expected_samples", analysis_samples))
    stim_dur = max(1e-6, float(stim_end) - float(stim_onset))
    effective_fs = float(raw_samples) / stim_dur
    drop_ratio = max(0.0, float(expected_samples - raw_samples) / max(expected_samples, 1))
    recv_start = float(meta.get("recv_start_monotonic", stim_onset))
    recv_end = float(meta.get("recv_end_monotonic", stim_end))
    recv_dur = max(0.0, recv_end - recv_start)
    input_ok = int(
        (int(raw_samples) >= int(0.95 * expected_samples))
        and (effective_fs >= 0.75 * float(sample_rate_hz))
    )
    return {
        "expected_samples": int(expected_samples),
        "analysis_samples": int(analysis_samples),
        "actual_samples": int(raw_samples),
        "effective_sample_rate_hz": float(effective_fs),
        "drop_ratio": float(drop_ratio),
        "recv_start_monotonic": float(recv_start),
        "recv_end_monotonic": float(recv_end),
        "recv_duration_sec": float(recv_dur),
        "input_quality_ok": int(input_ok),
    }
