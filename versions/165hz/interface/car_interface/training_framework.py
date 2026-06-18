# -*- coding: utf-8 -*-
import json
import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Callable, List, Sequence

import numpy as np
from scipy.io import loadmat


LEGACY_CAR_FREQS = np.array([6.67, 7.50, 8.57, 12.00, 15.00], dtype=float)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def _classifier_targets(classifier):
    try:
        return np.asarray(getattr(classifier, "targets", []), dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def _stim_freqs_compatible(mat, classifier):
    current = _classifier_targets(classifier)
    if current.size == 0:
        return True
    if not isinstance(mat, dict) or "stim_freqs_hz" not in mat:
        # Old data files did not store screen-adapted frequencies.  Only accept
        # them when the current profile is the legacy 60 Hz-compatible profile.
        return current.shape == LEGACY_CAR_FREQS.shape and np.allclose(current, LEGACY_CAR_FREQS, atol=0.05, rtol=0.0)
    try:
        saved = np.asarray(mat.get("stim_freqs_hz"), dtype=float).reshape(-1)
        return saved.shape == current.shape and np.allclose(saved, current, atol=0.03, rtol=0.0)
    except Exception:
        return False


def extract_label_text(label_val):
    if isinstance(label_val, np.ndarray):
        if label_val.size == 0:
            return ""
        label_val = label_val.reshape(-1)[0]
    if isinstance(label_val, bytes):
        return label_val.decode(errors="ignore")
    return str(label_val)


def extract_int(idx_val, default=-1):
    if isinstance(idx_val, np.ndarray):
        if idx_val.size == 0:
            return default
        idx_val = idx_val.reshape(-1)[0]
    try:
        return int(idx_val)
    except Exception:
        return default


def build_training_plan(total, labels_text, default_labels):
    total = int(total)
    labels = [x.strip() for x in str(labels_text).split(",") if len(x.strip()) > 0]
    if len(labels) == 0:
        labels = list(default_labels)
    plan = []
    while len(plan) < total:
        plan.extend(labels)
    return plan[:total]


@dataclass
class WeightTrainingResult:
    ok: bool
    message: str
    save_file: str = ""
    weights: np.ndarray = None
    class_score_scale: np.ndarray = None
    used_count: int = 0
    point_hist: dict = None


class CarTrainingFramework:
    def __init__(
        self,
        subject,
        commands: Sequence[str],
        sample_rate_hz: int,
        model_name: str,
        classifier,
        prepare_model_input: Callable,
        required_points_func: Callable[[], int],
    ):
        self.subject = subject
        self.commands = list(commands)
        self.sample_rate_hz = int(sample_rate_hz)
        self.model_name = str(model_name).upper()
        self.classifier = classifier
        self.prepare_model_input = prepare_model_input
        self.required_points_func = required_points_func

    def _needs_fit(self):
        return hasattr(self.classifier, "fit") and self.model_name in {
            "TDCA",
            "IMPROVEDTDCA",
            "TRIBRANCHTDCA",
        }

    def subject_root(self):
        return _project_path("saveCarData", self.subject)

    def train_root(self):
        return os.path.join(self.subject_root(), "train")

    def weights_root(self):
        return os.path.join(self.subject_root(), "weights")

    def load_training_matrix(self, file_list: List[str]):
        required_points = int(self.required_points_func())
        x_train, y_train, eval_samples = [], [], []
        label_counts = np.zeros(self.classifier.Nf, dtype=float)
        excluded_short = np.zeros(self.classifier.Nf, dtype=float)
        excluded_fs = np.zeros(self.classifier.Nf, dtype=float)
        point_hist = {}

        for fp in file_list:
            try:
                mat = loadmat(fp)
                if not _stim_freqs_compatible(mat, self.classifier):
                    continue
                data = mat.get("data", None)
                if not isinstance(data, np.ndarray) or data.ndim != 2:
                    continue

                label_idx = extract_int(mat.get("label_idx", -1))
                if label_idx < 0 or label_idx >= self.classifier.Nf:
                    continue

                pts = int(data.shape[-1])
                point_hist[pts] = point_hist.get(pts, 0) + 1

                file_sr = extract_int(mat.get("sample_rate_hz", -1))
                if file_sr > 0 and file_sr != self.sample_rate_hz:
                    excluded_fs[label_idx] += 1.0
                    continue
                if pts < required_points:
                    excluded_short[label_idx] += 1.0
                    continue

                sample = np.asarray(data[:, -required_points:], dtype=float)
                sample = self.prepare_model_input(sample)
                if not isinstance(sample, np.ndarray) or sample.ndim != 2:
                    continue

                x_train.append(sample)
                y_train.append(int(label_idx))
                eval_samples.append((int(label_idx), sample))
                label_counts[label_idx] += 1.0
            except Exception:
                continue

        return {
            "x": x_train,
            "y": y_train,
            "eval_samples": eval_samples,
            "label_counts": label_counts,
            "excluded_short": excluded_short,
            "excluded_fs": excluded_fs,
            "point_hist": point_hist,
            "required_points": required_points,
        }

    def fit_tdca_from_files(self, file_list):
        if not self._needs_fit():
            return True, "skip"
        if not isinstance(file_list, list) or len(file_list) == 0:
            return False, "缺少训练文件"

        matrix = self.load_training_matrix(file_list)
        if len(matrix["x"]) == 0:
            return False, f"无可用样本，需要 >= {matrix['required_points']} 点"

        labels = np.asarray(matrix["y"], dtype=int)
        missing = [self.commands[i] for i in range(self.classifier.Nf) if np.sum(labels == i) <= 0]
        if len(missing) > 0:
            return False, "缺少类别:" + ",".join(missing)

        try:
            self.classifier.fit(np.asarray(matrix["x"], dtype=float), labels)
            return True, f"样本{len(matrix['x'])}条"
        except Exception as exc:
            return False, str(exc)

    def train_weights(self, checked_files):
        matrix = self.load_training_matrix(checked_files)
        used = len(matrix["x"])
        if used == 0:
            return WeightTrainingResult(
                False,
                "未训练: 勾选数据与当前配置不兼容",
                point_hist=matrix["point_hist"],
            )

        label_counts = matrix["label_counts"]
        missing = [self.commands[i] for i in range(self.classifier.Nf) if label_counts[i] <= 0]
        if len(missing) > 0:
            return WeightTrainingResult(
                False,
                "未训练: 缺少类别样本 " + ",".join(missing),
                point_hist=matrix["point_hist"],
            )

        valid = label_counts > 0
        min_per_class = int(np.min(label_counts[valid])) if np.any(valid) else 0
        if min_per_class < 3:
            return WeightTrainingResult(False, f"未训练: 类别样本不均衡，最少类别仅{min_per_class}条，建议每类>=3")

        per_class_scores = [[] for _ in range(self.classifier.Nf)]

        if self._needs_fit():
            try:
                self.classifier.fit(np.asarray(matrix["x"], dtype=float), np.asarray(matrix["y"], dtype=int))
            except Exception as exc:
                return WeightTrainingResult(False, f"{self.model_name}拟合失败({str(exc)})")

        for label_idx, sample in matrix["eval_samples"]:
            try:
                scores = self.classifier.score_vector(sample)
                if isinstance(scores, np.ndarray) and scores.shape[0] == self.classifier.Nf:
                    per_class_scores[label_idx].append(np.asarray(scores, dtype=float))
            except Exception:
                continue

        class_mean_matrix = np.zeros((self.classifier.Nf, self.classifier.Nf), dtype=float)
        for idx in range(self.classifier.Nf):
            arr = np.asarray(per_class_scores[idx], dtype=float)
            if arr.ndim == 2 and arr.shape[0] > 0:
                class_mean_matrix[idx] = np.mean(arr, axis=0)

        diag = np.maximum(np.diag(class_mean_matrix), 1e-8)
        off_mean = np.maximum(
            (np.sum(class_mean_matrix, axis=1) - diag) / max(self.classifier.Nf - 1, 1),
            1e-8,
        )
        margin_ratio = diag / (off_mean + 1e-8)
        margin_ratio = margin_ratio / (np.mean(margin_ratio) + 1e-8)
        weights = np.clip(margin_ratio, 0.75, 1.25)
        class_score_scale = np.clip(diag / (np.mean(diag) + 1e-8), 0.80, 1.20)

        self.classifier.set_frequency_weights(weights)
        weights_root = self.weights_root()
        os.makedirs(weights_root, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(
            weights_root,
            f"car_{self.model_name.lower()}_{self.sample_rate_hz}hz_weights_{ts}.json",
        )
        payload = {
            "subject": self.subject,
            "created_at": ts,
            "model_name": self.model_name,
            "sample_rate_hz": self.sample_rate_hz,
            "stim_freqs_hz": [float(x) for x in _classifier_targets(self.classifier)],
            "weights": [float(x) for x in weights],
            "class_score_scale": [float(x) for x in class_score_scale],
            "commands": self.commands,
            "train_files": checked_files,
            "used_count": used,
        }
        with open(save_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return WeightTrainingResult(
            True,
            f"已训练并应用 ({os.path.basename(save_file)})",
            save_file=save_file,
            weights=weights,
            class_score_scale=class_score_scale,
            used_count=used,
            point_hist=matrix["point_hist"],
        )


def list_weight_files(weights_root):
    if not os.path.exists(weights_root):
        return []
    return sorted(glob(os.path.join(weights_root, "*.json")), reverse=True)
