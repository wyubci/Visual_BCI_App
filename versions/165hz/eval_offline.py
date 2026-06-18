# -*- coding: utf-8 -*-
"""Offline accuracy evaluation on collected training data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.io import loadmat
from glob import glob
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score

from models.FBCCA import FBCCA
from models.CCA import CCA
from models.TDCA import TDCA
from models.OptimizedTDCA import ImprovedTDCA, TriBranchTDCA

SUBJECT = "hc33"
STI_LST = [6.0, 7.5, 10.0, 12.0, 15.0]
COMMANDS = ["前进", "后退", "左转", "停止", "右转"]
SAMPLE_RATE = 250
ANALYSIS_DELAY_SEC = 0.14
NUM_HARMONICS = 3


def extract_scalar(mat_dict, key, default=0):
    if key in mat_dict:
        return float(np.array(mat_dict[key]).reshape(-1)[0])
    return default


def infer_model_times(data_point_count, sample_rate=250, delay_sec=0.14):
    ws = data_point_count / sample_rate
    return ws + delay_sec


def load_labeled_data(data_root):
    files = sorted(glob(os.path.join(data_root, "**", "*.mat"), recursive=True))
    files = [f for f in files if "bad_samples" not in f.replace("\\", "/")]

    samples, labels, metas = [], [], []
    for fp in files:
        try:
            m = loadmat(fp)
            if "label_idx" not in m or "data" not in m:
                continue
            label_idx = int(np.array(m["label_idx"]).reshape(-1)[0])
            if label_idx < 0 or label_idx >= len(COMMANDS):
                continue

            data = np.array(m["data"], dtype=float)
            if data.ndim != 2 or data.shape[0] < 3:
                continue

            samples.append(data)
            labels.append(label_idx)

            analysis_sec = extract_scalar(m, "analysis_window_sec", -1)
            stim_sec = extract_scalar(m, "trial_stim_sec", -1)
            if analysis_sec <= 0:
                analysis_sec = data.shape[-1] / SAMPLE_RATE
            metas.append({
                "file": fp,
                "n_pts": data.shape[-1],
                "analysis_sec": analysis_sec,
                "stim_sec": stim_sec,
            })
        except Exception:
            continue

    return samples, np.array(labels, dtype=int), metas


def evaluate_model(model, samples, labels):
    correct = 0
    predictions = []
    for data, true_label in zip(samples, labels):
        try:
            pred = model.classify(data)
        except Exception:
            pred = -1
        predictions.append(pred)
        if pred == true_label:
            correct += 1

    predictions = np.array(predictions, dtype=int)
    total = len(labels)
    acc = correct / total * 100 if total > 0 else 0

    per_class = {}
    for i in range(len(COMMANDS)):
        mask = labels == i
        if mask.sum() > 0:
            per_class[i] = (predictions[mask] == i).sum() / mask.sum() * 100
        else:
            per_class[i] = float("nan")

    cm = np.zeros((len(COMMANDS), len(COMMANDS)), dtype=int)
    for t, p in zip(labels, predictions):
        if 0 <= t < len(COMMANDS) and 0 <= p < len(COMMANDS):
            cm[t, p] += 1

    return acc, per_class, cm, predictions


def evaluate_leave_one_out(model_factory, samples, labels):
    predictions = []
    for i in range(len(samples)):
        train_x = [samples[j] for j in range(len(samples)) if j != i]
        train_y = np.array([labels[j] for j in range(len(samples)) if j != i], dtype=int)
        try:
            model = model_factory()
            model.fit(np.array(train_x, dtype=float), train_y)
            pred = model.classify(samples[i])
        except Exception:
            pred = -1
        predictions.append(pred)

    predictions = np.array(predictions, dtype=int)
    acc = float(np.mean(predictions == labels) * 100.0) if len(labels) > 0 else 0.0
    per_class = {}
    for ci in range(len(COMMANDS)):
        mask = labels == ci
        per_class[ci] = (predictions[mask] == ci).sum() / mask.sum() * 100 if mask.sum() > 0 else float("nan")
    cm = np.zeros((len(COMMANDS), len(COMMANDS)), dtype=int)
    for t, p in zip(labels, predictions):
        if 0 <= t < len(COMMANDS) and 0 <= p < len(COMMANDS):
            cm[t, p] += 1
    return acc, per_class, cm, predictions


def print_confusion_matrix(name, cm, file=None):
    print(f"\n--- {name} ---", file=file)
    header = "real\\pred " + " ".join(f"[{i}]{COMMANDS[i]:<4}" for i in range(len(COMMANDS)))
    print(header, file=file)
    for i in range(len(COMMANDS)):
        row = f"[{i}]{COMMANDS[i]:<5} " + "".join(f"{cm[i,j]:>8d}" for j in range(len(COMMANDS)))
        print(row, file=file)


def main():
    data_root = os.path.join("saveCarData", SUBJECT, "train", "2026-05-22")
    print(f"Loading from: {data_root}")
    samples, labels, metas = load_labeled_data(data_root)
    # Exclude the earlier trial (22:50) — keep only the 20-sample batch (22:57-22:58)
    keep_idx = [i for i, m in enumerate(labels) if labels[i] != -1]
    # filter out 0001_前进_225032.mat style early trial files
    keep_idx = [i for i in range(len(samples)) if "225032" not in metas[i].get("file", "")]
    samples = [samples[i] for i in keep_idx]
    labels = labels[keep_idx]
    metas = [metas[i] for i in keep_idx]
    print(f"Loaded {len(samples)} samples (after filtering)")
    print(f"Label distribution: ", end="")
    dist = Counter(labels)
    print(", ".join(f"[{i}]{COMMANDS[i]}={dist.get(i,0)}" for i in range(len(COMMANDS))))

    # Group by data point count
    pt_groups = sorted(set(m["n_pts"] for m in metas))
    print(f"Data point counts: {pt_groups}")

    if len(pt_groups) > 1:
        print("\n=== NOTE: Multiple window sizes detected ===")
        for pts in pt_groups:
            indices = [j for j, m in enumerate(metas) if m["n_pts"] == pts]
            print(f"  {pts} points ({pts/SAMPLE_RATE:.1f}s): {len(indices)} samples")

    sample_pts = pt_groups[0]
    model_times = infer_model_times(sample_pts, SAMPLE_RATE, ANALYSIS_DELAY_SEC)
    print(f"\nModel params: times={model_times:.3f}s, T={int(SAMPLE_RATE * (model_times - ANALYSIS_DELAY_SEC))}")
    print()

    # Initialize models
    models = {
        "FBCCA": FBCCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
        "CCA": CCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
        "TDCA": TDCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
        "ImprovedTDCA": ImprovedTDCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
        "TriBranchTDCA": TriBranchTDCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
    }

    # Results header
    sep = "=" * 110
    header = f"{'Model':<14} {'Acc':>7} {'BAcc':>7} {'F1':>7}"
    for i in range(len(COMMANDS)):
        header += f" [{i}]{COMMANDS[i]:>5}"
    print(sep)
    print(header)
    print("-" * 110)

    all_cms = {}
    all_predictions = {}
    for name, model in models.items():
        if name in {"TDCA", "ImprovedTDCA", "TriBranchTDCA"}:
            acc, per_class, cm, predictions = evaluate_leave_one_out(
                lambda n=name: {
                    "TDCA": TDCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
                    "ImprovedTDCA": ImprovedTDCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
                    "TriBranchTDCA": TriBranchTDCA(NUM_HARMONICS, model_times, STI_LST, sample_rate=SAMPLE_RATE),
                }[n],
                samples,
                labels,
            )
        else:
            acc, per_class, cm, predictions = evaluate_model(model, samples, labels)

        all_cms[name] = cm
        all_predictions[name] = predictions

        bacc = balanced_accuracy_score(labels, predictions) * 100
        f1_macro = f1_score(labels, predictions, average="macro") * 100

        row = f"{name:<14} {acc:6.2f}% {bacc:6.2f}% {f1_macro:6.2f}%"
        for i in range(len(COMMANDS)):
            row += f" {per_class.get(i, float('nan')):6.1f}%"
        print(row)

    print("-" * 110)

    # Confusion matrices
    print("\n" + "=" * 85)
    print("Confusion Matrices (row=true, col=predicted)")
    print("=" * 85)

    for name in models:
        print_confusion_matrix(name, all_cms[name])

    # Save UTF-8 report
    report_path = os.path.join("saveCarData", SUBJECT, "accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Subject: {SUBJECT}\n")
        f.write(f"Date: 2026-05-22\n")
        f.write(f"Samples: {len(samples)}\n")
        f.write(f"Data points: {sample_pts} ({sample_pts/SAMPLE_RATE:.1f}s)\n")
        f.write(f"Model times: {model_times:.3f}s\n\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Model':<14} {'Acc':>7} {'BAcc':>7} {'F1':>7}")
        for i in range(len(COMMANDS)):
            f.write(f" {COMMANDS[i]:>7}")
        f.write("\n")
        f.write("-" * 110 + "\n")
        for name in models:
            cm = all_cms[name]
            predictions = all_predictions[name]
            acc = sum(cm[i, i] for i in range(len(COMMANDS))) / len(labels) * 100
            bacc = balanced_accuracy_score(labels, predictions) * 100
            f1_macro = f1_score(labels, predictions, average="macro") * 100
            row = f"{name:<14} {acc:6.2f}% {bacc:6.2f}% {f1_macro:6.2f}%"
            for i in range(len(COMMANDS)):
                mask = labels == i
                pc = cm[i, i] / mask.sum() * 100 if mask.sum() > 0 else float("nan")
                row += f" {pc:6.1f}%"
            f.write(row + "\n")
        f.write("\nConfusion Matrices:\n")
        for name in models:
            print_confusion_matrix(name, all_cms[name], file=f)

    print(f"\nReport saved to: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
