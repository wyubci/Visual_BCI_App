# -*- coding: utf-8 -*-
"""Offline evaluation for car SSVEP training samples.

Use the same preprocessing and classifiers as the car control page.  The script
is meant for quick collection checks: collect a small batch, run this file, and
decide whether the data are separable before spending time on a full session.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from glob import glob

import numpy as np
from scipy.io import loadmat

from config import config
from interface.car_interface.acquisition import preprocess_model_input
from models.CCA import CCA
from models.FBCCA import FBCCA
from models.TDCA import TDCA


COMMANDS = ["前进", "后退", "左转", "停止", "右转"]
DEFAULT_FREQS = np.asarray([8.25, 9.16666667, 10.3125, 11.78571429, 13.75], dtype=float)


def _scalar(value, default):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    try:
        return type(default)(value)
    except Exception:
        return default


def _text(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in ("U", "S", "O"):
            flat = value.reshape(-1)
            return "".join(str(x) for x in flat).strip()
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value).strip()


def _subject_root(subject):
    return os.path.join("saveCarData", subject)


def _latest_data_dir(subject):
    pattern = os.path.join(_subject_root(subject), "train", "*", "*")
    candidates = []
    for folder in glob(pattern):
        if not os.path.isdir(folder):
            continue
        files = glob(os.path.join(folder, "*.mat"))
        if files:
            newest = max(os.path.getmtime(fp) for fp in files)
            candidates.append((newest, folder))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _load_samples(data_dir, use_raw=True):
    files = sorted(fp for fp in glob(os.path.join(data_dir, "*.mat")) if "bad_samples" not in fp)
    samples, labels, rows = [], [], []
    sample_rate, stim_sec, freqs = 250, 4.0, None

    for fp in files:
        mat = loadmat(fp)
        if "data" not in mat or "label_idx" not in mat:
            continue
        label_idx = _scalar(mat.get("label_idx", -1), -1)
        if label_idx < 0 or label_idx >= len(COMMANDS):
            continue

        sample_rate = _scalar(mat.get("sample_rate_hz", sample_rate), sample_rate)
        stim_sec = _scalar(mat.get("trial_stim_sec", stim_sec), stim_sec)
        freqs = np.asarray(mat.get("stim_freqs_hz", freqs if freqs is not None else DEFAULT_FREQS), dtype=float).reshape(-1)

        source = mat.get("raw_data", mat["data"]) if use_raw else mat["data"]
        source = np.asarray(source, dtype=float)
        sample = preprocess_model_input(source, sample_rate)
        if not isinstance(sample, np.ndarray) or sample.ndim != 2:
            continue

        samples.append(sample)
        labels.append(label_idx)
        rows.append(
            {
                "file": os.path.basename(fp),
                "label": COMMANDS[label_idx],
                "points": int(sample.shape[-1]),
                "sample_rate_hz": int(sample_rate),
                "trial_stim_sec": float(stim_sec),
                "input_quality_ok": _scalar(mat.get("input_quality_ok", -1), -1),
                "drop_ratio": _scalar(mat.get("drop_ratio", np.nan), float("nan")),
                "effective_sample_rate_hz": _scalar(mat.get("effective_sample_rate_hz", np.nan), float("nan")),
                "display_refresh_hz": _scalar(mat.get("display_refresh_hz", np.nan), float("nan")),
                "label_text": _text(mat.get("label_text", "")),
            }
        )

    return {
        "files": files,
        "samples": samples,
        "labels": np.asarray(labels, dtype=int),
        "rows": rows,
        "sample_rate": int(sample_rate),
        "stim_sec": float(stim_sec),
        "freqs": np.asarray(freqs if freqs is not None else DEFAULT_FREQS, dtype=float),
    }


def _confidence(scores):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size < 2:
        return float(scores[0]) if scores.size else 0.0
    top2 = np.partition(scores, -2)[-2:]
    return float(top2[1] - top2[0])


def _confusion(labels, preds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(labels, preds):
        if 0 <= int(truth) < n_classes and 0 <= int(pred) < n_classes:
            cm[int(truth), int(pred)] += 1
    return cm


def _eval_direct(name, model, samples, labels):
    preds, confs, details = [], [], []
    for sample, truth in zip(samples, labels):
        try:
            pred, scores, conf = model.classify_with_scores(sample)
        except Exception:
            pred, scores, conf = -1, np.zeros(len(COMMANDS)), 0.0
        preds.append(int(pred))
        confs.append(float(conf))
        details.append({"truth": int(truth), "pred": int(pred), "confidence": float(conf), "scores": np.asarray(scores, dtype=float).tolist()})
    return _result(name, labels, np.asarray(preds, dtype=int), confs, details)


def _eval_tdca_loo(samples, labels, stim_sec, freqs, sample_rate):
    counts = Counter(labels.tolist())
    if any(counts.get(i, 0) < 2 for i in range(len(COMMANDS))):
        return {
            "method": "TDCA_LOO",
            "ok": False,
            "message": "TDCA leave-one-out needs at least 2 samples per class.",
        }
    required = int(getattr(TDCA(3, stim_sec, freqs, sample_rate=sample_rate), "required_points", 1))
    if any(sample.shape[-1] < required for sample in samples):
        return {
            "method": "TDCA_LOO",
            "ok": False,
            "message": f"TDCA needs {required} points per sample; current data points are {sorted(set(x.shape[-1] for x in samples))}.",
        }

    preds, confs, details = [], [], []
    for test_idx, sample in enumerate(samples):
        train_x = [samples[i] for i in range(len(samples)) if i != test_idx]
        train_y = np.asarray([labels[i] for i in range(len(samples)) if i != test_idx], dtype=int)
        model = TDCA(3, stim_sec, freqs, sample_rate=sample_rate)
        try:
            train_x = [x[:, :required] for x in train_x if x.shape[-1] >= required]
            if len(train_x) != train_y.shape[0] or sample.shape[-1] < required:
                raise ValueError(f"sample shorter than TDCA required points: {required}")
            model.fit(np.asarray(train_x, dtype=float), train_y)
            pred, scores, conf = model.classify_with_scores(sample[:, :required])
        except Exception as exc:
            pred, scores, conf = -1, np.zeros(len(COMMANDS)), 0.0
            details.append({"truth": int(labels[test_idx]), "pred": -1, "confidence": 0.0, "error": str(exc)})
            preds.append(-1)
            confs.append(0.0)
            continue
        preds.append(int(pred))
        confs.append(float(conf))
        details.append({"truth": int(labels[test_idx]), "pred": int(pred), "confidence": float(conf), "scores": np.asarray(scores, dtype=float).tolist()})
    return _result("TDCA_LOO", labels, np.asarray(preds, dtype=int), confs, details)


def _result(name, labels, preds, confs, details):
    hits = preds == labels
    cm = _confusion(labels, preds, len(COMMANDS))
    return {
        "method": name,
        "ok": True,
        "accuracy": float(np.mean(hits) * 100.0) if labels.size else 0.0,
        "correct": int(np.sum(hits)),
        "total": int(labels.size),
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "confusion": cm.tolist(),
        "details": details,
    }


def _print_result(result):
    if not result.get("ok", False):
        print(f"\n{result['method']}: SKIPPED - {result.get('message', '')}")
        return
    print(
        f"\n{result['method']}: {result['accuracy']:.2f}% "
        f"({result['correct']}/{result['total']}), mean_conf={result['mean_confidence']:.4f}"
    )
    print("confusion rows=true cols=pred")
    print("      " + " ".join(f"{name[:2]:>3}" for name in COMMANDS))
    for idx, row in enumerate(result["confusion"]):
        print(f"{COMMANDS[idx]:>4} " + " ".join(f"{int(v):3d}" for v in row))
    for i, detail in enumerate(result["details"], start=1):
        truth = COMMANDS[detail["truth"]]
        pred = COMMANDS[detail["pred"]] if 0 <= detail["pred"] < len(COMMANDS) else "ERR"
        suffix = f" error={detail['error']}" if "error" in detail else ""
        print(f"  {i:02d}. truth={truth} pred={pred} hit={int(truth == pred)} conf={detail['confidence']:.4f}{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Offline car SSVEP data evaluator.")
    parser.add_argument("--subject", default=getattr(config, "subjectName", "TestSubject"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--processed-data", action="store_true", help="Use saved data instead of raw_data.")
    parser.add_argument("--save-report", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = args.data_dir or _latest_data_dir(args.subject)
    if not data_dir:
        raise FileNotFoundError(f"No training data found for subject {args.subject!r}")

    bundle = _load_samples(data_dir, use_raw=not args.processed_data)
    samples, labels = bundle["samples"], bundle["labels"]
    if len(samples) == 0:
        raise RuntimeError(f"No usable labeled samples in {data_dir}")

    freqs = bundle["freqs"]
    sample_rate = bundle["sample_rate"]
    stim_sec = bundle["stim_sec"]
    print(f"data_dir={os.path.abspath(data_dir)}")
    print(f"samples={len(samples)} labels={dict(Counter(labels.tolist()))}")
    print(f"sample_rate={sample_rate}Hz stim_sec={stim_sec:.3f}s points={sorted(set(x.shape[-1] for x in samples))}")
    print("freqs=" + " | ".join(f"{COMMANDS[i]}:{freqs[i]:.4f}Hz" for i in range(min(len(COMMANDS), freqs.size))))

    results = []
    results.append(_eval_direct("CCA", CCA(3, stim_sec, freqs, sample_rate=sample_rate), samples, labels))
    results.append(_eval_direct("FBCCA", FBCCA(3, stim_sec, freqs, sample_rate=sample_rate), samples, labels))
    results.append(_eval_tdca_loo(samples, labels, stim_sec, freqs, sample_rate))

    for result in results:
        _print_result(result)

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": os.path.abspath(data_dir),
        "subject": args.subject,
        "sample_rate_hz": sample_rate,
        "trial_stim_sec": stim_sec,
        "stim_freqs_hz": [float(x) for x in freqs],
        "samples": bundle["rows"],
        "results": results,
    }

    if args.save_report:
        out_dir = os.path.join(_subject_root(args.subject), "offline_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "offline_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        with open(out_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f"\nreport={os.path.abspath(out_file)}")


if __name__ == "__main__":
    main()
