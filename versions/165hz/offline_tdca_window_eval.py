import argparse
import os
from datetime import datetime
from glob import glob

import numpy as np
from scipy.io import loadmat

from config import config
from interface.car_interface.acquisition import preprocess_model_input
from interface.car_interface.training_framework import extract_int
from models.CCA import CCA
from models.FBCCA import FBCCA
from models.TDCA import TDCA


COMMANDS = ["前进", "后退", "左转", "停止", "右转"]
TARGET_FREQS = [6.67, 7.5, 8.57, 12.0, 15.0]
SAMPLE_RATE = 250


def _safe_float(mat, key, default=-1.0):
    try:
        return float(np.asarray(mat.get(key, [[default]])).reshape(-1)[0])
    except Exception:
        return float(default)


def _load_rows(files, required_points):
    rows = []
    for fp in files:
        try:
            mat = loadmat(fp)
            data = mat.get("data", None)
            label_idx = extract_int(mat.get("label_idx", -1))
            sample_rate = extract_int(mat.get("sample_rate_hz", SAMPLE_RATE))
            if sample_rate != SAMPLE_RATE:
                continue
            if label_idx < 0 or label_idx >= len(COMMANDS):
                continue
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                continue
            if data.shape[-1] < required_points:
                continue
            sample = preprocess_model_input(np.asarray(data[:, -required_points:], dtype=float))
            if not isinstance(sample, np.ndarray) or sample.ndim != 2:
                continue
            rows.append({"fp": fp, "label": int(label_idx), "sample": sample})
        except Exception as exc:
            print(f"[OFFLINE_EVAL] skip {os.path.basename(fp)}: {exc}", flush=True)
    return rows


def _predict_margin(model, sample):
    try:
        scores = np.asarray(model.score_vector(sample), dtype=float).reshape(-1)
        if scores.size == 0:
            return -1, 0.0
        pred = int(np.argmax(scores))
        if scores.size >= 2:
            top2 = np.partition(scores, -2)[-2:]
            margin = float(top2[1] - top2[0])
        else:
            margin = float(scores[0])
        return pred, margin
    except Exception:
        return -1, 0.0


def _accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    valid = y_pred >= 0
    if not np.any(valid):
        return 0.0, 0, int(y_true.size)
    correct = int(np.sum(y_true[valid] == y_pred[valid]))
    total = int(np.sum(valid))
    return 100.0 * correct / max(total, 1), correct, total


def _loo_tdca(rows, model_times):
    preds = []
    for i, row in enumerate(rows):
        train_rows = [r for j, r in enumerate(rows) if j != i]
        x_train = np.asarray([r["sample"] for r in train_rows], dtype=float)
        y_train = np.asarray([r["label"] for r in train_rows], dtype=int)
        try:
            model = TDCA(3, model_times, TARGET_FREQS, sample_rate=SAMPLE_RATE)
            model.fit(x_train, y_train)
            pred, _ = _predict_margin(model, row["sample"])
        except Exception as exc:
            pred = -1
            print(f"[OFFLINE_EVAL] TDCA LOO fail {i + 1}/{len(rows)}: {exc}", flush=True)
        preds.append(int(pred))
        print(f"[OFFLINE_EVAL] TDCA {model_times:.2f}s LOO {i + 1}/{len(rows)}", flush=True)
    return preds


def _fit_predict_tdca(rows, model_times):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    samples = np.asarray([row["sample"] for row in rows], dtype=float)
    model = TDCA(3, model_times, TARGET_FREQS, sample_rate=SAMPLE_RATE)
    model.fit(samples, labels)
    preds = []
    for i, row in enumerate(rows):
        pred, _ = _predict_margin(model, row["sample"])
        preds.append(int(pred))
        print(f"[OFFLINE_EVAL] TDCA {model_times:.2f}s train-predict {i + 1}/{len(rows)}", flush=True)
    return preds


def _eval_tdca(rows, model_times, mode):
    if mode == "loo":
        return _loo_tdca(rows, model_times)
    return _fit_predict_tdca(rows, model_times)


def _eval_reference_models(rows, model_times):
    fbcca = FBCCA(3, model_times, TARGET_FREQS, sample_rate=SAMPLE_RATE)
    cca = CCA(3, model_times, TARGET_FREQS, sample_rate=SAMPLE_RATE)
    fb_preds, cca_preds = [], []
    for i, row in enumerate(rows):
        fb_pred, _ = _predict_margin(fbcca, row["sample"])
        cca_pred, _ = _predict_margin(cca, row["sample"])
        fb_preds.append(int(fb_pred))
        cca_preds.append(int(cca_pred))
        print(f"[OFFLINE_EVAL] FBCCA/CCA {model_times:.2f}s {i + 1}/{len(rows)}", flush=True)
    return fb_preds, cca_preds


def _eval_window_models(rows, model_times, tdca_mode):
    y_true = [r["label"] for r in rows]
    fb_preds, cca_preds = _eval_reference_models(rows, model_times)
    tdca_preds = _eval_tdca(rows, model_times, tdca_mode)
    out = {}
    for name, preds in [("FBCCA", fb_preds), ("CCA", cca_preds), ("TDCA", tdca_preds)]:
        acc, correct, total = _accuracy(y_true, preds)
        out[name] = {"acc": acc, "correct": correct, "total": total, "preds": preds}
    return out


def _slice_rows(rows, start_points, window_points):
    sliced = []
    for row in rows:
        sample = row["sample"]
        end = int(start_points + window_points)
        if sample.shape[-1] < end:
            continue
        sliced.append({
            "fp": row["fp"],
            "label": row["label"],
            "sample": np.asarray(sample[:, start_points:end], dtype=float),
        })
    return sliced


def _print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate today's 4s SSVEP samples with FBCCA/CCA/TDCA and TDCA sliding windows.")
    parser.add_argument("--subject", default=config.subjectName or "hc33")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--root", default="saveCarData")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--step-sec", type=float, default=0.25)
    parser.add_argument("--tdca-mode", choices=["train", "loo"], default="train")
    parser.add_argument("--eval-windows", default="4,3,2,1")
    parser.add_argument("--position", choices=["end", "start"], default="end")
    args = parser.parse_args()

    data_dir = os.path.join(args.root, args.subject, "train", args.date, "4.00s")
    files = sorted(glob(os.path.join(data_dir, "*.mat")))
    if args.limit > 0:
        files = files[-args.limit:]
    if len(files) == 0:
        raise SystemExit(f"No .mat files found in {data_dir}")

    full_points = int(round(4.0 * SAMPLE_RATE))
    rows_4s = _load_rows(files, full_points)
    print(f"[OFFLINE_EVAL] loaded rows={len(rows_4s)} dir={data_dir}", flush=True)

    window_values = []
    for part in str(args.eval_windows).split(","):
        part = part.strip().lower().replace("s", "")
        if part:
            window_values.append(float(part))
    window_values = sorted(set(window_values), reverse=True)

    summary = []
    last_detail = None
    for window_sec in window_values:
        window_points = int(round(window_sec * SAMPLE_RATE))
        start = full_points - window_points if args.position == "end" else 0
        start = max(0, int(start))
        start_sec = start / SAMPLE_RATE
        sliced = _slice_rows(rows_4s, start, window_points)
        if len(sliced) != len(rows_4s):
            continue
        result = _eval_window_models(sliced, window_sec, args.tdca_mode)
        summary.append([
            f"{window_sec:.2f}s",
            f"{start_sec:.2f}s",
            f"{result['FBCCA']['acc']:.2f}",
            f"{result['FBCCA']['correct']}/{result['FBCCA']['total']}",
            f"{result['CCA']['acc']:.2f}",
            f"{result['CCA']['correct']}/{result['CCA']['total']}",
            f"{result['TDCA']['acc']:.2f}",
            f"{result['TDCA']['correct']}/{result['TDCA']['total']}",
        ])
        if abs(window_sec - 4.0) < 1e-6:
            last_detail = (sliced, result)

    print()
    _print_table(["窗口", "起点", "FBCCA ACC", "FBCCA", "CCA ACC", "CCA", "TDCA ACC", "TDCA"], summary)

    if last_detail is not None:
        print()
        detail_rows = []
        detail_rows_source, detail_result = last_detail
        fb_preds = detail_result["FBCCA"]["preds"]
        cca_preds = detail_result["CCA"]["preds"]
        tdca_preds = detail_result["TDCA"]["preds"]
        for i, row in enumerate(detail_rows_source):
            detail_rows.append([
                os.path.basename(row["fp"]),
                COMMANDS[row["label"]],
                COMMANDS[fb_preds[i]] if 0 <= fb_preds[i] < len(COMMANDS) else "-",
                COMMANDS[cca_preds[i]] if 0 <= cca_preds[i] < len(COMMANDS) else "-",
                COMMANDS[tdca_preds[i]] if 0 <= tdca_preds[i] < len(COMMANDS) else "-",
            ])
        _print_table(["文件", "真实", "FBCCA4s", "CCA4s", "TDCA4s"], detail_rows)


if __name__ == "__main__":
    main()
