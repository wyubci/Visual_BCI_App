# -*- coding: utf-8 -*-
"""Offline evaluation of hc33 4s recorded data — handles multiple stimulus frequency groups.

Groups data by (n_timepoints, stimulus_frequencies) so all trials in a batch are the same shape.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.io import loadmat
from glob import glob
from collections import Counter, defaultdict

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

SUBJECT = "hc33"
COMMANDS = ["前进", "后退", "左转", "停止", "右转"]
SAMPLE_RATE = 250
DELAY_SEC = 0.14
NUM_HARMONICS_LIST = [3, 5]

TRAIN_ROOT = os.path.join("saveCarData", SUBJECT, "train")
SESSIONS = ["2026-05-28/4.00s", "2026-05-29/4.00s", "2026-05-30/4.00s", "2026-06-02/4.00s"]


def load_all_data():
    records = []
    for session in SESSIONS:
        full_dir = os.path.join(TRAIN_ROOT, session)
        if not os.path.isdir(full_dir):
            continue
        for fp in sorted(glob(os.path.join(full_dir, "*.mat"))):
            try:
                m = loadmat(fp)
                if "data" not in m or "label_idx" not in m:
                    continue
                label = int(np.array(m["label_idx"]).reshape(-1)[0])
                if label < 0 or label >= len(COMMANDS):
                    continue
                data = np.array(m["data"], dtype=float)
                if data.ndim != 2 or data.shape[0] < 3:
                    continue
                freqs_raw = np.array(m.get("stim_freqs_hz", []), dtype=float).flatten()
                if len(freqs_raw) != len(COMMANDS):
                    continue
                freq_key = tuple(np.round(freqs_raw, 3))
                n_pts = data.shape[-1]
                records.append({
                    "data": data,
                    "label": label,
                    "group_key": (n_pts, freq_key),
                    "freqs": freqs_raw,
                    "n_pts": n_pts,
                })
            except Exception:
                pass
    return records


def group_records(records):
    """Group by (n_pts, freq_key) so all trials have the same shape."""
    groups = defaultdict(list)
    for r in records:
        groups[r["group_key"]].append(r)
    return dict(groups)


def evaluate_group(group_recs, num_harmonics):
    """LOO-CV for TDCA, direct classify for FBCCA/CCA."""
    freqs = group_recs[0]["freqs"]
    samples = [r["data"] for r in group_recs]
    labels = np.array([r["label"] for r in group_recs], dtype=int)
    n = len(samples)
    n_pts = samples[0].shape[-1]
    model_times = n_pts / SAMPLE_RATE

    results = {}

    # FBCCA
    fbcca = FBCCA(num_harmonics, model_times, list(freqs), sample_rate=SAMPLE_RATE)
    fbcca_correct = 0
    fbcca_preds = []
    for data in samples:
        try:
            pred = fbcca.classify(data)
        except Exception:
            pred = -1
        fbcca_preds.append(pred)
        if pred == labels[len(fbcca_preds) - 1]:
            fbcca_correct += 1
    results["FBCCA"] = {"acc": fbcca_correct / n * 100, "preds": fbcca_preds}

    # CCA
    cca = CCA(num_harmonics, model_times, list(freqs), sample_rate=SAMPLE_RATE)
    cca_correct = 0
    cca_preds = []
    for data in samples:
        try:
            pred = cca.classify(data)
        except Exception:
            pred = -1
        cca_preds.append(pred)
        if pred == labels[len(cca_preds) - 1]:
            cca_correct += 1
    results["CCA"] = {"acc": cca_correct / n * 100, "preds": cca_preds}

    # TDCA (LOO-CV)
    tdca_correct = 0
    tdca_preds = []
    t0 = time.time()
    for i in range(n):
        train_x = [samples[j] for j in range(n) if j != i]
        train_y = np.array([labels[j] for j in range(n) if j != i], dtype=int)

        if len(set(train_y.tolist())) < 2:
            tdca_preds.append(-1)
            continue

        try:
            model = TDCA(num_harmonics, model_times, list(freqs),
                         sample_rate=SAMPLE_RATE, delay_sec=DELAY_SEC)
            model.fit(np.array(train_x, dtype=float), train_y)
            pred = model.classify(samples[i])
        except Exception:
            pred = -1
        tdca_preds.append(pred)
        if pred == labels[i]:
            tdca_correct += 1

        # Progress every 10% or 20 trials
        if (i + 1) % max(1, n // 10) == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"    TDCA LOO: {i+1}/{n} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    tdca_acc = tdca_correct / n * 100
    results["TDCA"] = {"acc": tdca_acc, "preds": tdca_preds}

    return results, labels, freqs


def per_class_accuracy(labels, preds):
    per_class = {}
    for ci in range(len(COMMANDS)):
        mask = labels == ci
        if mask.sum() > 0:
            per_class[ci] = (np.array(preds)[mask] == ci).sum() / mask.sum() * 100
        else:
            per_class[ci] = float("nan")
    return per_class


def confusion_matrix(labels, preds):
    cm = np.zeros((len(COMMANDS), len(COMMANDS)), dtype=int)
    for t, p in zip(labels, preds):
        if 0 <= t < len(COMMANDS) and 0 <= p < len(COMMANDS):
            cm[t, p] += 1
    return cm


def main():
    t_total = time.time()
    print("=" * 100, flush=True)
    print("hc33 4s Offline Evaluation", flush=True)
    print("=" * 100, flush=True)

    records = load_all_data()
    print(f"\nTotal valid records: {len(records)}", flush=True)

    groups = group_records(records)
    print(f"Groups (by n_pts + freq): {len(groups)}", flush=True)
    for (n_pts, fk), recs in sorted(groups.items()):
        lbl_dist = Counter(r["label"] for r in recs)
        freqs_short = [round(f, 1) for f in fk]
        print(f"  {n_pts}pts ({n_pts/250:.2f}s) {freqs_short}: {len(recs)} trials, labels={dict(sorted(lbl_dist.items()))}", flush=True)

    all_results = {}

    for num_harmonics in NUM_HARMONICS_LIST:
        print(f"\n{'=' * 100}", flush=True)
        print(f"RESULTS — num_harmonics = {num_harmonics}", flush=True)
        print(f"{'=' * 100}", flush=True)

        for (n_pts, freq_key), group_recs in sorted(groups.items()):
            freqs = group_recs[0]["freqs"]
            n = len(group_recs)
            lbl_dist = Counter(r["label"] for r in group_recs)

            print(f"\n--- {n_pts}pts ({n_pts/250:.2f}s) | freq={[round(f,2) for f in freqs]} | n={n} ---", flush=True)
            print(f"  Labels: {dict(sorted(lbl_dist.items()))}", flush=True)

            results, labels, _ = evaluate_group(group_recs, num_harmonics)

            # Table
            header = f"  {'Model':<8} {'Acc':>8}"
            for ci in range(len(COMMANDS)):
                header += f"  [{ci}]{COMMANDS[ci]:<4}"
            print(header, flush=True)
            print("  " + "-" * 82, flush=True)

            for model_name in ["FBCCA", "CCA", "TDCA"]:
                r = results[model_name]
                preds = np.array(r["preds"], dtype=int)
                acc = r["acc"]
                pc = per_class_accuracy(labels, preds)
                row = f"  {model_name:<8} {acc:7.2f}%"
                for ci in range(len(COMMANDS)):
                    row += f"  {pc.get(ci, float('nan')):5.1f}%"
                print(row, flush=True)

            # Confusion matrix
            cm = confusion_matrix(labels, np.array(results["TDCA"]["preds"], dtype=int))
            print(f"\n  TDCA Confusion Matrix (row=true, col=pred):", flush=True)
            hdr = "       " + " ".join(f"{COMMANDS[i]:>6}" for i in range(len(COMMANDS)))
            print(f"  {hdr}", flush=True)
            for i in range(len(COMMANDS)):
                row_cm = f"  {COMMANDS[i]:<4}  " + "".join(f"{cm[i, j]:>6d}" for j in range(len(COMMANDS)))
                print(row_cm, flush=True)

            key = f"h={num_harmonics} | {n_pts}pts | f={[round(f,2) for f in freqs]}"
            all_results[key] = {"results": results, "labels": labels.tolist(), "n": n}

    # Summary
    print(f"\n{'=' * 100}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 100}", flush=True)

    for num_harmonics in NUM_HARMONICS_LIST:
        print(f"\n  num_harmonics = {num_harmonics}:", flush=True)
        print(f"  {'Group':<50} {'FBCCA':>8} {'CCA':>8} {'TDCA':>8}", flush=True)
        print("  " + "-" * 78, flush=True)

        for (n_pts, freq_key), group_recs in sorted(groups.items()):
            freqs = group_recs[0]["freqs"]
            f_short = f"{n_pts}pts {[round(f,1) for f in freqs]}"
            n = len(group_recs)

            key = f"h={num_harmonics} | {n_pts}pts | f={[round(f,2) for f in freqs]}"
            if key in all_results:
                r = all_results[key]["results"]
                print(f"  {f_short:<50} {r['FBCCA']['acc']:7.2f}% {r['CCA']['acc']:7.2f}% {r['TDCA']['acc']:7.2f}%", flush=True)

    print(f"\nTotal time: {time.time() - t_total:.0f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
