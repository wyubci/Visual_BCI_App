# -*- coding: utf-8 -*-
"""Full benchmark evaluation: TDCA, FBCCA, CCA on all S1-S35 subjects.

Paper-standard processing:
  - Visual delay: 0.14s (SSVEP latency)
  - Pre-stimulus baseline: 0.5s
  - Harmonics: 5
  - Filter bank: 8 sub-bands Chebyshev Type I
  - Sub-band weights: (i+1)^(-1.25) + 0.25
  - Leave-one-block-out cross-validation
  - Parallel across subjects via ProcessPoolExecutor
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

# ---------------------------------------------------------------------------
# Configuration (paper standard)
# ---------------------------------------------------------------------------
BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14       # SSVEP visual latency (seconds)
PRE_STIMULUS = 0.5         # pre-stimulus baseline (seconds)
NUM_HARMONICS = 5          # harmonics for reference signals
N_JOBS = 8                 # parallel workers

# Tsinghua benchmark: 40-class SSVEP
# Frequencies: 8.0, 8.2, 8.4, ..., 15.8 Hz
# Correct Tsinghua benchmark order: 5 groups of 8 freqs
TARGET_FREQS = []
for _offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for _base in range(8, 16):
        TARGET_FREQS.append(_base + _offset)

# Data lengths to evaluate (seconds)
DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_trial(data, tgt, block, data_length_sec, lagging_len=0):
    """Extract EEG segment with paper-standard onset alignment.

    Onset = pre-stimulus (0.5s) + visual delay (0.14s) = sample 160 at 250Hz.
    """
    onset = int((PRE_STIMULUS + VISUAL_DELAY) * SAMPLE_RATE)
    T = int(data_length_sec * SAMPLE_RATE)
    end = onset + T + max(0, int(lagging_len))
    return np.asarray(data[:, onset:end, tgt, block], dtype=float)


def compute_itr(acc, n_targets, data_length_sec, gap_sec=0.5):
    """Information Transfer Rate in bits/min.

    ITR = (log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))) * 60 / T
    where T = data_length + gap between trials.
    """
    N = n_targets
    P = max(acc / 100.0, 1.0 / N)  # clamp to chance level
    T = data_length_sec + gap_sec
    if P >= 1.0:
        return N * np.log2(N) * 60.0 / T  # perfect accuracy upper bound
    if P <= 1.0 / N:
        return 0.0
    itr = (np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))) * 60.0 / T
    return max(0.0, itr)


# ---------------------------------------------------------------------------
# Subject-level evaluator (called by worker processes)
# ---------------------------------------------------------------------------

def _eval_one_subject(args):
    """Picklable worker: evaluates one subject at all data lengths for all models."""
    subj_id, data_lengths, model_types = args

    # Load data
    fpath = os.path.join(BENCHMARK_DIR, f"S{subj_id}.mat", f"S{subj_id}.mat")
    try:
        data = np.array(loadmat(fpath)["data"], dtype=float)
    except Exception as e:
        return {"subject": subj_id, "error": f"LOAD: {e}", "results": {}}

    _, _, n_targets, n_blocks = data.shape
    results = {}

    for dl_sec in data_lengths:
        total_times = dl_sec + VISUAL_DELAY
        T = int(dl_sec * SAMPLE_RATE)

        for mt in model_types:
            correct, total = 0, 0

            if mt == "TDCA":
                # ---- TDCA with leave-one-block-out CV ----
                # Pre-compute lag params
                temp = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                            sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
                lag_len = temp.lagging_len
                req_pts = temp.required_points

                for test_block in range(n_blocks):
                    train_blocks = [b for b in range(n_blocks) if b != test_block]

                    # Build training set
                    train_x, train_y = [], []
                    for tgt in range(n_targets):
                        for b in train_blocks:
                            trial = extract_trial(data, tgt, b, dl_sec, lag_len)
                            train_x.append(trial)
                            train_y.append(tgt)

                    train_x = np.stack(train_x, axis=0)
                    train_y = np.array(train_y, dtype=int)

                    model = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                                 sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
                    try:
                        model.fit(train_x, train_y)
                    except Exception:
                        continue

                    for tgt in range(n_targets):
                        trial = extract_trial(data, tgt, test_block, dl_sec, lag_len)
                        try:
                            pred = model.classify(trial)
                        except Exception:
                            pred = -1
                        if pred == tgt:
                            correct += 1
                        total += 1

            else:
                # ---- FBCCA / CCA: reference-based, no training needed ----
                if mt == "FBCCA":
                    model = FBCCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                                  sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
                else:
                    model = CCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                                sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)

                for tgt in range(n_targets):
                    for block in range(n_blocks):
                        trial = extract_trial(data, tgt, block, dl_sec)
                        try:
                            pred = model.classify(trial)
                        except Exception:
                            pred = -1
                        if pred == tgt:
                            correct += 1
                        total += 1

            acc = correct / total * 100.0 if total > 0 else 0.0
            results[(mt, dl_sec)] = acc

    return {"subject": subj_id, "error": None, "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    subjects = list(range(1, 36))
    model_types = ["TDCA", "FBCCA", "CCA"]

    print("=" * 80, flush=True)
    print("TDCA / FBCCA / CCA — Benchmark Full Evaluation", flush=True)
    print(f"Subjects: S1-S35 | Data: {DATA_LENGTHS} | Targets: {len(TARGET_FREQS)}", flush=True)
    print(f"Harmonics: {NUM_HARMONICS} | Delay: {VISUAL_DELAY}s | CV: L1BO (6 blocks)", flush=True)
    print(f"Workers: {N_JOBS}", flush=True)
    print("=" * 80, flush=True)

    # Accumulator
    all_accs = OrderedDict()
    for mt in model_types:
        for dl in DATA_LENGTHS:
            all_accs[(mt, dl)] = []

    t_start = time.time()
    completed = 0

    tasks = [(sid, DATA_LENGTHS, model_types) for sid in subjects]

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(_eval_one_subject, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            sid = result["subject"]
            completed += 1

            if result["error"]:
                print(f"[{completed:2d}/35] S{sid:02d}: {result['error']}", flush=True)
            else:
                parts = []
                for (mt, dl), acc in sorted(result["results"].items()):
                    all_accs[(mt, dl)].append(acc)
                    parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
                et = time.time() - t_start
                print(f"[{completed:2d}/35] S{sid:02d}: " + " | ".join(parts) + f"  ({et:.0f}s)", flush=True)

    total_time = time.time() - t_start

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: Mean accuracy across S1-S35", flush=True)
    print("=" * 80, flush=True)

    print(f"{'Model':<8} {'0.3s':>10} {'0.5s':>10} {'0.7s':>10} {'1.0s':>10}  {'N':>4}", flush=True)
    print("-" * 62, flush=True)

    summary = {}
    for mt in model_types:
        row = f"{mt:<8}"
        for dl in DATA_LENGTHS:
            accs = all_accs[(mt, dl)]
            if accs:
                m = np.mean(accs)
                s = np.std(accs)
                itr = compute_itr(m, len(TARGET_FREQS), dl)
                summary[(mt, dl)] = {"mean": m, "std": s, "n": len(accs), "itr": itr, "accs": list(accs)}
                row += f" {m:7.2f}%±{s:.1f}"
            else:
                summary[(mt, dl)] = {"mean": 0, "std": 0, "n": 0, "itr": 0, "accs": []}
                row += f" {'--':>10}"
        n_vals = [len(all_accs[(mt, dl)]) for dl in DATA_LENGTHS if all_accs[(mt, dl)]]
        n_str = str(n_vals[0]) if len(set(n_vals)) == 1 and n_vals else "?"
        row += f"  {n_str:>4}"
        print(row, flush=True)

    # -------------------------------------------------------------------
    # Detailed report
    # -------------------------------------------------------------------
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"\n{'='*80}", flush=True)
    print("PER-SUBJECT DETAILS", flush=True)
    print(f"{'='*80}", flush=True)

    # Sort subjects by TDCA@1.0s accuracy for readability
    tdca_1s = all_accs[("TDCA", 1.0)]
    if tdca_1s:
        sorted_idx = np.argsort(tdca_1s)[::-1]
        for rank, idx in enumerate(sorted_idx):
            sid = idx + 1
            accs_str = " | ".join(f"{mt}={all_accs[(mt, dl)][idx]:.2f}%"
                                  for mt in model_types for dl in DATA_LENGTHS)
            print(f"  {rank+1:2d}. S{sid:02d}: {accs_str}", flush=True)

    print(f"\n{'='*80}", flush=True)
    for mt in model_types:
        for dl in DATA_LENGTHS:
            info = summary[(mt, dl)]
            if info["n"] > 0:
                print(f"{mt} @ {dl:.1f}s: mean={info['mean']:.2f}% ±{info['std']:.2f}%  "
                      f"min={min(info['accs']):.2f}%  max={max(info['accs']):.2f}%  "
                      f"ITR={info['itr']:.1f} bits/min", flush=True)

    # -------------------------------------------------------------------
    # Save results to JSON for later use
    # -------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_baseline_results.json")
    # Convert to serializable format
    serializable = {}
    for (mt, dl), info in summary.items():
        serializable[f"{mt}_{dl}s"] = {
            "mean": info["mean"], "std": info["std"],
            "n": info["n"], "itr": info["itr"]
        }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {out_path}", flush=True)

    return summary, total_time


if __name__ == "__main__":
    main()
