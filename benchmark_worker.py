"""Importable worker module for benchmark evaluation (required for Windows spawn)."""
import os, warnings
import numpy as np
from scipy.io import loadmat

warnings.filterwarnings("ignore")

from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

# Paper-standard constants
BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14
PRE_STIMULUS = 0.5
NUM_HARMONICS = 5

# Correct frequency order for Tsinghua benchmark
TARGET_FREQS = []
for _offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for _base in range(8, 16):
        TARGET_FREQS.append(_base + _offset)

# Standard 10-20 occipital channel indices (0-based, Neuroscan 64-cap)
OCCIPITAL_CHANNELS = [45, 51, 52, 53, 54, 55, 58, 59, 60]


def extract_trial(data, tgt, block, data_length_sec, lagging_len=0, channels=None):
    onset = int((PRE_STIMULUS + VISUAL_DELAY) * SAMPLE_RATE)
    T = int(data_length_sec * SAMPLE_RATE)
    end = onset + T + max(0, int(lagging_len))
    trial = data[:, onset:end, tgt, block]
    if channels is not None:
        trial = trial[channels, :]
    return np.asarray(trial, dtype=float)


def evaluate_subject(args):
    """Evaluate one subject at all data lengths for all models.
    Args: (subj_id, data_lengths, model_types, channels)
    Returns: dict with subject, error, results keys.
    """
    subj_id, data_lengths, model_types, channels = args

    fpath = os.path.join(BENCHMARK_DIR, f"S{subj_id}.mat", f"S{subj_id}.mat")
    try:
        data = np.asarray(loadmat(fpath)["data"], dtype=float)
    except Exception as e:
        return {"subject": subj_id, "error": f"LOAD({type(e).__name__}): {e}", "results": {}}

    _, _, n_targets, n_blocks = data.shape
    results = {}

    for dl_sec in data_lengths:
        total_times = dl_sec + VISUAL_DELAY
        T = int(dl_sec * SAMPLE_RATE)

        for mt in model_types:
            correct, total = 0, 0

            if mt == "TDCA":
                temp = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                            sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
                lag_len = temp.lagging_len
                req_pts = temp.required_points

                for test_block in range(n_blocks):
                    train_blocks = [b for b in range(n_blocks) if b != test_block]

                    train_x, train_y = [], []
                    for tgt in range(n_targets):
                        for b in train_blocks:
                            train_x.append(extract_trial(data, tgt, b, dl_sec, lag_len, channels))
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
                        trial = extract_trial(data, tgt, test_block, dl_sec, lag_len, channels)
                        try:
                            pred = model.classify(trial)
                        except Exception:
                            pred = -1
                        if pred == tgt:
                            correct += 1
                        total += 1

            else:
                if mt == "FBCCA":
                    model = FBCCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                                  sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
                else:
                    model = CCA(NUM_HARMONICS, total_times, TARGET_FREQS,
                                sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)

                for tgt in range(n_targets):
                    for block in range(n_blocks):
                        trial = extract_trial(data, tgt, block, dl_sec, channels=channels)
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
