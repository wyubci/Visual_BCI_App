"""Quick S1-only validation with 9 occipital channels."""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import loadmat
from models.TDCA import TDCA
from models.FBCCA import FBCCA
from models.CCA import CCA

BENCHMARK_DIR = r"C:\Users\adam\Desktop\benchmark"
SAMPLE_RATE = 250
VISUAL_DELAY = 0.14
PRE_STIMULUS = 0.5
NUM_HARMONICS = 5
TARGET_FREQS = []
for offset in [0, 0.2, 0.4, 0.6, 0.8]:
    for base in range(8, 16):
        TARGET_FREQS.append(base + offset)

# 9 occipital channels (PZ, PO5, PO3, POZ, PO4, PO6, O1, OZ, O2)
CHANNELS = [46, 52, 53, 54, 55, 56, 58, 59, 60]

def extract_trial(data, tgt, block, data_length_sec, lagging_len=0):
    onset = int((PRE_STIMULUS + VISUAL_DELAY) * SAMPLE_RATE)
    T = int(data_length_sec * SAMPLE_RATE)
    end = onset + T + max(0, int(lagging_len))
    trial = data[CHANNELS, onset:end, tgt, block]
    return np.asarray(trial, dtype=float)

print("Loading S1 with 9 occipital channels...", flush=True)
data = np.array(loadmat(os.path.join(BENCHMARK_DIR, "S1.mat", "S1.mat"))["data"], dtype=float)
n_targets, n_blocks = 40, 6

for dl_sec in [0.3, 0.5, 0.7, 1.0]:
    total_times = dl_sec + VISUAL_DELAY
    T = int(dl_sec * SAMPLE_RATE)
    print(f"\n--- {dl_sec}s (T={T}) ---", flush=True)

    # FBCCA
    t0 = time.time()
    fbcca = FBCCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
    correct = sum(1 for tgt in range(n_targets) for b in range(n_blocks)
                  if fbcca.classify(extract_trial(data, tgt, b, dl_sec)) == tgt)
    print(f"FBCCA:  {correct/240*100:.2f}% ({correct}/240) [{time.time()-t0:.1f}s]", flush=True)

    # CCA
    t0 = time.time()
    cca = CCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
    correct = sum(1 for tgt in range(n_targets) for b in range(n_blocks)
                  if cca.classify(extract_trial(data, tgt, b, dl_sec)) == tgt)
    print(f"CCA:    {correct/240*100:.2f}% ({correct}/240) [{time.time()-t0:.1f}s]", flush=True)

    # TDCA
    t0 = time.time()
    temp = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
    lag_len, req_pts = temp.lagging_len, temp.required_points
    print(f"TDCA:   lag={lag_len}, req_pts={req_pts}", flush=True)
    correct = 0
    for test_block in range(n_blocks):
        train_blocks = [b for b in range(n_blocks) if b != test_block]
        train_x = np.stack([extract_trial(data, tgt, b, dl_sec, lag_len)
                           for tgt in range(n_targets) for b in train_blocks], axis=0)
        train_y = np.array([tgt for tgt in range(n_targets) for b in train_blocks], dtype=int)

        model = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
        model.fit(train_x, train_y)
        for tgt in range(n_targets):
            if model.classify(extract_trial(data, tgt, test_block, dl_sec, lag_len)) == tgt:
                correct += 1
    print(f"TDCA:   {correct/240*100:.2f}% ({correct}/240) [{time.time()-t0:.1f}s]", flush=True)

print("\nDone.", flush=True)
