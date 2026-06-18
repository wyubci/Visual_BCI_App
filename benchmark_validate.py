"""Quick validation: TDCA/FBCCA/CCA on S1 only."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
TARGET_FREQS = [round(8.0 + 0.2 * i, 2) for i in range(40)]


def extract_trial(data, tgt, block, data_length_sec, lagging_len=0):
    onset = int((PRE_STIMULUS + VISUAL_DELAY) * SAMPLE_RATE)
    T = int(data_length_sec * SAMPLE_RATE)
    end = onset + T + max(0, int(lagging_len))
    return data[:, onset:end, tgt, block].copy()


def main():
    print("Loading S1...", flush=True)
    data = np.array(loadmat(os.path.join(BENCHMARK_DIR, "S1.mat", "S1.mat"))["data"], dtype=float)
    print(f"Data shape: {data.shape}", flush=True)
    n_targets, n_blocks = 40, 6

    for dl_sec in [0.5, 1.0]:
        total_times = dl_sec + VISUAL_DELAY
        T = int(dl_sec * SAMPLE_RATE)
        print(f"\n{'='*50}", flush=True)
        print(f"Data length: {dl_sec}s (T={T}, total_times={total_times:.2f}s)", flush=True)

        # --- TDCA ---
        t0 = time.time()
        correct, total = 0, 0
        for test_block in range(n_blocks):
            train_blocks = [b for b in range(n_blocks) if b != test_block]

            temp = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
            lag_len, req_pts = temp.lagging_len, temp.required_points

            train_x, train_y = [], []
            for tgt in range(n_targets):
                for b in train_blocks:
                    train_x.append(extract_trial(data, tgt, b, dl_sec, lag_len))
                    train_y.append(tgt)
            train_x = np.stack(train_x, axis=0)
            train_y = np.array(train_y, dtype=int)

            model = TDCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
            model.fit(train_x, train_y)

            for tgt in range(n_targets):
                trial = extract_trial(data, tgt, test_block, dl_sec, lag_len)
                try:
                    pred = model.classify(trial)
                except Exception:
                    pred = -1
                if pred == tgt:
                    correct += 1
                total += 1

        tdca_acc = correct / total * 100 if total > 0 else 0
        print(f"TDCA:   {tdca_acc:.2f}% ({correct}/{total}) [{time.time()-t0:.1f}s]", flush=True)

        # --- FBCCA ---
        t0 = time.time()
        fbcca = FBCCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
        correct, total = 0, 0
        for tgt in range(n_targets):
            for block in range(n_blocks):
                trial = extract_trial(data, tgt, block, dl_sec)
                pred = fbcca.classify(trial)
                if pred == tgt:
                    correct += 1
                total += 1
        fbcca_acc = correct / total * 100 if total > 0 else 0
        print(f"FBCCA:  {fbcca_acc:.2f}% ({correct}/{total}) [{time.time()-t0:.1f}s]", flush=True)

        # --- CCA ---
        t0 = time.time()
        cca = CCA(NUM_HARMONICS, total_times, TARGET_FREQS, sample_rate=SAMPLE_RATE, delay_sec=VISUAL_DELAY)
        correct, total = 0, 0
        for tgt in range(n_targets):
            for block in range(n_blocks):
                trial = extract_trial(data, tgt, block, dl_sec)
                pred = cca.classify(trial)
                if pred == tgt:
                    correct += 1
                total += 1
        cca_acc = correct / total * 100 if total > 0 else 0
        print(f"CCA:    {cca_acc:.2f}% ({correct}/{total}) [{time.time()-t0:.1f}s]", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
