"""Compare offline simulation vs online pipeline on the SAME data."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, os, traceback
from scipy.io import loadmat
from models.FBCCA import FBCCA

SAMPLE_RATE = 250
COMMANDS_EN = ['forward', 'backward', 'left', 'stop', 'right']

wf = r'saveCarData\hc33\weights\car_tdca_250hz_weights_20260602_211526.json'
with open(wf, encoding='utf-8') as f:
    cfg = json.load(f)
freqs = np.array(cfg['stim_freqs_hz'], dtype=float)
fp = cfg['train_files'][0]
m = loadmat(fp)
data_full = np.array(m['data'], dtype=float)
label = int(np.array(m['label_idx']).reshape(-1)[0])

print('Data shape:', data_full.shape)
print('True label:', label, '=', COMMANDS_EN[label])
print('Freqs:', [round(f, 2) for f in freqs])
print()

# ===== OFFLINE: 4s single window =====
TRIM = 160
data_off = data_full[:, TRIM:]  # (8, 1008)
total_times = (data_off.shape[1] - 8) / 250 + 0.14  # = 4.14
fbcca_off = FBCCA(3, total_times, list(freqs), sample_rate=SAMPLE_RATE)
arr = np.asarray(data_off, dtype=float)
arr = arr - np.mean(arr, axis=1, keepdims=True)
arr = arr / (np.std(arr, axis=1, keepdims=True) + 1e-6)
arr = np.clip(arr, -6.0, 6.0)
pred_off, scores_off, _ = fbcca_off.classify_with_scores(arr)
print('OFFLINE 4s:')
print('  times=%.3f, ws=%.3f, T=%d, data=%d pts' % (total_times, fbcca_off.ws, fbcca_off.T, data_off.shape[1]))
print('  pred=%s, scores=' % COMMANDS_EN[pred_off],
      ', '.join('%s=%.3f' % (COMMANDS_EN[i], s) for i, s in enumerate(scores_off)))

# ===== ONLINE: 3.5s cycle, 3 x 2.5s sliding windows =====
data_on = data_full[:, TRIM:TRIM + 875]  # first 3.5s after trim
fbcca_on = FBCCA(3, 2.5, list(freqs), sample_rate=SAMPLE_RATE)
print()
print('ONLINE sliding (3.5s cycle, 2.5s windows):')
print('  times=2.5, ws=%.3f, T=%d, cycle_data=%d pts' % (fbcca_on.ws, fbcca_on.T, data_on.shape[1]))

results = []
for wi, offset in enumerate([0, 125, 250]):
    w = data_on[:, offset:offset + 625]
    arr = np.asarray(w, dtype=float)
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    arr = arr / (np.std(arr, axis=1, keepdims=True) + 1e-6)
    arr = np.clip(arr, -6.0, 6.0)
    try:
        pred, scores, _ = fbcca_on.classify_with_scores(arr)
        results.append(int(pred))
        print('  W%d [%d:%d]: pred=%s' % (wi, offset, offset+625, COMMANDS_EN[pred]),
              ', '.join('%s=%.3f' % (COMMANDS_EN[i], s) for i, s in enumerate(scores)))
    except Exception as e:
        print('  W%d: FAILED - %s' % (wi, e))
        traceback.print_exc()
        results.append(-1)

if len(results) >= 3:
    consensus = results[0] == results[1] == results[2]
    print('  Consensus:', consensus, ' results=', [COMMANDS_EN[r] if r >= 0 else 'ERR' for r in results])

# ===== ONLINE with score normalization =====
print()
print('ONLINE with EMA normalization:')
ema = np.ones(len(freqs), dtype=float)
alpha = 0.08
results_norm = []
for wi, offset in enumerate([0, 125, 250]):
    w = data_on[:, offset:offset + 625]
    arr = np.asarray(w, dtype=float)
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    arr = arr / (np.std(arr, axis=1, keepdims=True) + 1e-6)
    arr = np.clip(arr, -6.0, 6.0)
    try:
        _, raw_scores, _ = fbcca_on.classify_with_scores(arr)
        # Simulate _normalize_online_scores
        raw = np.maximum(np.asarray(raw_scores, dtype=float), 1e-8)
        norm = raw / (ema + 1e-8)
        ema = (1.0 - alpha) * ema + alpha * raw
        pred = int(np.argmax(norm))
        results_norm.append(pred)
        print('  W%d: pred=%s norm_scores=' % (wi, COMMANDS_EN[pred]),
              ', '.join('%s=%.3f' % (COMMANDS_EN[i], s) for i, s in enumerate(norm)))
    except Exception as e:
        print('  W%d: FAILED - %s' % (wi, e))
        results_norm.append(-1)

if len(results_norm) >= 3:
    consensus_norm = results_norm[0] == results_norm[1] == results_norm[2]
    print('  Consensus (norm):', consensus_norm)

# ===== SYNTHETIC TEST =====
print()
print('SYNTHETIC pure tone test:')
for tgt_idx, tgt_freq in enumerate(freqs):
    t = np.arange(625) / 250.0
    synth = np.zeros((8, 625))
    for ch in range(8):
        synth[ch] = (np.sin(2 * np.pi * tgt_freq * t) +
                     0.3 * np.sin(4 * np.pi * tgt_freq * t) +
                     0.05 * np.random.randn(625))
    arr = np.asarray(synth, dtype=float)
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    arr = arr / (np.std(arr, axis=1, keepdims=True) + 1e-6)
    arr = np.clip(arr, -6.0, 6.0)
    pred, scores, _ = fbcca_on.classify_with_scores(arr)
    ok = 'OK' if pred == tgt_idx else 'WRONG->' + COMMANDS_EN[pred]
    top_score = max(scores)
    print('  %s (%.2fHz): %s (top_score=%.3f)' % (COMMANDS_EN[tgt_idx], tgt_freq, ok, top_score))
