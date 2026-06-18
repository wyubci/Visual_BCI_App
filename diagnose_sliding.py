"""Diagnose sliding window classification — show scores, bias, algorithm details."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, os
from scipy.io import loadmat
from models.FBCCA import FBCCA

SAMPLE_RATE = 250
TRIM_PTS = 160

CYCLE_SEC = 3.5
WINDOW_SEC = 2.5
STEP_SEC = 0.5
cycle_samples = int(CYCLE_SEC * SAMPLE_RATE)
window_samples = int(WINDOW_SEC * SAMPLE_RATE)
step_samples = int(STEP_SEC * SAMPLE_RATE)

COMMANDS = ['forward', 'backward', 'left', 'stop', 'right']

wf = r'saveCarData\hc33\weights\car_tdca_250hz_weights_20260602_211526.json'
with open(wf, encoding='utf-8') as f:
    cfg = json.load(f)
freqs = np.array(cfg['stim_freqs_hz'], dtype=float)

print('=== SLIDING WINDOW ALGORITHM DIAGNOSTIC ===')
print()
print('Parameters:')
print(f'  Cycle: {CYCLE_SEC}s = {cycle_samples} samples')
print(f'  Window: {WINDOW_SEC}s = {window_samples} samples')
print(f'  Step: {STEP_SEC}s = {step_samples} samples')
print(f'  Trim (pre_stim+delay): {TRIM_PTS} samples')
print(f'  Stimulus frequencies:')
for i, (cmd, f) in enumerate(zip(COMMANDS, freqs)):
    print(f'    [{i}] {cmd} = {f:.3f} Hz')
print()

# FBCCA sliding classifier
fbcca = FBCCA(3, WINDOW_SEC, list(freqs), sample_rate=SAMPLE_RATE)
print(f'FBCCA sliding classifier:')
print(f'  ws = {fbcca.ws:.3f}s (window - delay)')
print(f'  T  = {fbcca.T} samples (effective)')
print(f'  Nm = {fbcca.Nm} filter banks')
print(f'  delay_sec = {fbcca.delay_sec}s')
print(f'  reference_signals shape = {fbcca.reference_signals.shape}')
print(f'  frequency_weights = {fbcca.frequency_weights}')
print()

# Test ALL samples
print('=' * 130)
print(f'{"#":<3} {"file":<30} {"true":<10} {"W0":<10} {"W1":<10} {"W2":<10} {"match":<8} {"scores (avg)"}')
print('-' * 130)

total = 0
correct = 0
pred_counts = {i: 0 for i in range(5)}

for idx, fp in enumerate(cfg['train_files']):
    m = loadmat(fp)
    true_l = int(np.array(m['label_idx']).reshape(-1)[0])
    data = np.array(m['data'], dtype=float)
    data_trimmed = data[:, TRIM_PTS:TRIM_PTS + cycle_samples]

    if data_trimmed.shape[-1] < cycle_samples:
        continue

    results = []
    all_scores = []
    for offset in [0, step_samples, 2 * step_samples]:
        window = data_trimmed[:, offset:offset + window_samples]
        arr = np.asarray(window, dtype=float)
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        ch_std = np.std(arr, axis=1, keepdims=True)
        arr = arr / (ch_std + 1e-6)
        arr = np.clip(arr, -6.0, 6.0)

        pred, scores, conf = fbcca.classify_with_scores(arr)
        results.append(int(pred))
        all_scores.append(scores)

    total += 1
    consensus = results[0] == results[1] == results[2]
    if consensus:
        pred_counts[results[0]] += 1
        if results[0] == true_l:
            correct += 1

    fname = os.path.basename(fp)
    avg_scores = np.mean(all_scores, axis=0)
    score_str = ', '.join(f'{COMMANDS[i]}={s:.2f}' for i, s in enumerate(avg_scores))
    match = 'HIT' if (consensus and results[0] == true_l) else ('MISS' if consensus else 'NOCON')

    print(f'{idx+1:<3} {fname:<30} {COMMANDS[true_l]:<10} {COMMANDS[results[0]]:<10} {COMMANDS[results[1]]:<10} {COMMANDS[results[2]]:<10} {match:<8} {score_str}')

print('-' * 130)
print()
print(f'Accuracy (consensus only): {correct}/{total} = {correct/total*100:.1f}%')
print(f'Prediction distribution (when consensus):')
for i in range(5):
    print(f'  [{i}] {COMMANDS[i]}: {pred_counts[i]} times')
print()

# Check if forward is systematically biased
print('=== Forward bias analysis ===')
forward_wins = 0
forward_scores_all = []
for idx, fp in enumerate(cfg['train_files']):
    m = loadmat(fp)
    true_l = int(np.array(m['label_idx']).reshape(-1)[0])
    data = np.array(m['data'], dtype=float)
    data_trimmed = data[:, TRIM_PTS:TRIM_PTS + cycle_samples]

    for offset in [0, step_samples, 2 * step_samples]:
        window = data_trimmed[:, offset:offset + window_samples]
        arr = np.asarray(window, dtype=float)
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        ch_std = np.std(arr, axis=1, keepdims=True)
        arr = arr / (ch_std + 1e-6)
        arr = np.clip(arr, -6.0, 6.0)
        _, scores, _ = fbcca.classify_with_scores(arr)
        forward_scores_all.append(scores[0])
        if np.argmax(scores) == 0:
            forward_wins += 1

total_windows = len(forward_scores_all)
print(f'Total windows classified: {total_windows}')
print(f'Forward (index 0) won: {forward_wins}/{total_windows} = {forward_wins/total_windows*100:.1f}%')
print(f'Forward avg score: {np.mean(forward_scores_all):.4f}')

# Score distribution by frequency
print()
print('=== Per-frequency score distribution (all windows) ===')
all_scores_matrix = []
for idx, fp in enumerate(cfg['train_files']):
    m = loadmat(fp)
    data = np.array(m['data'], dtype=float)
    data_trimmed = data[:, TRIM_PTS:TRIM_PTS + cycle_samples]
    for offset in [0, step_samples, 2 * step_samples]:
        window = data_trimmed[:, offset:offset + window_samples]
        arr = np.asarray(window, dtype=float)
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        ch_std = np.std(arr, axis=1, keepdims=True)
        arr = arr / (ch_std + 1e-6)
        arr = np.clip(arr, -6.0, 6.0)
        _, scores, _ = fbcca.classify_with_scores(arr)
        all_scores_matrix.append(scores)

all_scores_matrix = np.array(all_scores_matrix)
for i in range(5):
    col = all_scores_matrix[:, i]
    print(f'  [{i}] {COMMANDS[i]:<10} ({freqs[i]:.2f}Hz): mean={col.mean():.4f}, std={col.std():.4f}, median={np.median(col):.4f}')

# Win rate per frequency
print()
print('=== Win rate per frequency (argmax across all windows) ===')
wins = np.argmax(all_scores_matrix, axis=1)
for i in range(5):
    count = np.sum(wins == i)
    print(f'  [{i}] {COMMANDS[i]:<10} ({freqs[i]:.2f}Hz): {count}/{len(wins)} = {count/len(wins)*100:.1f}%')
