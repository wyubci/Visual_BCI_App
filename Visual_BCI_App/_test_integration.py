"""Quick integration test for new SSVEP modules."""
import sys
import os

PROJECT = r"E:\世界机器人大赛\projects\visual_bci_car\Visual_BCI_App"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

def test_spatial_decoder():
    import numpy as np
    from models.SpatialDecoder import SpatialDecoder
    np.random.seed(1)
    X = np.random.randn(30, 8, 250)
    yb = np.repeat([0, 3], 15)
    yf = np.tile([0, 1, 2], 10)[:30]
    d = SpatialDecoder(250, 5)
    d.fit(X, yb, yf)
    print(f"  SpatialDecoder: fitted={d.is_fitted}, models={list(d._models.keys())}")
    return True

def test_eeg_buffer():
    import numpy as np
    from interface.car_interface.acquisition import EegSampleBuffer
    buf = EegSampleBuffer()
    buf.set_trial_meta(block_idx=0, fp_idx=1, freq_hz=8.0, phase_rad=0.5)
    buf.append(np.random.randn(8, 100))
    mat = buf.materialize()
    print(f"  EegSampleBuffer: target_id={buf.target_id}, freq={buf.freq_hz}, shape={mat.shape}")
    return True

def test_extract_target_id():
    import numpy as np
    import tempfile, os
    from scipy.io import savemat, loadmat
    from interface.car_interface.training_framework import extract_target_id
    tmp = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
    savemat(tmp.name, {'block_idx': 5, 'fp_idx': 3, 'stim_freqs_hz': np.arange(40)})
    tmp.close()
    mat = loadmat(tmp.name)
    tid = extract_target_id(mat)
    print(f"  extract_target_id: block=5, fp=3 -> target_id={tid}")
    os.unlink(tmp.name)
    return tid == 28  # 5*5+3

def test_grid_config():
    from interface.car_interface.ssvep_grid_canvas import GridConfig
    cfg = GridConfig
    print(f"  GridConfig: {cfg.N_ROWS}x{cfg.N_COLS}={cfg.N_BLOCKS} blocks, "
          f"{cfg.N_TARGETS} targets")
    print(f"  Freqs: {cfg.FREQ_START}~{cfg.FREQ_END} Hz, step={cfg.FREQ_STEP}")
    print(f"  Phases: {[f'{p/math.pi:.2f}pi' for p in cfg.PHASE_VALUES]}")
    return cfg.N_BLOCKS == 40 and cfg.N_TARGETS == 200

def test_electrode_sim():
    from evaluation.electrode_sim import ElectrodeSimulator
    sim = ElectrodeSimulator(n_total_channels=8)
    print(f"  ElectrodeSimulator: montages={sim.available_montages()}")
    return True

def test_offline_eval():
    from evaluation.offline_eval_200 import OfflineEvaluator, _compute_itr
    ev = OfflineEvaluator(subject='test', n_classes=40)
    assert ev.Nf == 40
    # Test ITR computation
    itr = _compute_itr(0.9, 40, 1.0, 0.5)
    print(f"  OfflineEvaluator: ITR(acc=0.9, N=40, T=1.5s) = {itr:.1f} bpm")
    return itr > 0

def test_online_eval():
    import numpy as np
    from evaluation.online_eval_200 import OnlineEvaluator
    ev = OnlineEvaluator(n_classes=40, model_name='FBCCA')
    ev.generate_test_sequence(n_per_class=2)
    print(f"  OnlineEvaluator: {ev.total_trials} trials generated")
    return ev.total_trials == 80

if __name__ == "__main__":
    import math
    print("=" * 50)
    print("SSVEP 200-Target System Integration Test")
    print("=" * 50)
    tests = [
        ("SpatialDecoder", test_spatial_decoder),
        ("EegSampleBuffer", test_eeg_buffer),
        ("extract_target_id", test_extract_target_id),
        ("GridConfig", test_grid_config),
        ("ElectrodeSimulator", test_electrode_sim),
        ("OfflineEvaluator", test_offline_eval),
        ("OnlineEvaluator", test_online_eval),
    ]
    passed = 0
    for name, func in tests:
        try:
            ok = func()
            if ok:
                passed += 1
                print(f"  [PASS] {name}")
            else:
                print(f"  [FAIL] {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
