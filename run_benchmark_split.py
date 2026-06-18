# -*- coding: utf-8 -*-
"""Split benchmark runner — each process handles a subset of subjects.
Usage: python run_benchmark_split.py <start_subj> <end_subj> <output_json>
Example: python run_benchmark_split.py 1 9 results_1_9.json
"""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
from benchmark_worker import evaluate_subject, OCCIPITAL_CHANNELS

DATA_LENGTHS = [0.3, 0.5, 0.7, 1.0]
MODEL_TYPES = ["TDCA", "FBCCA", "CCA"]

def main():
    start_subj = int(sys.argv[1])
    end_subj = int(sys.argv[2])
    output_path = sys.argv[3]

    subjects = list(range(start_subj, end_subj + 1))
    results = {}

    print(f"Processing S{start_subj:02d}-S{end_subj:02d} ({len(subjects)} subjects)...", flush=True)

    for idx, sid in enumerate(subjects):
        t0 = time.time()
        result = evaluate_subject((sid, DATA_LENGTHS, MODEL_TYPES, OCCIPITAL_CHANNELS))

        if result["error"]:
            print(f"  [{idx+1:2d}/{len(subjects)}] S{sid:02d}: {result['error']}", flush=True)
            results[f"S{sid:02d}"] = {"error": result["error"]}
        else:
            parts = []
            for (mt, dl), acc in sorted(result["results"].items()):
                results.setdefault(f"S{sid:02d}", {})[f"{mt}_{dl}s"] = acc
                parts.append(f"{mt}@{dl:.1f}s={acc:.2f}%")
            print(f"  [{idx+1:2d}/{len(subjects)}] S{sid:02d}: " + " | ".join(parts) +
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()
