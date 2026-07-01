from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ssvep_tdca_experiment as exp
from ssvep_tdca_more_models import TriBranchTDCA, top_margin


DEFAULT_WINDOWS = (0.3, 0.5, 1.0, 2.0, 4.0)
DEFAULT_METHODS = ("CCA", "FBCCA", "TDCA", "ImprovedTDCA", "TriBranchTDCA")


def make_model(name: str, x_train: np.ndarray, y_train: np.ndarray, block_train: np.ndarray):
    if name == "TDCA":
        return exp.TDCA().fit(x_train, y_train)
    if name == "ImprovedTDCA":
        return exp.ImprovedTDCA().fit(x_train, y_train, block_train)
    if name == "TriBranchTDCA":
        return TriBranchTDCA().fit(x_train, y_train)
    raise KeyError(name)


def predict_method(name: str, x_train: np.ndarray, y_train: np.ndarray, block_train: np.ndarray, x_test: np.ndarray):
    if name == "CCA":
        return exp.predict_cca(x_test)
    if name == "FBCCA":
        return exp.predict_fbcca(x_test)
    model = make_model(name, x_train, y_train, block_train)
    return model.predict(x_test)


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    exists = path.exists()
    df.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8-sig")


def load_done_keys(path: Path) -> set[tuple[int, float, int, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["subject"].astype(int), df["window_sec"].astype(float), df["block"].astype(int), df["method"].astype(str)))


def evaluate_subject_window(
    sid: int,
    mat_path: Path,
    window_sec: float,
    methods: tuple[str, ...],
    done: set[tuple[int, float, int, str]],
) -> tuple[list[dict], list[dict]]:
    exp.configure_window(window_sec)
    data = exp.load_subject_data(mat_path)
    x, y, block_ids = exp.extract_trials(data, extra_points=exp.MAX_DELAY_POINTS)
    block_rows: list[dict] = []
    trial_rows: list[dict] = []

    for block in np.unique(block_ids):
        train = block_ids != block
        test = block_ids == block
        x_train, y_train, block_train = x[train], y[train], block_ids[train]
        x_test, y_test = x[test], y[test]
        for method in methods:
            key = (sid, float(window_sec), int(block), method)
            if key in done:
                continue
            pred, scores = predict_method(method, x_train, y_train, block_train, x_test)
            margins = top_margin(scores)
            acc = float(np.mean(pred == y_test))
            block_rows.append(
                {
                    "subject": sid,
                    "window_sec": float(window_sec),
                    "block": int(block),
                    "method": method,
                    "accuracy": acc,
                    "itr_bits_min": exp.itr_bits_per_min(acc),
                    "mean_margin": float(np.mean(margins)),
                    "median_margin": float(np.median(margins)),
                    "min_margin": float(np.min(margins)),
                }
            )
            for yt, yp, margin in zip(y_test, pred, margins):
                trial_rows.append(
                    {
                        "subject": sid,
                        "window_sec": float(window_sec),
                        "block": int(block),
                        "method": method,
                        "true_label": int(yt),
                        "pred_label": int(yp),
                        "true_freq": float(exp.TARGET_FREQS[int(yt)]),
                        "pred_freq": float(exp.TARGET_FREQS[int(yp)]),
                        "correct": int(yt == yp),
                        "margin": float(margin),
                    }
                )
            print(
                f"S{sid:02d} window={window_sec:g}s block={int(block)+1}/6 {method:14s} "
                f"acc={acc:.3f} itr={exp.itr_bits_per_min(acc):.1f}",
                flush=True,
            )
    return block_rows, trial_rows


def summarize(out_dir: Path) -> None:
    block_df = pd.read_csv(out_dir / "block_metrics.csv")
    trial_df = pd.read_csv(out_dir / "trial_predictions.csv")

    subject_df = (
        block_df.groupby(["subject", "window_sec", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            itr_bits_min=("itr_bits_min", "mean"),
            mean_margin=("mean_margin", "mean"),
            accuracy_std_across_blocks=("accuracy", "std"),
        )
        .sort_values(["window_sec", "subject", "method"])
    )
    method_df = (
        subject_df.groupby(["window_sec", "method"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_itr_bits_min=("itr_bits_min", "mean"),
            std_itr_bits_min=("itr_bits_min", "std"),
            mean_margin=("mean_margin", "mean"),
            std_margin=("mean_margin", "std"),
            subjects=("subject", "count"),
        )
        .sort_values(["window_sec", "mean_accuracy"], ascending=[True, False])
    )
    subject_df.to_csv(out_dir / "subject_summary.csv", index=False, encoding="utf-8-sig")
    method_df.to_csv(out_dir / "method_summary.csv", index=False, encoding="utf-8-sig")

    methods = list(dict.fromkeys(method_df["method"].tolist()))
    colors = {
        "CCA": "#4C78A8",
        "FBCCA": "#F58518",
        "TDCA": "#54A24B",
        "ImprovedTDCA": "#B279A2",
        "TriBranchTDCA": "#E45756",
    }
    for metric, ylabel, out_name in [
        ("mean_accuracy", "Mean accuracy", "full_accuracy.png"),
        ("mean_itr_bits_min", "Mean ITR (bits/min)", "full_itr.png"),
        ("mean_margin", "Mean top-2 margin", "full_margin.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
        for method in methods:
            sub = method_df[method_df["method"] == method].sort_values("window_sec")
            ax.plot(sub["window_sec"], sub[metric], marker="o", label=method, color=colors.get(method))
        ax.set_xlabel("Analysis window (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Full Benchmark SSVEP - {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / out_name)

    # Per-subject best-model comparison for the final candidate.
    pivot = subject_df.pivot_table(index=["subject", "window_sec"], columns="method", values="accuracy").reset_index()
    pivot.to_csv(out_dir / "subject_accuracy_pivot.csv", index=False, encoding="utf-8-sig")


def parse_list(value: str, cast):
    return tuple(cast(x.strip()) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path, default=Path("ssvep_results") / "full_benchmark")
    parser.add_argument("--windows", default="0.3,0.5,1,2,4")
    parser.add_argument("--methods", default="CCA,FBCCA,TDCA,ImprovedTDCA,TriBranchTDCA")
    parser.add_argument("--limit", type=int, default=0, help="0 means all subjects")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    windows = parse_list(args.windows, float)
    methods = parse_list(args.methods, str)
    subjects = exp.discover_subject_files(args.benchmark, None if args.limit == 0 else args.limit)
    block_path = args.out / "block_metrics.csv"
    trial_path = args.out / "trial_predictions.csv"
    done = load_done_keys(block_path)

    print("Full Benchmark run")
    print("Subjects:", len(subjects), [sid for sid, _ in subjects])
    print("Windows:", windows)
    print("Methods:", methods)
    print("Output:", args.out.resolve())
    start = time.time()

    for sid, mat_path in subjects:
        for window_sec in windows:
            block_rows, trial_rows = evaluate_subject_window(sid, mat_path, window_sec, methods, done)
            append_rows(block_path, block_rows)
            append_rows(trial_path, trial_rows)
            for row in block_rows:
                done.add((int(row["subject"]), float(row["window_sec"]), int(row["block"]), str(row["method"])))
            summarize(args.out)
            print(f"Saved progress after S{sid:02d}, window={window_sec:g}s", flush=True)

    summarize(args.out)
    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
