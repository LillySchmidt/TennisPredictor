#!/usr/bin/env python3
"""Compare DNN classification runs.

Reads JSON reports produced by `dnn/train_classification.py` (stored in
`dnn/reports/`), builds a leaderboard CSV, and generates a small set of
aggregate plots.

Outputs:
  - comparison_dnn/leaderboard.csv
  - comparison_dnn/images/*.png
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DNN_REPORTS = BASE_DIR / "dnn" / "reports"
OUT_DIR = BASE_DIR / "comparison_dnn"
IMG_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_reports() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not DNN_REPORTS.exists():
        return runs
    for p in sorted(DNN_REPORTS.glob("classification_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_report_path"] = str(p)
            runs.append(data)
        except Exception:
            continue
    return runs


def write_leaderboard(runs: List[Dict[str, Any]]) -> Path:
    rows = []
    for r in runs:
        tm = r.get("test_metrics", {})
        rows.append(
            {
                "run_name": r.get("run_name", ""),
                "preset": r.get("preset", ""),
                "arch": r.get("arch", ""),
                "hidden_dim": r.get("hidden_dim", ""),
                "num_layers": r.get("num_layers", ""),
                "dropout": r.get("dropout", ""),
                "l2": r.get("l2", ""),
                "norm": r.get("norm", ""),
                "activation": r.get("activation", ""),
                "amp": r.get("amp", ""),
                "grad_accum": r.get("grad_accum", ""),
                "batch_size": r.get("batch_size", ""),
                "accuracy": tm.get("accuracy", np.nan),
                "roc_auc": tm.get("roc_auc", np.nan),
                "pr_auc": tm.get("pr_auc", np.nan),
                "f1_score": tm.get("f1_score", np.nan),
                "precision": tm.get("precision", np.nan),
                "recall": tm.get("recall", np.nan),
                "training_time_seconds": r.get("training_time_seconds", np.nan),
                "peak_vram_allocated_mb": r.get("peak_vram_allocated_mb", np.nan),
                "params": r.get("params", np.nan),
                "report_path": r.get("_report_path", ""),
            }
        )

    def sort_key(x):
        v = x.get("roc_auc", np.nan)
        return -(v if v == v else -1)  # NaN-safe

    rows.sort(key=sort_key)
    out_csv = OUT_DIR / "leaderboard.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")
    return out_csv


def plot_top15(runs: List[Dict[str, Any]]) -> None:
    if not runs:
        return

    def roc(r):
        return float(r.get("test_metrics", {}).get("roc_auc", float("nan")))

    runs_sorted = sorted(runs, key=roc, reverse=True)
    top = runs_sorted[: min(15, len(runs_sorted))]

    names = [r.get("run_name", "") for r in top]
    roc_aucs = [roc(r) for r in top]
    f1s = [float(r.get("test_metrics", {}).get("f1_score", float("nan"))) for r in top]
    accs = [float(r.get("test_metrics", {}).get("accuracy", float("nan"))) for r in top]

    # ROC-AUC
    plt.figure(figsize=(12, 7), dpi=140)
    plt.barh(range(len(names)), roc_aucs)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("ROC-AUC")
    plt.title("Top DNN runs by ROC-AUC")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "leaderboard_roc_auc.png")
    plt.close()

    # Accuracy
    plt.figure(figsize=(12, 7), dpi=140)
    plt.barh(range(len(names)), accs)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Accuracy")
    plt.title("Top DNN runs by ROC-AUC — Accuracy")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "leaderboard_accuracy.png")
    plt.close()

    # F1
    plt.figure(figsize=(12, 7), dpi=140)
    plt.barh(range(len(names)), f1s)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("F1")
    plt.title("Top DNN runs by ROC-AUC — F1")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "leaderboard_f1.png")
    plt.close()


def main() -> None:
    runs = load_reports()
    if not runs:
        print("No DNN classification reports found in dnn/reports/")
        return
    csv_path = write_leaderboard(runs)
    plot_top15(runs)
    print(f"Wrote leaderboard: {csv_path}")
    print(f"Wrote images: {IMG_DIR}")


if __name__ == "__main__":
    main()
