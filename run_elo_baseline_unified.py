#!/usr/bin/env python3
"""
Evaluate the Elo baseline on the unified test split using the pre-match
Elo ratings stored in the ELO-ML dataset. Computes metrics and bootstrap CIs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
)

ROOT = Path(__file__).resolve().parent
UNIFIED_DIR = ROOT / "unified_data"
OUT_DIR = ROOT / "results" / "elo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y_true, y_prob),
    }


def bootstrap_ci(y_true, y_prob, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "log_loss": [], "brier": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        ypred = (yp >= 0.5).astype(int)
        metrics["accuracy"].append(accuracy_score(yt, ypred))
        metrics["precision"].append(precision_score(yt, ypred, zero_division=0))
        metrics["recall"].append(recall_score(yt, ypred, zero_division=0))
        metrics["f1"].append(f1_score(yt, ypred, zero_division=0))
        if len(np.unique(yt)) > 1:
            metrics["roc_auc"].append(roc_auc_score(yt, yp))
        metrics["log_loss"].append(log_loss(yt, np.clip(yp, 1e-6, 1 - 1e-6)))
        metrics["brier"].append(brier_score_loss(yt, yp))
    ci = {}
    for k, vals in metrics.items():
        if not vals:
            continue
        a = np.array(vals)
        ci[k] = {"mean": float(a.mean()), "low": float(np.percentile(a, 2.5)), "high": float(np.percentile(a, 97.5))}
    return ci


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(y_true))


def main():
    train = pd.read_csv(UNIFIED_DIR / "eloml_train.csv")
    val = pd.read_csv(UNIFIED_DIR / "eloml_val.csv")
    test = pd.read_csv(UNIFIED_DIR / "eloml_test.csv")

    y_test = test["label"].values
    elo_a = test["elo_a"].values
    elo_b = test["elo_b"].values
    y_prob = expected_score(elo_a, elo_b)

    metrics = classification_metrics(y_test, y_prob)
    ci = bootstrap_ci(y_test, y_prob, n_bootstrap=500)
    ece = compute_ece(y_test, y_prob)

    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test, "y_prob": y_prob, "elo_a": elo_a, "elo_b": elo_b}).to_csv(
        pred_dir / "elo_test.csv", index=False
    )

    report = {
        "model": "elo_baseline",
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "metrics": metrics,
        "bootstrap_ci": ci,
        "ece": ece,
    }
    with open(OUT_DIR / "elo_baseline.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Elo baseline on unified test: n={len(y_test)}")
    print(f"  accuracy={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f} brier={metrics['brier']:.4f} ece={ece:.4f}")
    print(f"  accuracy 95% CI: [{ci['accuracy']['low']:.4f}, {ci['accuracy']['high']:.4f}]")


if __name__ == "__main__":
    main()
