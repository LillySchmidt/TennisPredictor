#!/usr/bin/env python3
"""Compute McNemar's test for key pairwise model comparisons."""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_preds(path: Path):
    df = pd.read_csv(path)
    return df["y_true"].values, (df["y_prob"].values >= 0.5).astype(int)


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n_a_only = int(((correct_a) & (~correct_b)).sum())
    n_b_only = int(((~correct_a) & (correct_b)).sum())
    if n_a_only + n_b_only == 0:
        return 1.0, n_a_only, n_b_only
    p = 2 * min(binomtest(n_a_only, n_a_only + n_b_only, p=0.5).pvalue,
                binomtest(n_b_only, n_a_only + n_b_only, p=0.5).pvalue)
    return min(p, 1.0), n_a_only, n_b_only


def best_by_accuracy(json_dir: Path, prefix: str = "", exclude_prefix: str = ""):
    """Return (name, json_path) of the model with highest accuracy."""
    best_name, best_acc, best_path = None, -1, None
    for p in json_dir.glob("*.json"):
        name = p.stem
        if prefix and not name.startswith(prefix):
            continue
        if exclude_prefix and name.startswith(exclude_prefix):
            continue
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "metrics" not in data:
            continue
        acc = data.get("metrics", {}).get("accuracy", -1)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_path = p
    return best_name, best_path


def main():
    # Best classical ML
    ml_name, ml_json = best_by_accuracy(RESULTS / "ml_eloml", exclude_prefix="eloml_")
    # Best ELO-ML
    eloml_name, eloml_json = best_by_accuracy(RESULTS / "ml_eloml", prefix="eloml_")
    # Best DNN
    dnn_name, dnn_json = best_by_accuracy(RESULTS / "dnn")

    paths = {
        "elo": RESULTS / "elo" / "predictions" / "elo_test.csv",
        "best_ml": RESULTS / "ml_eloml" / "predictions" / "ml" / f"{ml_name}_test.csv",
        "best_eloml": RESULTS / "ml_eloml" / "predictions" / "eloml" / f"{eloml_name}_test.csv",
        "best_dnn": RESULTS / "dnn" / "predictions" / f"{dnn_name}_test.csv",
    }

    print(f"Selected best ML: {ml_name}")
    print(f"Selected best ELO-ML: {eloml_name}")
    print(f"Selected best DNN: {dnn_name}")

    preds = {k: load_preds(p) for k, p in paths.items()}

    comparisons = [
        ("ELO-ML", "best_eloml", "Classical ML", "best_ml"),
        ("ELO-ML", "best_eloml", "DNN", "best_dnn"),
        ("ELO-ML", "best_eloml", "Elo", "elo"),
        ("Classical ML", "best_ml", "DNN", "best_dnn"),
        ("Classical ML", "best_ml", "Elo", "elo"),
        ("DNN", "best_dnn", "Elo", "elo"),
    ]

    results = []
    for name_a, key_a, name_b, key_b in comparisons:
        y_true_a, pred_a = preds[key_a]
        y_true_b, pred_b = preds[key_b]
        assert (y_true_a == y_true_b).all()
        p, n_a_only, n_b_only = mcnemar_test(y_true_a, pred_a, pred_b)
        acc_diff = (pred_a == y_true_a).mean() - (pred_b == y_true_b).mean()
        results.append({
            "comparison": f"{name_a} vs. {name_b}",
            "model_1": name_a,
            "model_2": name_b,
            "acc_diff_pp": acc_diff * 100,
            "p_value": p,
            "n_1_only": int(n_a_only),
            "n_2_only": int(n_b_only),
        })
        print(f"{name_a} vs {name_b}: acc_diff={acc_diff*100:+.2f}pp, p={p:.4f}")

    mcnemar_dir = RESULTS / "mcnemar"
    mcnemar_dir.mkdir(parents=True, exist_ok=True)
    with open(mcnemar_dir / "mcnemar_results.json", "w") as f:
        json.dump({"comparisons": results}, f, indent=2)


if __name__ == "__main__":
    main()
