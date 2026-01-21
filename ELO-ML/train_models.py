import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ML_DIR = REPO_ROOT / "ML"

for path in (REPO_ROOT, ML_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ML import models_enhanced as ml_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train best ML models with Elo features.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "elo_ml_dataset.csv",
        help="Path to ELO-ML dataset CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "ML" / "outputs" / "reports" / "final_evaluation_report.json",
        help="ML report JSON used to select the best models.",
    )
    parser.add_argument("--top-n", type=int, default=3, help="Number of best models to train.")
    parser.add_argument("--metric", type=str, default="accuracy", help="Metric to rank models by.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split fraction of train.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for supported models (default: -1).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-models", action="store_true", help="Persist trained models with joblib.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Base output directory.")
    return parser.parse_args()


def _select_best_models(report_path: Path, metric: str, top_n: int) -> List[str]:
    fallback = ["logistic_regression", "gradient_boosting", "hist_gradient_boosting"]
    if not report_path.exists():
        return fallback[:top_n]

    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("all_model_results", {})
    scored: List[Tuple[str, float]] = []
    for name, payload in results.items():
        metrics = payload.get("metrics", {})
        metric_block = metrics.get(metric, {})
        score = metric_block.get("test_mean")
        if score is None:
            continue
        scored.append((name, float(score)))

    if not scored:
        return fallback[:top_n]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_n]]


def _get_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(X)
    return preds.astype(float)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output_dir / "reports"
    models_dir = args.output_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    y = df["label"].astype(int)
    X = df.drop(columns=["label", "match_id"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    val_size_adj = args.val_size / (1 - args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_size_adj,
        random_state=args.random_state,
        stratify=y_temp,
    )

    model_names = _select_best_models(args.report, args.metric, args.top_n)
    all_models = ml_models.make_models(X_train)
    models = {name: all_models[name] for name in model_names if name in all_models}

    summary = []
    for name, model in models.items():
        # Apply n_jobs where supported by the estimator.
        try:
            model.set_params(clf__n_jobs=args.n_jobs)
        except Exception:
            pass

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        val_prob = _get_probabilities(model, X_val)
        test_prob = _get_probabilities(model, X_test)

        val_metrics = _compute_metrics(y_val.to_numpy(), val_prob)
        test_metrics = _compute_metrics(y_test.to_numpy(), test_prob)

        report = {
            "model_name": name,
            "metric_rank": args.metric,
            "train_time_seconds": float(train_time),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "features": list(X.columns),
        }
        report_path = reports_dir / f"{name}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if args.save_models:
            joblib.dump(model, models_dir / f"{name}.joblib")

        summary.append(
            {
                "model": name,
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "train_time_seconds": float(train_time),
            }
        )

    summary_df = pd.DataFrame(summary).sort_values("test_accuracy", ascending=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"Wrote reports to {reports_dir}")
    print(f"Wrote summary to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
