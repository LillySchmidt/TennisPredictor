"""Run 5-fold CV for classic ML models on tennis matches."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

# Support both package (`python -m ML.eval_cv`) and script (`python eval_cv.py`) execution
if __package__ is None or __package__ == "":
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import data_prep as dp  # type: ignore
    import models as mdl  # type: ignore
else:
    from . import data_prep as dp  # type: ignore
    from . import models as mdl  # type: ignore

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "all_matches.csv"
REPORT_PATH = ROOT / "reports/precision_cv.json"


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, float]:
    """Run cross-validate and return aggregated metrics."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    metrics = {k: (np.mean(v), np.std(v)) for k, v in cv_results.items() if k.startswith("test_")}
    return {f"{k}_mean": v[0] for k, v in metrics.items()} | {f"{k}_std": v[1] for k, v in metrics.items()} | {
        "folds": cv.get_n_splits()
    }


def run_cv(data_path: Path) -> Dict[str, Dict[str, float]]:
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    # Clean raw matches first; use cleaned file for modeling
    cleaned_path = dp.clean_raw_matches(data_path, dp.CLEANED_FILE)
    X, y = dp.load_features_and_target(cleaned_path, clean=False)

    models = mdl.make_models(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    summaries: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        print(f"Running CV for {name}...")
        summaries[name] = evaluate_model(name, model, X, y, cv)
        print(
            f"  precision={summaries[name]['test_precision_mean']:.4f} "
            f"+/- {summaries[name]['test_precision_std']:.4f}"
        )
    return summaries


def save_report(report: Dict[str, Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main(data_path: Path = DEFAULT_RAW, report_path: Path = REPORT_PATH) -> None:
    report = run_cv(data_path)
    save_report(report, report_path)
    print(f"\nSaved CV report to {report_path}")


if __name__ == "__main__":
    main()
