#!/usr/bin/env python3
"""
Train classical ML and ELO-ML models on the unified 70/15/15 split,
evaluate on the held-out test set, save predictions, and compute bootstrap CIs.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT_DIR = ROOT / "results" / "ml_eloml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_DIR = ROOT / "unified_data"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ml_preprocess = load_module("ml_preprocess", REPO / "ML" / "preprocess.py")
build_preprocessor = ml_preprocess.build_preprocessor


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000, seed: int = 42):
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


def make_models(X, feature_set: str):
    """Return model pipelines. feature_set is 'ml' or 'eloml'."""
    preprocessor, _, _ = build_preprocessor(X)

    def _pipe(est):
        return Pipeline([("prep", preprocessor), ("clf", est)])

    models = {
        "logistic_regression": _pipe(LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)),
        "ridge_classifier": _pipe(RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42)),
        "naive_bayes": _pipe(GaussianNB()),
        "knn": _pipe(KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        "decision_tree": _pipe(DecisionTreeClassifier(max_depth=10, min_samples_split=20, class_weight="balanced", random_state=42)),
        "random_forest": _pipe(RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=4, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        "extra_trees": _pipe(ExtraTreesClassifier(n_estimators=200, max_depth=20, min_samples_split=4, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        "gradient_boosting": _pipe(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
        "hist_gradient_boosting": _pipe(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=7, random_state=42)),
        "adaboost": _pipe(AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=42), n_estimators=100, learning_rate=1.0, random_state=42)),
        "sgd_classifier": _pipe(SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42, max_iter=1000, tol=1e-3)),
    }

    if feature_set == "eloml":
        # Only the three algorithms reported in the thesis
        selected = ["logistic_regression", "ridge_classifier", "adaboost"]
        return {f"eloml_{k}": models[k] for k in selected}
    return models


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X)
        prob = 1 / (1 + np.exp(-d))
    else:
        prob = model.predict(X).astype(float)
    return prob


def train_and_evaluate(feature_set: str):
    print(f"\n=== {feature_set.upper()} models ===")
    train = pd.read_csv(UNIFIED_DIR / f"{feature_set}_train.csv")
    val = pd.read_csv(UNIFIED_DIR / f"{feature_set}_val.csv")
    test = pd.read_csv(UNIFIED_DIR / f"{feature_set}_test.csv")

    feature_cols = [c for c in train.columns if c != "label"]
    X_train, y_train = train[feature_cols], train["label"].values
    X_val, y_val = val[feature_cols], val["label"].values
    X_test, y_test = test[feature_cols], test["label"].values

    models = make_models(X_train, feature_set)
    results = []
    pred_dir = OUT_DIR / "predictions" / feature_set
    pred_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"Training {name} ...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_prob = get_probabilities(model, X_test)
        pred_time = time.perf_counter() - t0

        # Save model and predictions
        model_dir = OUT_DIR / "models" / feature_set
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"{name}.joblib")

        pd.DataFrame({"y_true": y_test, "y_prob": test_prob}).to_csv(pred_dir / f"{name}_test.csv", index=False)

        metrics = classification_metrics(y_test, test_prob)
        ci = bootstrap_ci(y_test, test_prob, n_bootstrap=500)

        # Calibrate ECE roughly
        ece = compute_ece(y_test, test_prob)

        result = {
            "model": name,
            "feature_set": feature_set,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "train_time_seconds": train_time,
            "pred_time_seconds": pred_time,
            "metrics": metrics,
            "bootstrap_ci": ci,
            "ece": ece,
        }
        results.append(result)
        with open(OUT_DIR / f"{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f} train={train_time:.1f}s pred={pred_time:.3f}s")

    summary = pd.DataFrame([{**r["metrics"], "model": r["model"], "train_time": r["train_time_seconds"], "pred_time": r["pred_time_seconds"]} for r in results])
    summary.to_csv(OUT_DIR / f"{feature_set}_summary.csv", index=False)
    return results


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(y_true))


def main():
    ml_results = train_and_evaluate("ml")
    eloml_results = train_and_evaluate("eloml")

    all_results = ml_results + eloml_results
    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
