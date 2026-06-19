#!/usr/bin/env python3
"""
Measure wall-clock inference time for each best model on the unified test set.
Loads saved models where available; for Elo baseline it uses the closed-form
expected_score computation.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
UNIFIED_DIR = ROOT / "unified_data"
OUT_DIR = ROOT / "results" / "timing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ml_preprocess = load_module("ml_preprocess", REPO / "ML" / "preprocess.py")
build_preprocessor = ml_preprocess.build_preprocessor


def _get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1 / (1 + np.exp(-d))
    else:
        return model.predict(X).astype(float)


def time_ml_model(model, X_test, n_runs=10, warmup=2):
    # Warm-up
    for _ in range(warmup):
        _ = _get_probabilities(model, X_test)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _get_probabilities(model, X_test)
        times.append(time.perf_counter() - t0)
    return {"mean": float(np.mean(times)), "std": float(np.std(times)), "min": float(np.min(times)), "max": float(np.max(times))}


def time_dnn_model(model_path, X_test, device, n_runs=10, warmup=2):
    sys.path.insert(0, str(REPO / "project"))
    from dnn.model import build_classifier

    ckpt = torch.load(model_path, map_location=device)
    cfg = {k: v for k, v in ckpt["config"].items() if k not in ("name", "batch_size", "lr", "l2", "epochs", "patience")}
    model = build_classifier(input_dim=ckpt["input_dim"], **cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_t = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = torch.sigmoid(model(X_t)).cpu().numpy()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = torch.sigmoid(model(X_t)).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return {"mean": float(np.mean(times)), "std": float(np.std(times)), "min": float(np.min(times)), "max": float(np.max(times))}


def time_elo_baseline(X_test_elo_a, X_test_elo_b, n_runs=10, warmup=2):
    for _ in range(warmup):
        _ = 1.0 / (1.0 + 10 ** ((X_test_elo_b - X_test_elo_a) / 400.0))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = 1.0 / (1.0 + 10 ** ((X_test_elo_b - X_test_elo_a) / 400.0))
        times.append(time.perf_counter() - t0)
    return {"mean": float(np.mean(times)), "std": float(np.std(times)), "min": float(np.min(times)), "max": float(np.max(times))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (raw features for pipeline models; preprocessed for DNN)
    ml_train = pd.read_csv(UNIFIED_DIR / "ml_train.csv")
    ml_test = pd.read_csv(UNIFIED_DIR / "ml_test.csv")
    ml_features = [c for c in ml_train.columns if c != "label"]
    X_ml_test_raw = ml_test[ml_features]
    y_test = ml_test["label"].values
    preprocessor, _, _ = build_preprocessor(ml_train[ml_features])
    preprocessor.fit(ml_train[ml_features])
    X_ml_test_tx = preprocessor.transform(ml_test[ml_features]).astype(np.float32)

    eloml_test = pd.read_csv(UNIFIED_DIR / "eloml_test.csv")

    results = []

    # Elo baseline
    print("Timing Elo baseline...")
    timing = time_elo_baseline(eloml_test["elo_a"].values, eloml_test["elo_b"].values)
    results.append({"model": "Elo Baseline", "approach": "Elo", "timing_seconds": timing, "test_samples": len(y_test)})

    # ML models
    ml_model_dir = ROOT / "results" / "ml_eloml" / "models" / "ml"
    if ml_model_dir.exists():
        for model_file in sorted(ml_model_dir.glob("*.joblib")):
            name = model_file.stem
            print(f"Timing ML model: {name} ...")
            model = joblib.load(model_file)
            timing = time_ml_model(model, X_ml_test_raw)
            results.append({"model": name, "approach": "ML", "timing_seconds": timing, "test_samples": len(y_test)})

    # ELO-ML models
    eloml_model_dir = ROOT / "results" / "ml_eloml" / "models" / "eloml"
    if eloml_model_dir.exists():
        eloml_train = pd.read_csv(UNIFIED_DIR / "eloml_train.csv")
        eloml_features = [c for c in eloml_train.columns if c != "label"]
        X_eloml_test = eloml_test[eloml_features]
        for model_file in sorted(eloml_model_dir.glob("*.joblib")):
            name = model_file.stem
            print(f"Timing ELO-ML model: {name} ...")
            model = joblib.load(model_file)
            timing = time_ml_model(model, X_eloml_test)
            results.append({"model": name, "approach": "ELO-ML", "timing_seconds": timing, "test_samples": len(y_test)})

    # DNN models
    dnn_model_dir = ROOT / "results" / "dnn"
    if dnn_model_dir.exists():
        for model_file in sorted(dnn_model_dir.glob("*.pt")):
            name = model_file.stem
            print(f"Timing DNN model: {name} ...")
            timing = time_dnn_model(model_file, X_ml_test_tx, device)
            results.append({"model": name, "approach": "DNN", "timing_seconds": timing, "test_samples": len(y_test)})

    with open(OUT_DIR / "inference_timing.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = pd.DataFrame([{
        "model": r["model"],
        "approach": r["approach"],
        "test_samples": r["test_samples"],
        "mean_time_s": r["timing_seconds"]["mean"],
        "std_time_s": r["timing_seconds"]["std"],
        "ms_per_1000": r["timing_seconds"]["mean"] / r["test_samples"] * 1000 * 1000,
    } for r in results])
    summary.to_csv(OUT_DIR / "inference_timing.csv", index=False)

    print(f"\nInference timing saved to {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
