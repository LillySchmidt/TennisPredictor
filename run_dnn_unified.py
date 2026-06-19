#!/usr/bin/env python3
"""
Train representative DNN configurations on the unified 70/15/15 split.
Uses the project/dnn model architecture and training loop, but with a
larger validation set (15% of total) and early stopping on validation loss.
Saves predictions and reports with bootstrap confidence intervals.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
UNIFIED_DIR = ROOT / "unified_data"
OUT_DIR = ROOT / "results" / "dnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import project/dnn modules
sys.path.insert(0, str(REPO / "project"))
from dnn.model import build_classifier, count_parameters

# Load ML preprocessor via importlib (avoid package-import shadowing issue)
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


def load_data():
    train = pd.read_csv(UNIFIED_DIR / "ml_train.csv")
    val = pd.read_csv(UNIFIED_DIR / "ml_val.csv")
    test = pd.read_csv(UNIFIED_DIR / "ml_test.csv")

    feature_cols = [c for c in train.columns if c != "label"]
    preprocessor, _, _ = build_preprocessor(train[feature_cols])

    X_train = preprocessor.fit_transform(train[feature_cols]).astype(np.float32)
    y_train = train["label"].values.astype(np.float32)
    X_val = preprocessor.transform(val[feature_cols]).astype(np.float32)
    y_val = val["label"].values.astype(np.float32)
    X_test = preprocessor.transform(test[feature_cols]).astype(np.float32)
    y_test = test["label"].values.astype(np.float32)

    print(f"Preprocessed feature dimension: {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.cpu().numpy())
    all_probs = np.concatenate(all_probs).squeeze()
    all_labels = np.concatenate(all_labels).squeeze()
    loss = total_loss / len(all_labels)
    return {"loss": loss, "probs": all_probs, "labels": all_labels}


def train_model(config, X_train, y_train, X_val, y_val, X_test, y_test, device):
    name = config["name"]
    batch_size = config.get("batch_size", 512)
    lr = config.get("lr", 1e-3)
    l2 = config.get("l2", 1e-4)
    epochs = config.get("epochs", 100)
    patience = config.get("patience", 10)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    model = build_classifier(
        input_dim=X_train.shape[1],
        preset=config.get("preset"),
        arch=config.get("arch", "plain"),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 6),
        dropout=config.get("dropout", 0.15),
        norm=config.get("norm", "batchnorm"),
        activation=config.get("activation", "relu"),
    ).to(device)

    params = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / len(X_train)

        val_result = evaluate(model, val_loader, device, criterion)
        val_loss = val_result["loss"]
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}")
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.perf_counter() - t0

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Test predictions
    test_result = evaluate(model, test_loader, device, criterion)
    test_prob = test_result["probs"]
    test_metrics = classification_metrics(y_test, test_prob)
    ci = bootstrap_ci(y_test, test_prob, n_bootstrap=1000)
    ece = compute_ece(y_test, test_prob)

    # Save predictions
    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test, "y_prob": test_prob}).to_csv(pred_dir / f"{name}_test.csv", index=False)

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": X_train.shape[1],
        "config": config,
    }, OUT_DIR / f"{name}.pt")

    report = {
        "model": name,
        "config": config,
        "params": params,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_time_seconds": train_time,
        "best_val_loss": best_val_loss,
        "epochs_ran": epoch,
        "metrics": test_metrics,
        "bootstrap_ci": ci,
        "ece": ece,
        "history": history,
    }
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  {name}: acc={test_metrics['accuracy']:.4f} auc={test_metrics['roc_auc']:.4f} f1={test_metrics['f1']:.4f} params={params} time={train_time:.1f}s")
    return report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    print(f"Data loaded: train={len(X_train)} val={len(X_val)} test={len(X_test)} features={len(feature_cols)}")

    configs = [
        {"name": "dnn_tiny_plain", "preset": "tiny", "batch_size": 512, "lr": 1e-3, "l2": 1e-4, "epochs": 100, "patience": 10},
        {"name": "dnn_small_plain", "preset": "small", "batch_size": 512, "lr": 1e-3, "l2": 1e-4, "epochs": 100, "patience": 10},
        {"name": "dnn_medium_residual", "preset": "medium", "batch_size": 512, "lr": 1e-3, "l2": 1e-4, "epochs": 100, "patience": 10},
        {"name": "dnn_large_residual", "preset": "large", "batch_size": 256, "lr": 1e-3, "l2": 1e-4, "epochs": 100, "patience": 10},
    ]

    results = []
    for cfg in configs:
        print(f"\nTraining {cfg['name']} ...")
        report = train_model(cfg, X_train, y_train, X_val, y_val, X_test, y_test, device)
        results.append(report)

    summary = pd.DataFrame([{
        "model": r["model"],
        "params": r["params"],
        "accuracy": r["metrics"]["accuracy"],
        "roc_auc": r["metrics"]["roc_auc"],
        "f1": r["metrics"]["f1"],
        "brier": r["metrics"]["brier"],
        "ece": r["ece"],
        "train_time": r["train_time_seconds"],
        "epochs": r["epochs_ran"],
    } for r in results])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDNN results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
