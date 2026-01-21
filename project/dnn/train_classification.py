import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dnn.config import data_cfg, paths, train_cfg  # noqa: E402
from dnn.datasets import build_splits, make_loaders  # noqa: E402
from dnn.metrics import classification_metrics, compute_calibration_metrics, compute_optimal_thresholds  # noqa: E402
from dnn.model import build_classifier, count_parameters, estimate_model_memory_bytes, estimate_optimizer_memory_bytes  # noqa: E402
from dnn.utils import ensure_dir, get_device, save_json, seed_everything, timed  # noqa: E402
from dnn.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_probability_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classification model (winner prediction).")
    parser.add_argument("--data", type=Path, default=data_cfg.processed_npz, help="Path to processed dataset npz")
    parser.add_argument(
        "--preset",
        type=str,
        default=getattr(train_cfg, "preset", "medium"),
        choices=["tiny", "small", "medium", "large"],
        help="Model preset (recommended). Overrides arch/hidden_dim/num_layers/dropout/norm.",
    )
    parser.add_argument("--arch", type=str, default="residual", choices=["plain", "residual"], help="Architecture family")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden width (used when preset=None)")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of hidden layers or residual blocks")
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout probability")
    parser.add_argument("--norm", type=str, default="layernorm", choices=["batchnorm", "layernorm", "none"], help="Normalization")
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"], help="Activation")
    parser.add_argument("--epochs", type=int, default=train_cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=train_cfg.lr)
    parser.add_argument("--l2", type=float, default=train_cfg.weight_decay, help="L2 regularization (AdamW weight_decay)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (recommended for <=6GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--random_state", type=int, default=data_cfg.random_state)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)
        # Targets come in with an extra singleton dim; flatten to match logits.
        yb = yb.to(device).view(-1)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(xb).squeeze(1)
            raw_loss = criterion(logits, yb)
            loss = raw_loss / max(1, grad_accum)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step % max(1, grad_accum)) == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Track the true (non-normalized) loss for reporting.
        total_loss += float(raw_loss.detach().cpu().item()) * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    probs = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    metrics = classification_metrics(y_true, y_prob)
    metrics["loss"] = avg_loss
    return metrics


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for a loader."""
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            logits = model(xb).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.cpu().numpy())
    return np.concatenate(targets), np.concatenate(probs)


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)
    device = get_device()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    splits = build_splits(args.data, target_prefix="cls")
    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=args.batch_size)

    input_dim = splits.X_train.shape[1]
    preset = args.preset if args.preset else None
    model = build_classifier(
        input_dim=input_dim,
        preset=preset,
        arch=args.arch,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm=args.norm,
        activation=args.activation,
    ).to(device)
    params = count_parameters(model)

    # Memory heuristics (rough, but useful when targeting <= 6GB)
    dtype_bytes = 2 if args.amp else 4
    est_model = estimate_model_memory_bytes(params, dtype_bytes=dtype_bytes)
    est_opt = estimate_optimizer_memory_bytes(params, optimizer="adamw", dtype_bytes=4)  # states usually fp32
    est_total_mb = (est_model + est_opt) / (1024**2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    use_scaler = scaler if (args.amp and device.type == "cuda") else None

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_roc_auc": [],
    }

    with timed() as t_train:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, use_scaler, args.grad_accum)
            val_metrics = evaluate(model, val_loader, device, criterion)
            scheduler.step(val_metrics["loss"])

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_accuracy"].append(float(val_metrics.get("accuracy", 0.0)))
            history["val_roc_auc"].append(float(val_metrics.get("roc_auc", float("nan"))))

            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = val_metrics["loss"]
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)

    y_true_test, y_prob_test = predict_probs(model, test_loader, device)

    # Use default 0.5 threshold for summary metrics, but also log optimals
    test_metrics = classification_metrics(y_true_test, y_prob_test, threshold=0.5)
    calibration = compute_calibration_metrics(y_true_test, y_prob_test, n_bins=10)
    optimal_thresholds = compute_optimal_thresholds(y_true_test, y_prob_test)

    # Save model and report
    ensure_dir(paths.dnn_models)
    ensure_dir(paths.dnn_reports)
    ensure_dir(paths.dnn_images)

    run_name = f"{args.preset}_{args.arch}_hd{args.hidden_dim}_L{args.num_layers}_do{args.dropout}_l2{args.l2}_bs{args.batch_size}_amp{int(args.amp)}"
    model_path = paths.dnn_models / f"cls_{run_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "config": {
                "preset": args.preset,
                "arch": args.arch,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "norm": args.norm,
                "activation": args.activation,
            },
        },
        model_path,
    )

    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    report = {
        "run_name": run_name,
        "preset": args.preset,
        "arch": args.arch,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "l2": args.l2,
        "batch_size": args.batch_size,
        "norm": args.norm,
        "activation": args.activation,
        "params": params,
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
        "grad_accum": int(args.grad_accum),
        "estimated_model_plus_optimizer_memory_mb": float(est_total_mb),
        "peak_vram_allocated_mb": peak_vram_mb,
        "training_time_seconds": t_train.get("seconds", 0.0),
        "val_loss": best_val,
        "test_metrics": test_metrics,
        "calibration": calibration,
        "optimal_thresholds": optimal_thresholds,
        "history": history,
        "epochs_ran": epoch,
    }
    report_path = paths.dnn_reports / f"classification_{run_name}.json"
    save_json(report, report_path)

    # Plots
    plot_training_history(history, save_path=paths.dnn_images / f"cls_{run_name}_history.png", title=f"DNN Classification ({args.arch}, preset={args.preset})")
    plot_probability_distribution(y_true_test, y_prob_test, save_path=paths.dnn_images / f"cls_{run_name}_prob_dist.png", title="Predicted probability distribution")
    plot_roc_curve(y_true_test, y_prob_test, save_path=paths.dnn_images / f"cls_{run_name}_roc.png", title="ROC curve")
    plot_precision_recall_curve(y_true_test, y_prob_test, save_path=paths.dnn_images / f"cls_{run_name}_pr.png", title="Precision-Recall curve")
    plot_calibration_curve(y_true_test, y_prob_test, save_path=paths.dnn_images / f"cls_{run_name}_calibration.png", title="Calibration curve")
    plot_confusion_matrix(y_true_test, (y_prob_test >= 0.5).astype(int), save_path=paths.dnn_images / f"cls_{run_name}_cm.png", title="Confusion matrix (threshold=0.5)")
    print(f"Saved model to {model_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
