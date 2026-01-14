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
from dnn.metrics import classification_metrics  # noqa: E402
from dnn.model import build_mlp, count_parameters  # noqa: E402
from dnn.utils import ensure_dir, get_device, save_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classification model (winner prediction).")
    parser.add_argument("--data", type=Path, default=data_cfg.processed_npz, help="Path to processed dataset npz")
    parser.add_argument(
        "--model_size",
        type=str,
        default=train_cfg.model_size,
        choices=["50k", "500k", "5m", "10m", "50m"],
    )
    parser.add_argument("--epochs", type=int, default=train_cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=train_cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=train_cfg.weight_decay)
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--random_state", type=int, default=data_cfg.random_state)
    return parser.parse_args()


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        # Targets come in with an extra singleton dim; flatten to match logits.
        yb = yb.to(device).view(-1)
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
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


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)
    device = get_device()

    splits = build_splits(args.data, target_prefix="cls")
    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=args.batch_size)

    input_dim = splits.X_train.shape[1]
    model = build_mlp(input_dim=input_dim, output_dim=1, model_size=args.model_size).to(device)
    params = count_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])

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

    test_metrics = evaluate(model, test_loader, device, criterion)

    # Save model and report
    ensure_dir(paths.models)
    model_path = paths.models / f"cls_{args.model_size}.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "model_size": args.model_size}, model_path)

    report = {
        "model_size": args.model_size,
        "params": params,
        "device": str(device),
        "val_loss": best_val,
        "test_metrics": test_metrics,
        "epochs_ran": epoch,
    }
    report_path = paths.reports / f"classification_{args.model_size}.json"
    save_json(report, report_path)
    print(f"Saved model to {model_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
