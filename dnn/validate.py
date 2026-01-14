import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dnn.config import data_cfg, paths  # noqa: E402
from dnn.datasets import build_splits, make_loaders  # noqa: E402
from dnn.metrics import classification_metrics, regression_metrics  # noqa: E402
from dnn.model import build_mlp  # noqa: E402
from dnn.utils import get_device, save_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained models on held-out test split.")
    parser.add_argument("--task", choices=["cls", "reg"], required=True, help="Task to validate (cls or reg)")
    parser.add_argument("--model_size", type=str, default="5m", choices=["50k", "500k", "5m", "10m", "50m"])
    parser.add_argument("--data", type=Path, default=data_cfg.processed_npz)
    parser.add_argument("--model_path", type=Path, default=None, help="Optional explicit model path")
    parser.add_argument("--random_state", type=int, default=data_cfg.random_state)
    return parser.parse_args()


def load_model(model_path: Path, input_dim: int, model_size: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    model = build_mlp(input_dim=input_dim, output_dim=1, model_size=model_size)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def evaluate_cls(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(prob)
            targets.append(yb.view(-1).cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    return classification_metrics(y_true, y_prob)


def evaluate_reg(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).squeeze(1).cpu().numpy()
            preds.append(pred)
            targets.append(yb.view(-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return regression_metrics(y_true, y_pred)


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)
    device = get_device()

    target_prefix = "cls" if args.task == "cls" else "reg"
    splits = build_splits(args.data, target_prefix=target_prefix)
    _, _, test_loader = make_loaders(splits, batch_size=1024, shuffle_train=False)

    input_dim = splits.X_train.shape[1]
    if args.model_path:
        model_path = args.model_path
    else:
        default_name = "cls" if args.task == "cls" else "reg"
        model_path = paths.models / f"{default_name}_{args.model_size}.pt"

    model = load_model(model_path, input_dim=input_dim, model_size=args.model_size, device=device)

    if args.task == "cls":
        metrics = evaluate_cls(model, test_loader, device)
    else:
        metrics = evaluate_reg(model, test_loader, device)

    out_name = "classification" if args.task == "cls" else "regression"
    report_path = paths.reports / f"{out_name}_test_{args.model_size}.json"
    save_json({"model": str(model_path), "metrics": metrics, "device": str(device)}, report_path)
    print(f"Saved {out_name} test metrics to {report_path}")
    print(metrics)


if __name__ == "__main__":
    main()
