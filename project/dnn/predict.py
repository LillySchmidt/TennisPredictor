import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dnn.model import build_classifier  # noqa: E402
from dnn.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load a trained DNN classifier and run inference on an .npz file.")
    p.add_argument("--model", type=Path, required=True, help="Path to a saved .pt from train_classification.py")
    p.add_argument("--npz", type=Path, required=True, help="NPZ containing X (shape [N,D]) and optionally y")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt.get("config", {})
    input_dim = int(ckpt.get("input_dim"))
    model = build_classifier(
        input_dim=input_dim,
        preset=cfg.get("preset"),
        arch=cfg.get("arch", "plain"),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        num_layers=int(cfg.get("num_layers", 6)),
        dropout=float(cfg.get("dropout", 0.15)),
        norm=cfg.get("norm", "batchnorm"),
        activation=cfg.get("activation", "relu"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = np.load(args.npz)
    X = data["X"].astype(np.float32)
    y = data["y"] if "y" in data.files else None

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= args.threshold).astype(int)

    print(f"N={len(probs)} | threshold={args.threshold}")
    print("probabilities (first 10):", probs[:10])
    print("predictions (first 10):", preds[:10])
    if y is not None:
        y = y.reshape(-1)
        acc = (preds == y).mean()
        print(f"accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
