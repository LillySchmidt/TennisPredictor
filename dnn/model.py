from typing import List

import torch
from torch import nn


PRESETS = {
    # ~66k params @ input_dim=52, depth 3 keeps it small
    "50k": {"hidden": [192, 192, 96], "dropout": 0.15},
    # ~511k params @ input_dim=52
    "500k": {"hidden": [576, 576, 256], "dropout": 0.2},
    "5m": {"hidden": [768, 768, 768, 768, 768, 768], "dropout": 0.2},
    "10m": {"hidden": [1024, 1024, 1024, 1024, 1024, 1024, 1024], "dropout": 0.25},
    "50m": {"hidden": [2560, 2560, 2560, 2560, 2560, 2560, 2560, 2560], "dropout": 0.35},
}


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(input_dim: int, output_dim: int, model_size: str) -> MLP:
    cfg = PRESETS.get(model_size)
    if cfg is None:
        raise ValueError(f"Unknown model_size '{model_size}', choose from {list(PRESETS.keys())}")
    return MLP(input_dim=input_dim, output_dim=output_dim, hidden=cfg["hidden"], dropout=cfg["dropout"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
