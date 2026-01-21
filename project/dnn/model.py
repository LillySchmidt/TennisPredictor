from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch
from torch import nn


NormType = Literal["batchnorm", "layernorm", "none"]
ArchType = Literal["plain", "residual"]


@dataclass(frozen=True)
class ModelPreset:
    """A compact definition that is easy to keep within modest VRAM budgets.

    Note: VRAM use depends heavily on batch size and optimizer states.
    These presets are designed to fit comfortably in <= 6 GB VRAM for typical
    tabular input dimensions (tens to a few hundred features) with AMP enabled.
    """

    hidden_dim: int
    num_layers: int
    dropout: float
    arch: ArchType
    norm: NormType


PRESETS: dict[str, ModelPreset] = {
    # Parameter counts scale roughly with hidden_dim^2 * num_layers.
    # These are deliberately conservative for 6GB GPUs.
    "tiny": ModelPreset(hidden_dim=256, num_layers=4, dropout=0.10, arch="plain", norm="batchnorm"),
    "small": ModelPreset(hidden_dim=512, num_layers=6, dropout=0.15, arch="plain", norm="batchnorm"),
    "medium": ModelPreset(hidden_dim=768, num_layers=8, dropout=0.20, arch="residual", norm="layernorm"),
    "large": ModelPreset(hidden_dim=1024, num_layers=10, dropout=0.25, arch="residual", norm="layernorm"),
}


def _make_norm(norm: NormType, dim: int) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm1d(dim)
    if norm == "layernorm":
        return nn.LayerNorm(dim)
    return nn.Identity()


class PlainMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        norm: NormType,
        activation: Literal["relu", "gelu"] = "relu",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act: nn.Module
        if activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), _make_norm(norm, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), _make_norm(norm, hidden_dim), act, nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        norm: NormType,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = _make_norm(norm, dim)
        self.norm2 = _make_norm(norm, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm style residual MLP block
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.norm2(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        norm: NormType,
        activation: Literal["relu", "gelu"] = "gelu",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(dim=hidden_dim, dropout=dropout, norm=norm, activation=activation)
                for _ in range(num_layers)
            ]
        )
        self.head_norm = _make_norm(norm, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.head_norm(h)
        return self.head(h)


def build_classifier(
    *,
    input_dim: int,
    preset: Optional[str] = None,
    arch: ArchType = "plain",
    hidden_dim: int = 512,
    num_layers: int = 6,
    dropout: float = 0.15,
    norm: NormType = "batchnorm",
    activation: Literal["relu", "gelu"] = "relu",
) -> nn.Module:
    if preset is not None:
        p = PRESETS.get(preset)
        if p is None:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
        arch = p.arch
        hidden_dim = p.hidden_dim
        num_layers = p.num_layers
        dropout = p.dropout
        norm = p.norm
        # Residual defaults tend to work better with GELU on tabular inputs.
        if arch == "residual":
            activation = "gelu"

    if arch == "residual":
        return ResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            activation=activation,
            output_dim=1,
        )
    return PlainMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        norm=norm,
        activation=activation,
        output_dim=1,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_optimizer_memory_bytes(params: int, optimizer: Literal["adamw", "sgd"] = "adamw", dtype_bytes: int = 4) -> int:
    """Very rough optimizer-state estimate.

    - SGD: 1x params (momentum) worst-case
    - Adam/AdamW: ~2x params for exp_avg and exp_avg_sq
    """

    if optimizer == "sgd":
        return int(params * dtype_bytes)
    return int(2 * params * dtype_bytes)


def estimate_model_memory_bytes(params: int, dtype_bytes: int = 4) -> int:
    return int(params * dtype_bytes)
