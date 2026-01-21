from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TensorSplits:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 1:
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
    return torch.tensor(arr, dtype=torch.float32)


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_splits(npz_path: Path, target_prefix: str) -> TensorSplits:
    data = load_npz(npz_path)
    X_train = _to_tensor(data["X_train"])
    X_val = _to_tensor(data["X_val"])
    X_test = _to_tensor(data["X_test"])
    y_train = _to_tensor(data[f"{target_prefix}_train"])
    y_val = _to_tensor(data[f"{target_prefix}_val"])
    y_test = _to_tensor(data[f"{target_prefix}_test"])
    return TensorSplits(X_train, y_train, X_val, y_val, X_test, y_test)


def make_loaders(
    splits: TensorSplits,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(splits.X_train, splits.y_train)
    val_ds = TensorDataset(splits.X_val, splits.y_val)
    test_ds = TensorDataset(splits.X_test, splits.y_test)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl
