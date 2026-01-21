from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Filesystem layout.

    The repository is intentionally simple:
    - shared data/artifacts live at the repo root
    - model-specific outputs live under ./dnn and ./ml
    """

    project_root: Path = Path(__file__).resolve().parent.parent

    # Shared
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"
    artifacts: Path = project_root / "artifacts"
    comparison: Path = project_root / "comparison_dnn"

    # DNN outputs
    dnn_dir: Path = project_root / "dnn"
    dnn_models: Path = dnn_dir / "models"
    dnn_reports: Path = dnn_dir / "reports"
    dnn_images: Path = dnn_dir / "images"

    # NOTE: This repo focuses on DNNs only.


@dataclass(frozen=True)
class DataConfig:
    raw_csv: Path = Paths().data_raw / "all_matches.csv"
    processed_npz: Path = Paths().data_processed / "dataset.npz"
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    men_levels: tuple[str, ...] = ("A", "G", "M")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    preset: str = "medium"  # tiny, small, medium, large


paths = Paths()
data_cfg = DataConfig()
train_cfg = TrainConfig()
