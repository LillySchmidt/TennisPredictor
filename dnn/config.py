from pathlib import Path
from dataclasses import dataclass


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parent
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"
    artifacts: Path = project_root / "artifacts"
    models: Path = project_root / "models"
    reports: Path = project_root / "reports"


@dataclass
class DataConfig:
    raw_csv: Path = Paths().data_raw / "all_matches.csv"
    processed_npz: Path = Paths().data_processed / "dataset.npz"
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    men_levels: tuple = ("A", "G", "M")  # ATP / Grand Slam / Masters-equivalent codes


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    model_size: str = "50k"  # options: 50k, 500k, 5m, 10m, 50m


paths = Paths()
data_cfg = DataConfig()
train_cfg = TrainConfig()
