# DNN Tennis Predictor

This package prepares ATP men's match data from `jupyter/data/all_matches.csv`, produces train/val/test splits, and trains/validates PyTorch MLPs at three sizes (~5M, ~10M, ~50M params) for both classification (winner) and regression (games differential).

## Quickstart
```bash
# 1) Prep data (uses random_state=42 for all splits/ops)
python dnn/data_prep.py --raw /home/stijn/lilly/thesis/tennis-predictor-final/jupyter/data/all_matches.csv

# 2) Train models (GPU auto-detected; override hyperparams via CLI)
python dnn/train_classification.py --model_size 5m --epochs 30 --batch_size 512
python dnn/train_regression.py --model_size 10m --epochs 40

# 3) Validate on held-out test split
python dnn/validate.py --task cls --model_size 5m
python dnn/validate.py --task reg --model_size 10m
```

## Outputs
- `dnn/data/processed/dataset.npz` — numpy arrays for features/targets (train/val/test) plus metadata.
- `dnn/artifacts/processor.joblib` — fitted ColumnTransformer (imputers, scaler, one-hot encoder) and feature names.
- `dnn/models/` — trained weights (`cls_{size}.pt`, `reg_{size}.pt`).
- `dnn/reports/` — JSON metrics for train/val/test.

## Notes
- Men's matches filtered via `tourney_level` in {A, G, M}.
- Score parsing covers standard sets, tiebreak markers, and skips retirements/walkovers; regression uses games differential, falling back to sets differential when needed.
- Deterministic: seeds for Python/NumPy/PyTorch and train/val/test splits fixed to 42.
