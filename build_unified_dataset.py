#!/usr/bin/env python3
"""
Build a unified train/validation/test split used for all updated experiments.
- Loads the ELO-ML dataset (495,374 prematch observations with 19 features).
- Applies a stratified 70 / 15 / 15 split with seed 42.
- Saves train/val/test CSVs plus metadata (sizes, seed, feature lists).
- Also prepares a 16-feature version for classical ML and a 19-feature version for ELO-ML.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT_DIR = ROOT / "unified_data"
OUT_DIR.mkdir(exist_ok=True)

ELO_ML_CSV = REPO / "ELO-ML" / "data" / "elo_ml_dataset.csv"

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main():
    print(f"Loading {ELO_ML_CSV} ...")
    df = pd.read_csv(ELO_ML_CSV, low_memory=False)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Define feature groups
    base_features = [
        "surface", "round", "tourney_level", "best_of", "draw_size", "year",
        "playerA_rank", "playerB_rank",
        "rank_diff", "log_playerA_rank", "log_playerB_rank", "log_rank_diff", "round_code",
    ]
    elo_features = ["elo_a", "elo_b", "elo_diff"]
    ml_features = base_features
    eloml_features = base_features + elo_features

    target = "label"

    # Stratified split: first separate train vs temp, then val vs test.
    X = df[eloml_features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - TRAIN_FRAC),
        random_state=RANDOM_STATE,
        stratify=y,
    )
    val_ratio = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    train_idx = X_train.index
    val_idx = X_val.index
    test_idx = X_test.index

    print(f"Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.2f}%)")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/len(df)*100:.2f}%)")
    print(f"Test:  {len(test_idx)} ({len(test_idx)/len(df)*100:.2f}%)")

    # Save full splits with all features
    df.loc[train_idx].to_csv(OUT_DIR / "train.csv", index=False)
    df.loc[val_idx].to_csv(OUT_DIR / "val.csv", index=False)
    df.loc[test_idx].to_csv(OUT_DIR / "test.csv", index=False)

    # Save lightweight versions with feature subsets
    for subset, cols in [("ml", ml_features), ("eloml", eloml_features)]:
        df.loc[train_idx, cols + [target]].to_csv(OUT_DIR / f"{subset}_train.csv", index=False)
        df.loc[val_idx, cols + [target]].to_csv(OUT_DIR / f"{subset}_val.csv", index=False)
        df.loc[test_idx, cols + [target]].to_csv(OUT_DIR / f"{subset}_test.csv", index=False)

    metadata = {
        "source": str(ELO_ML_CSV),
        "total_rows": len(df),
        "random_state": RANDOM_STATE,
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "ml_features": ml_features,
        "eloml_features": eloml_features,
        "elo_features": elo_features,
        "target": target,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved unified splits to {OUT_DIR}")


if __name__ == "__main__":
    main()
