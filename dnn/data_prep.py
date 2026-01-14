import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dnn.config import data_cfg, paths  # noqa: E402
from dnn.utils import ensure_dir, save_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tennis matches data for DNN training.")
    parser.add_argument("--raw", type=Path, default=data_cfg.raw_csv, help="Path to raw all_matches.csv")
    parser.add_argument("--out", type=Path, default=data_cfg.processed_npz, help="Output npz path for processed data")
    parser.add_argument("--test_size", type=float, default=data_cfg.test_size, help="Test split size")
    parser.add_argument("--val_size", type=float, default=data_cfg.val_size, help="Validation split size (fraction of train)")
    parser.add_argument("--random_state", type=int, default=data_cfg.random_state, help="Random seed")
    return parser.parse_args()


def parse_score(score: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """
    Returns (games_diff, sets_diff) if parseable else None.
    Handles standard set strings and tiebreak markers; skips retirements/walkovers.
    """
    if not isinstance(score, str) or not score.strip():
        return None
    raw = score.lower()
    if "ret" in raw or "w/o" in raw or "walk" in raw:
        return None
    tokens = score.replace("\u00a0", " ").split()
    games_w = games_l = sets_w = sets_l = 0
    seen = False
    for tok in tokens:
        tok = tok.strip()
        if not tok or "-" not in tok:
            continue
        base = tok.split("-")
        if len(base) < 2:
            continue
        try:
            g_w = int(base[0].split("(")[0])
            g_l = int(base[1].split("(")[0])
        except ValueError:
            continue
        games_w += g_w
        games_l += g_l
        if g_w > g_l:
            sets_w += 1
        elif g_l > g_w:
            sets_l += 1
        seen = True
    if not seen:
        return None
    games_diff = games_w - games_l
    sets_diff = sets_w - sets_l
    return games_diff, sets_diff


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
    df["seed_diff"] = df["winner_seed"] - df["loser_seed"]
    df["age_diff"] = df["winner_age"] - df["loser_age"]
    df["ht_diff"] = df["winner_ht"] - df["loser_ht"]
    df["ace_diff"] = df["w_ace"] - df["l_ace"]
    df["df_diff"] = df["w_df"] - df["l_df"]
    df["svpt_diff"] = df["w_svpt"] - df["l_svpt"]
    df["bp_saved_diff"] = df["w_bpSaved"] - df["l_bpSaved"]
    df["bp_faced_diff"] = df["w_bpFaced"] - df["l_bpFaced"]
    return df


def _swap_winner_loser(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a view where winner/loser stats are swapped (loser perspective).
    Used to generate negative-class samples for classification.
    """
    swap_pairs = [
        ("winner_rank", "loser_rank"),
        ("winner_seed", "loser_seed"),
        ("winner_age", "loser_age"),
        ("winner_ht", "loser_ht"),
        ("w_ace", "l_ace"),
        ("w_df", "l_df"),
        ("w_svpt", "l_svpt"),
        ("w_1stIn", "l_1stIn"),
        ("w_1stWon", "l_1stWon"),
        ("w_2ndWon", "l_2ndWon"),
        ("w_bpSaved", "l_bpSaved"),
        ("w_bpFaced", "l_bpFaced"),
    ]
    df_swapped = df.copy()
    for a, b in swap_pairs:
        df_swapped[a], df_swapped[b] = df_swapped[b], df_swapped[a]
    return df_swapped


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)

    raw_path = args.raw
    ensure_dir(paths.data_processed)
    ensure_dir(paths.artifacts)
    ensure_dir(paths.models)
    ensure_dir(paths.reports)

    df = pd.read_csv(raw_path, parse_dates=["tourney_date"])
    df = df[df["tourney_level"].str.upper().isin(data_cfg.men_levels)]
    df = df.drop_duplicates().reset_index(drop=True)

    # Critical columns for retention
    cat_cols = ["surface", "round", "tourney_level"]
    critical_cols = cat_cols + ["best_of", "draw_size", "winner_rank", "loser_rank", "score"]
    df = df.dropna(subset=critical_cols)

    # Parse score targets
    parsed = df["score"].apply(parse_score)
    df["games_diff"] = parsed.apply(lambda x: x[0] if x else np.nan)
    df["sets_diff"] = parsed.apply(lambda x: x[1] if x else np.nan)
    df = df.dropna(subset=["games_diff", "sets_diff"], how="all")
    df["reg_target"] = df["games_diff"].fillna(df["sets_diff"])
    df["cls_target"] = 1  # winner is positive class (always given order)

    # Augment with loser-perspective rows to introduce negative class.
    df_neg = _swap_winner_loser(df)
    df_neg["cls_target"] = 0
    df_neg["reg_target"] = -df_neg["reg_target"]
    df_neg["games_diff"] = -df_neg["games_diff"]
    df_neg["sets_diff"] = -df_neg["sets_diff"]
    df = pd.concat([df, df_neg], ignore_index=True)

    # Feature selection
    numeric_cols = [
        "best_of",
        "draw_size",
        "winner_rank",
        "loser_rank",
        "winner_seed",
        "loser_seed",
        "winner_age",
        "loser_age",
        "winner_ht",
        "loser_ht",
        "w_ace",
        "l_ace",
        "w_df",
        "l_df",
        "w_svpt",
        "l_svpt",
        "w_1stIn",
        "l_1stIn",
        "w_1stWon",
        "l_1stWon",
        "w_2ndWon",
        "l_2ndWon",
        "w_bpSaved",
        "l_bpSaved",
        "w_bpFaced",
        "l_bpFaced",
    ]
    df = build_features(df)
    numeric_cols += [
        "rank_diff",
        "seed_diff",
        "age_diff",
        "ht_diff",
        "ace_diff",
        "df_diff",
        "svpt_diff",
        "bp_saved_diff",
        "bp_faced_diff",
    ]

    # Final feature frame
    feature_cols = numeric_cols + cat_cols
    df_features = df[feature_cols].copy()
    y_cls = df["cls_target"].astype(int)
    y_reg = df["reg_target"].astype(float)

    # Split data
    X_train, X_temp, y_cls_train, y_cls_temp, y_reg_train, y_reg_temp = train_test_split(
        df_features,
        y_cls,
        y_reg,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_cls,
    )
    val_size_adj = args.val_size / (1 - args.test_size)
    X_val, X_test, y_cls_val, y_cls_test, y_reg_val, y_reg_test = train_test_split(
        X_temp,
        y_cls_temp,
        y_reg_temp,
        test_size=1 - val_size_adj,
        random_state=args.random_state,
        stratify=y_cls_temp,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Save arrays
    out_path = args.out
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        X_train=X_train_t.astype(np.float32),
        X_val=X_val_t.astype(np.float32),
        X_test=X_test_t.astype(np.float32),
        cls_train=y_cls_train.to_numpy().astype(np.float32),
        cls_val=y_cls_val.to_numpy().astype(np.float32),
        cls_test=y_cls_test.to_numpy().astype(np.float32),
        reg_train=y_reg_train.to_numpy().astype(np.float32),
        reg_val=y_reg_val.to_numpy().astype(np.float32),
        reg_test=y_reg_test.to_numpy().astype(np.float32),
    )

    # Save processor and metadata
    processor_path = paths.artifacts / "processor.joblib"
    joblib.dump(preprocessor, processor_path)
    meta = {
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "shapes": {
            "X_train": X_train_t.shape,
            "X_val": X_val_t.shape,
            "X_test": X_test_t.shape,
        },
        "splits": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "random_state": args.random_state,
    }
    save_json(meta, paths.artifacts / "metadata.json")
    print(f"Saved processed arrays to {out_path}")
    print(f"Processor saved to {processor_path}")


if __name__ == "__main__":
    main()
