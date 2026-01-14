"""Lightweight data prep for classic ML models on tennis matches.

This mirrors the prematch transformation used in `jupyter/Data_Prep.ipynb`:
- load the raw matches CSV (winner/loser rows)
- optionally clean out incomplete / retired matches
- randomly flip A/B orientation to avoid leakage
- derive the binary label (1 = player A is the original winner)
- engineer rank-based features and a round code
- clean critical NaNs
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Base directory for standalone execution (where this file lives)
ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DATA = ROOT / "all_matches.csv"
CLEANED_DIR = ROOT / "data/cleaned_matches"
CLEANED_FILE = CLEANED_DIR / "all_matches_clean.csv"

ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
CAT_COLS = ["surface", "round", "tourney_level"]


def clean_raw_matches(raw_csv: Path | str = DEFAULT_RAW_DATA, output_path: Path | str = CLEANED_FILE) -> Path:
    """Remove incomplete/retired matches and basic nulls, saving to cleaned file."""
    raw_csv = Path(raw_csv)
    if not raw_csv.is_absolute() and not raw_csv.exists():
        raw_csv = ROOT / raw_csv

    df = pd.read_csv(raw_csv, low_memory=False)
    original_rows = len(df)

    # Drop rows with missing winner/loser ids (incomplete entries)
    critical_id_cols = [c for c in ["winner_id", "loser_id"] if c in df.columns]
    if critical_id_cols:
        df = df.dropna(subset=critical_id_cols)

    # Filter out retired/walkover/unfinished matches based on score text if present
    if "score" in df.columns:
        bad_tokens = ["RET", "W/O", "DEF", "ABD", "Walkover", "walkover", "In Progress", "unfinished"]
        pattern = "|".join(bad_tokens)
        mask_bad = df["score"].astype(str).str.contains(pattern, case=False, na=False)
        df = df.loc[~mask_bad]

    # Ensure date parseable
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce", utc=True)

    # Save cleaned copy
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned matches: {original_rows} -> {len(df)} rows. Saved to {output_path}")
    return output_path


def _make_ab_row(row: pd.Series, flip: bool) -> dict:
    """Create a single A/B oriented row plus label."""
    if flip:
        # flip: player A is original loser (label=0)
        label = 0
        A_rank = row.get("loser_rank")
        B_rank = row.get("winner_rank")
    else:
        # no flip: player A is original winner (label=1)
        label = 1
        A_rank = row.get("winner_rank")
        B_rank = row.get("loser_rank")

    return {
        "surface": row.get("surface"),
        "round": row.get("round"),
        "tourney_level": row.get("tourney_level"),
        "best_of": row.get("best_of"),
        "draw_size": row.get("draw_size"),
        "year": row.get("year"),
        "match_id": row.get("match_id"),
        "playerA_rank": A_rank,
        "playerB_rank": B_rank,
        "label": label,
    }


def build_prematch_frame(csv_path: Path | str = DEFAULT_RAW_DATA, seed: int = 42, clean: bool = True) -> pd.DataFrame:
    """Load raw matches CSV and return a cleaned prematch frame ready for ML."""
    csv_path = Path(csv_path)
    if not csv_path.is_absolute() and not csv_path.exists():
        csv_path = ROOT / csv_path

    if clean:
        csv_path = clean_raw_matches(csv_path, CLEANED_FILE)

    df = pd.read_csv(csv_path, low_memory=False)

    # Parse dates and basic identifiers
    df["tourney_date"] = pd.to_datetime(df.get("tourney_date"), errors="coerce", utc=True)
    df["year"] = df["tourney_date"].dt.year
    df["match_id"] = df.index

    rng = np.random.default_rng(seed)
    flip_mask = rng.random(len(df)) < 0.5

    rows = [_make_ab_row(row, bool(flip_mask[i])) for i, row in df.iterrows()]
    prematch = pd.DataFrame(rows)

    # Feature engineering
    prematch["rank_diff"] = prematch["playerA_rank"] - prematch["playerB_rank"]
    prematch["log_playerA_rank"] = np.log1p(prematch["playerA_rank"])
    prematch["log_playerB_rank"] = np.log1p(prematch["playerB_rank"])
    prematch["log_rank_diff"] = prematch["log_playerA_rank"] - prematch["log_playerB_rank"]
    prematch["round_code"] = prematch["round"].map(ROUND_ORDER)

    # Critical columns must be present
    critical_cols = [
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "year",
        "playerA_rank",
        "playerB_rank",
    ]
    prematch = prematch.dropna(subset=critical_cols).reset_index(drop=True)

    # Fill draw_size with per-year median (fallback to global median)
    if "draw_size" in prematch.columns:
        med_by_year = prematch.groupby("year")["draw_size"].transform("median")
        prematch["draw_size"] = prematch["draw_size"].fillna(med_by_year)
        prematch["draw_size"] = prematch["draw_size"].fillna(prematch["draw_size"].median())

    # Drop remaining NaNs on features (excluding label/match_id)
    feature_cols = [c for c in prematch.columns if c not in ["label", "match_id"]]
    prematch = prematch.dropna(subset=feature_cols).reset_index(drop=True)

    prematch["label"] = prematch["label"].astype(int)
    return prematch


def load_features_and_target(csv_path: Path | str = DEFAULT_RAW_DATA, clean: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience: return (X, y) for modeling."""
    prematch = build_prematch_frame(csv_path, clean=clean)
    y = prematch["label"]
    X = prematch.drop(columns=["label", "match_id"])
    return X, y


if __name__ == "__main__":
    clean_raw_matches()
