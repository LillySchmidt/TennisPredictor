import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ML_DIR = REPO_ROOT / "ML"
ELO_DIR = REPO_ROOT / "elo-system"

for path in (REPO_ROOT, ML_DIR, ELO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ML import data_prep as ml_dp  # noqa: E402
from tennis_elo import EloConfig, TennisEloSystem  # noqa: E402


ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ELO-ML prematch dataset with Elo features.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=REPO_ROOT / "project" / "data" / "raw" / "all_matches.csv",
        help="Path to all_matches.csv (default: project/data/raw/all_matches.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "elo_ml_dataset.csv",
        help="Output CSV path for the ELO-ML dataset.",
    )
    parser.add_argument(
        "--cleaned-out",
        type=Path,
        default=ROOT / "data" / "cleaned_matches" / "all_matches_clean.csv",
        help="Where to store the cleaned matches CSV.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for A/B flipping.")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning and use raw CSV as-is.")
    parser.add_argument("--no-decay", action="store_true", help="Disable Elo inactivity decay.")
    parser.add_argument("--no-mov", action="store_true", help="Disable Elo margin-of-victory adjustment.")
    return parser.parse_args()


def _extract_date(date_str: str) -> Optional[datetime]:
    """Parse the dataset's custom date encoding used by elo-system."""
    try:
        frac = str(date_str).split(".")[1].split("+")[0]
        year = int(frac[1:5])
        month = int(frac[5:7])
        day = int(frac[7:9])
        return datetime(year=year, month=month, day=day)
    except Exception:
        return None


def _make_ab_row(row: pd.Series, flip: bool, elo_w: float, elo_l: float) -> dict:
    """Create a single A/B oriented row with Elo features."""
    if flip:
        label = 0
        A_rank = row.get("loser_rank")
        B_rank = row.get("winner_rank")
        elo_a, elo_b = elo_l, elo_w
    else:
        label = 1
        A_rank = row.get("winner_rank")
        B_rank = row.get("loser_rank")
        elo_a, elo_b = elo_w, elo_l

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
        "elo_a": elo_a,
        "elo_b": elo_b,
        "label": label,
    }


def build_elo_prematch_frame(
    csv_path: Path,
    cleaned_out: Path,
    seed: int = 42,
    clean: bool = True,
    elo_config: Optional[EloConfig] = None,
) -> Tuple[pd.DataFrame, EloConfig]:
    if clean:
        csv_path = ml_dp.clean_raw_matches(csv_path, cleaned_out)

    df = pd.read_csv(csv_path, low_memory=False)

    # ML-style year feature (matches ML data_prep exactly)
    raw_date = df.get("tourney_date")
    df["tourney_date"] = pd.to_datetime(raw_date, errors="coerce", utc=True)
    df["year"] = df["tourney_date"].dt.year
    df["match_id"] = df.index

    # Elo requires chronological processing using the dataset's custom date encoding.
    df["match_date"] = raw_date.apply(_extract_date)
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    # Stable player identifiers for Elo (fallback to IDs if names are missing)
    df["winner_key"] = df["winner_name"]
    df["loser_key"] = df["loser_name"]
    if "winner_id" in df.columns:
        df["winner_key"] = df["winner_key"].fillna(df["winner_id"].astype(str))
    if "loser_id" in df.columns:
        df["loser_key"] = df["loser_key"].fillna(df["loser_id"].astype(str))
    df["winner_key"] = df["winner_key"].fillna(df["match_id"].astype(str))
    df["loser_key"] = df["loser_key"].fillna(df["match_id"].astype(str))

    config = elo_config or EloConfig()
    elo = TennisEloSystem(config)

    # Precompute Elo ratings for each match (before update)
    elo_by_match = {}
    for _, row in df.sort_values("match_date").iterrows():
        winner = row["winner_key"]
        loser = row["loser_key"]
        match_date = row["match_date"]
        surface = row.get("surface", None)
        tourney_level = row.get("tourney_level", "")
        round_name = row.get("round", "")
        score = row.get("score", "")

        # Apply decay prior to prediction
        elo.apply_decay(winner, match_date)
        elo.apply_decay(loser, match_date)

        elo_w = elo._get_rating(winner)
        elo_l = elo._get_rating(loser)
        elo_by_match[int(row["match_id"])] = (float(elo_w), float(elo_l))

        # Update ratings with actual outcome
        elo.process_match(
            winner=winner,
            loser=loser,
            match_date=match_date,
            tourney_level=tourney_level,
            round_name=round_name,
            score=score,
            surface=surface,
        )

    rng = np.random.default_rng(seed)
    flip_mask = rng.random(len(df)) < 0.5

    rows = []
    for i, row in df.iterrows():
        match_id = int(row["match_id"])
        elo_w, elo_l = elo_by_match.get(match_id, (np.nan, np.nan))
        rows.append(_make_ab_row(row, bool(flip_mask[i]), elo_w, elo_l))

    prematch = pd.DataFrame(rows)
    prematch["rank_diff"] = prematch["playerA_rank"] - prematch["playerB_rank"]
    prematch["log_playerA_rank"] = np.log1p(prematch["playerA_rank"])
    prematch["log_playerB_rank"] = np.log1p(prematch["playerB_rank"])
    prematch["log_rank_diff"] = prematch["log_playerA_rank"] - prematch["log_playerB_rank"]
    prematch["round_code"] = prematch["round"].map(ROUND_ORDER)
    prematch["elo_diff"] = prematch["elo_a"] - prematch["elo_b"]

    # ML-style critical columns
    critical_cols = [
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "year",
        "playerA_rank",
        "playerB_rank",
        "elo_a",
        "elo_b",
    ]
    prematch = prematch.dropna(subset=critical_cols).reset_index(drop=True)

    # Fill draw_size with per-year median (fallback to global median)
    if "draw_size" in prematch.columns:
        med_by_year = prematch.groupby("year")["draw_size"].transform("median")
        prematch["draw_size"] = prematch["draw_size"].fillna(med_by_year)
        prematch["draw_size"] = prematch["draw_size"].fillna(prematch["draw_size"].median())

    feature_cols = [c for c in prematch.columns if c not in ["label", "match_id"]]
    prematch = prematch.dropna(subset=feature_cols).reset_index(drop=True)
    prematch["label"] = prematch["label"].astype(int)

    return prematch, config


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cleaned_out.parent.mkdir(parents=True, exist_ok=True)

    prematch, config = build_elo_prematch_frame(
        csv_path=args.raw,
        cleaned_out=args.cleaned_out,
        seed=args.random_state,
        clean=not args.no_clean,
        elo_config=EloConfig(decay_enabled=not args.no_decay, mov_enabled=not args.no_mov),
    )

    prematch.to_csv(args.out, index=False)

    meta = {
        "raw_csv": str(args.raw),
        "cleaned_csv": str(args.cleaned_out),
        "rows": int(len(prematch)),
        "features": [c for c in prematch.columns if c not in ["label", "match_id"]],
        "random_state": args.random_state,
        "elo_config": asdict(config),
    }

    meta_path = args.out.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote dataset: {args.out} ({len(prematch)} rows)")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
