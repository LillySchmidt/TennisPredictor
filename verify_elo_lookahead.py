#!/usr/bin/env python3
"""
Verify that Elo ratings used in the ELO-ML dataset are strictly pre-match:
- Recompute Elo chronologically over the cleaned match corpus.
- For every match, record the rating *before* process_match() updates it.
- Compare those recomputed pre-match ratings with the elo_a / elo_b values
  stored in ELO-ML/data/elo_ml_dataset.csv.
- Also test an "incorrect" variant that updates ratings *before* capture to
  quantify how large a look-ahead effect would be.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load project modules manually (the hyphenated package names and the import
# shadowing observed when running scripts make direct package imports unreliable).
elo_ml_dp = load_module("elo_ml_data_prep", REPO / "ELO-ML" / "data_prep.py")
ml_dp = load_module("ml_data_prep", REPO / "ML" / "data_prep.py")
tennis_elo = load_module("tennis_elo", REPO / "elo-system" / "tennis_elo.py")

build_elo_prematch_frame = elo_ml_dp.build_elo_prematch_frame
_extract_date = elo_ml_dp._extract_date
EloConfig = tennis_elo.EloConfig
TennisEloSystem = tennis_elo.TennisEloSystem

DATASET_CSV = REPO / "ELO-ML" / "data" / "elo_ml_dataset.csv"
RAW_CSV = REPO / "project" / "data" / "raw" / "all_matches.csv"
CLEANED_CSV = ROOT / "cleaned_matches_verify.csv"


def recompute_pre_match_elo(cleaned_csv: Path, update_first: bool = False):
    """Return DataFrame with correct pre-match ratings; if update_first=True,
    simulate a look-ahead-biased pipeline where ratings are updated before capture."""
    df = pd.read_csv(cleaned_csv, low_memory=False)
    raw_date = df.get("tourney_date")
    df["tourney_date"] = pd.to_datetime(raw_date, errors="coerce", utc=True)
    df["match_id"] = df.index
    df["match_date"] = raw_date.apply(_extract_date)
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    df["winner_key"] = df["winner_name"]
    df["loser_key"] = df["loser_name"]
    if "winner_id" in df.columns:
        df["winner_key"] = df["winner_key"].fillna(df["winner_id"].astype(str))
    if "loser_id" in df.columns:
        df["loser_key"] = df["loser_key"].fillna(df["loser_id"].astype(str))
    df["winner_key"] = df["winner_key"].fillna(df["match_id"].astype(str))
    df["loser_key"] = df["loser_key"].fillna(df["match_id"].astype(str))

    config = EloConfig()
    elo = TennisEloSystem(config)
    records = []
    for _, row in df.sort_values("match_date").iterrows():
        mid = int(row["match_id"])
        winner = row["winner_key"]
        loser = row["loser_key"]
        match_date = row["match_date"]
        surface = row.get("surface", None)
        level = row.get("tourney_level", "")
        rnd = row.get("round", "")
        score = row.get("score", "")

        elo.apply_decay(winner, match_date)
        elo.apply_decay(loser, match_date)

        if update_first:
            elo.process_match(winner, loser, match_date, level, rnd, score, surface)
            elo_w = elo._get_rating(winner)
            elo_l = elo._get_rating(loser)
        else:
            elo_w = elo._get_rating(winner)
            elo_l = elo._get_rating(loser)
            elo.process_match(winner, loser, match_date, level, rnd, score, surface)

        records.append({"match_id": mid, "elo_w_pre": elo_w, "elo_l_pre": elo_l})
    return pd.DataFrame(records)


def main():
    print("Cleaning raw matches...")
    cleaned_csv = ml_dp.clean_raw_matches(RAW_CSV, CLEANED_CSV)

    print("Loading stored ELO-ML dataset...")
    stored = pd.read_csv(DATASET_CSV, low_memory=False)

    # Reconstruct winner-perspective pre-match ratings from stored dataset.
    stored["match_id"] = stored["match_id"].astype(int)
    stored["elo_w_stored"] = np.where(stored["label"] == 1, stored["elo_a"], stored["elo_b"])
    stored["elo_l_stored"] = np.where(stored["label"] == 1, stored["elo_b"], stored["elo_a"])

    print("Recomputing CORRECT pre-match Elo ratings...")
    correct = recompute_pre_match_elo(cleaned_csv, update_first=False)
    merged = stored[["match_id", "elo_w_stored", "elo_l_stored"]].merge(
        correct, on="match_id", how="inner"
    )

    diff_w = (merged["elo_w_stored"] - merged["elo_w_pre"]).abs()
    diff_l = (merged["elo_l_stored"] - merged["elo_l_pre"]).abs()
    print(f"Matches compared: {len(merged)}")
    print(f"Winner rating max |stored - recomputed|: {diff_w.max():.6f}")
    print(f"Loser  rating max |stored - recomputed|: {diff_l.max():.6f}")
    print(f"Winner rating mean |stored - recomputed|: {diff_w.mean():.6f}")
    print(f"Loser  rating mean |stored - recomputed|: {diff_l.mean():.6f}")

    # Spot-check: compare with an update-first (look-ahead-biased) variant
    print("\nRecomputing LOOK-AHEAD-BIASED Elo ratings (update before capture)...")
    biased = recompute_pre_match_elo(cleaned_csv, update_first=True)
    merged_biased = stored[["match_id", "elo_w_stored", "elo_l_stored"]].merge(
        biased, on="match_id", how="inner"
    )
    diff_w_b = (merged_biased["elo_w_stored"] - merged_biased["elo_w_pre"]).abs()
    diff_l_b = (merged_biased["elo_l_stored"] - merged_biased["elo_l_pre"]).abs()
    print(f"Look-ahead variant max abs diff winner: {diff_w_b.max():.2f}")
    print(f"Look-ahead variant max abs diff loser:  {diff_l_b.max():.2f}")
    print(f"Look-ahead variant mean abs diff winner: {diff_w_b.mean():.2f}")
    print(f"Look-ahead variant mean abs diff loser:  {diff_l_b.mean():.2f}")

    # Threshold-based verdict
    tol = 1e-4
    if diff_w.max() < tol and diff_l.max() < tol:
        print("\nVERDICT: Stored Elo ratings are consistent with strict pre-match computation (no look-ahead bias).")
    else:
        print("\nVERDICT: Stored Elo ratings DEVIATE from pre-match computation — look-ahead bias detected.")

    # Save verification artifacts
    out_dir = ROOT / "verification"
    out_dir.mkdir(exist_ok=True)
    merged.to_csv(out_dir / "elo_pre_match_verification.csv", index=False)
    with open(out_dir / "elo_lookahead_verdict.txt", "w") as f:
        f.write(f"matches_compared,{len(merged)}\n")
        f.write(f"max_abs_diff_winner,{diff_w.max():.6f}\n")
        f.write(f"max_abs_diff_loser,{diff_l.max():.6f}\n")
        f.write(f"mean_abs_diff_winner,{diff_w.mean():.6f}\n")
        f.write(f"mean_abs_diff_loser,{diff_l.mean():.6f}\n")
        f.write(f"lookahead_max_abs_diff_winner,{diff_w_b.max():.6f}\n")
        f.write(f"lookahead_max_abs_diff_loser,{diff_l_b.max():.6f}\n")
        verdict = "NO_LOOKAHEAD_BIAS" if (diff_w.max() < tol and diff_l.max() < tol) else "BIAS_DETECTED"
        f.write(f"verdict,{verdict}\n")


if __name__ == "__main__":
    main()
