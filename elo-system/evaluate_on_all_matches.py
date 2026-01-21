"""Evaluate the TennisEloSystem on the same dataset used by the DNN pipeline.

Goal
----
The DNN project trains on `all_matches.csv` after applying a few key filters
(tourney levels, critical columns present, excluding retirements/walkovers).
This script evaluates an ELO-based win probability model on the *same*
(filtered) match set.

It generates *pre-match* win probabilities (before updating ratings for the
match), then computes classification metrics and writes outputs under:

    ./output/

in the `elo-system` directory.

Notes on alignment with the DNN dataset
--------------------------------------
The DNN pipeline augments the dataset by swapping winner/loser to create a
negative class. For ELO evaluation, we generate predictions for both:
- the winner-vs-loser perspective (label=1)
- the loser-vs-winner perspective (label=0)

This keeps the label convention consistent with the DNN classifier.

Usage
-----
From the `elo-system` directory:

    python evaluate_on_all_matches.py --csv ../dnn/all_matches.csv

If your path differs, pass an explicit path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib is optional but recommended for plots.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tennis_elo import EloConfig, TennisEloSystem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Tennis ELO on all_matches.csv with DNN-compatible filtering.")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("../dnn/all_matches.csv"),
        help="Path to all_matches.csv (default: ../dnn/all_matches.csv)",
    )
    p.add_argument(
        "--levels",
        type=str,
        default="A,G,M",
        help="Comma-separated tourney levels to include (default: A,G,M)",
    )
    p.add_argument(
        "--no_decay",
        action="store_true",
        help="Disable inactivity decay for this evaluation run.",
    )
    p.add_argument(
        "--no_mov",
        action="store_true",
        help="Disable margin-of-victory adjustment for this evaluation run.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )
    p.add_argument(
        "--max_matches",
        type=int,
        default=0,
        help="If >0, only process the first N matches (after sorting by date) for quick tests.",
    )
    return p.parse_args()


def _extract_date(date_str: str) -> Optional[datetime]:
    """Parse the 'weird' tourney_date format used in this dataset.

    Example: '1970-01-01 00:00:00.019680708+00:00'
    We follow the same extraction logic as the original elo-system code.
    """
    try:
        frac = str(date_str).split(".")[1].split("+")[0]
        year = int(frac[1:5])
        month = int(frac[5:7])
        day = int(frac[7:9])
        return datetime(year=year, month=month, day=day)
    except Exception:
        return None


def _parse_score_is_valid(score: str) -> bool:
    """Match the DNN data prep's idea of a valid score.

    We exclude retirements/walkovers and rows with empty/garbled scores.
    """
    if not isinstance(score, str) or not score.strip():
        return False
    s = score.lower()
    if "ret" in s or "w/o" in s or "walk" in s:
        return False
    # Require at least one set token like '6-4'
    tokens = score.replace("\u00a0", " ").split()
    return any("-" in t for t in tokens)


def load_filtered_matches(csv_path: Path, levels: Tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # DNN-like filters
    df["tourney_level"] = df["tourney_level"].astype(str).str.upper()
    df = df[df["tourney_level"].isin(levels)].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    cat_cols = ["surface", "round", "tourney_level"]
    critical_cols = cat_cols + ["best_of", "draw_size", "winner_rank", "loser_rank", "score"]
    df = df.dropna(subset=critical_cols)

    # Score validity (exclude RET/W.O.)
    df = df[df["score"].apply(_parse_score_is_valid)].copy()

    # Parse/derive match date used by the elo system
    df["match_date"] = df["tourney_date"].apply(_extract_date)
    df = df.dropna(subset=["match_date"]).copy()

    df = df.sort_values("match_date").reset_index(drop=True)
    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute a compact set of classification metrics."""
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 1e-8, 1 - 1e-8)
    y_pred = (y_prob >= 0.5).astype(int)

    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "log_loss": float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean()),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }

    # Optional sklearn metrics if available
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

        out.update(
            {
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                "pr_auc": float(average_precision_score(y_true, y_prob)),
                "f1": float(f1_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
            }
        )
    except Exception:
        pass

    return out


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    """Write ROC + PR plots if sklearn is available."""
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (ELO)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=160)
        plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (ELO)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=160)
        plt.close()
    except Exception:
        # If sklearn is not present, silently skip.
        return


def main() -> None:
    args = parse_args()
    levels = tuple([s.strip().upper() for s in args.levels.split(",") if s.strip()])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_filtered_matches(args.csv, levels)

    if args.max_matches and args.max_matches > 0:
        df = df.iloc[: args.max_matches].copy()

    # Configure ELO to match existing defaults, with optional toggles.
    config = EloConfig(
        decay_enabled=not args.no_decay,
        mov_enabled=not args.no_mov,
    )

    elo = TennisEloSystem(config)

    rows = []

    # Chronological processing: predict then update.
    for i, r in df.iterrows():
        winner = r["winner_name"]
        loser = r["loser_name"]
        match_date = r["match_date"]
        tourney_level = r["tourney_level"]
        round_name = r["round"]
        surface = r.get("surface", None)
        score = r.get("score", "")

        # Apply decay before prediction (for both players) if enabled.
        elo.apply_decay(winner, match_date)
        elo.apply_decay(loser, match_date)

        elo_w = elo._get_rating(winner)
        elo_l = elo._get_rating(loser)

        p_wins = elo.expected_score(elo_w, elo_l)
        p_loser_wins = elo.expected_score(elo_l, elo_w)

        # Record winner-vs-loser (label=1)
        rows.append(
            {
                "match_id": int(i),
                "match_date": match_date,
                "player_a": winner,
                "player_b": loser,
                "label": 1,
                "elo_a": float(elo_w),
                "elo_b": float(elo_l),
                "elo_diff": float(elo_w - elo_l),
                "p_a_win": float(p_wins),
                "tourney_level": tourney_level,
                "round": round_name,
                "surface": surface,
            }
        )

        # Record swapped perspective (label=0)
        rows.append(
            {
                "match_id": int(i),
                "match_date": match_date,
                "player_a": loser,
                "player_b": winner,
                "label": 0,
                "elo_a": float(elo_l),
                "elo_b": float(elo_w),
                "elo_diff": float(elo_l - elo_w),
                "p_a_win": float(p_loser_wins),
                "tourney_level": tourney_level,
                "round": round_name,
                "surface": surface,
            }
        )

        # Update ELO using the real match outcome.
        elo.process_match(
            winner=winner,
            loser=loser,
            match_date=match_date,
            tourney_level=tourney_level,
            round_name=round_name,
            score=score,
            surface=surface,
        )

    pred_df = pd.DataFrame(rows)

    # Save predictions
    pred_path = out_dir / "elo_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Metrics on the (augmented) dataset
    y_true = pred_df["label"].to_numpy(dtype=int)
    y_prob = pred_df["p_a_win"].to_numpy(dtype=float)
    metrics = compute_metrics(y_true, y_prob)

    # Add config and dataset summary
    summary = {
        "csv": str(args.csv),
        "levels": list(levels),
        "matches_used": int(len(df)),
        "rows_scored": int(len(pred_df)),
        "date_min": df["match_date"].min().strftime("%Y-%m-%d") if len(df) else None,
        "date_max": df["match_date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "elo_config": asdict(config),
        "metrics": metrics,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Simple plots
    plot_roc_pr(y_true, y_prob, out_dir)

    # Probability histogram
    plt.figure()
    plt.hist(y_prob, bins=50)
    plt.xlabel("Predicted P(player_a wins)")
    plt.ylabel("Count")
    plt.title("ELO Predicted Probability Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "prob_hist.png", dpi=160)
    plt.close()

    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics:     {out_dir / 'metrics.json'}")
    print(f"Wrote plots to:    {out_dir}")


if __name__ == "__main__":
    main()
