# ELO-ML: Elo-Enhanced ML Baselines

This model family combines the **best-performing classic ML models** with **Elo ratings computed at match time**.  
It uses the **same raw dataset** as the other models (`all_matches.csv`), and adds **Elo features** as extra inputs.

## What is new vs. ML?
- **Same base prematch features** as `ML/data_prep.py` (surface, round, tourney level, ranks, etc.).
- **Additional Elo features**:
  - `elo_a`: Elo for player A **before** the match
  - `elo_b`: Elo for player B **before** the match
  - `elo_diff`: `elo_a - elo_b`

## Data Consistency (Exact Dataset)
The dataset is built from the same `all_matches.csv` file and uses the **same cleaning rules** as `ML/data_prep.clean_raw_matches`.  
The only extra requirement is a valid match date for Elo ordering (using the dataset’s custom date encoding).

If a player name is missing, **player IDs are used as Elo keys** so that rows are not dropped.

## How Elo is computed
Elo is computed **chronologically**. For each match:
1. Apply decay (if enabled).
2. Record winner and loser Elo *before* updating.
3. Update Elo with the match result.

This mirrors the logic in `elo-system/evaluate_on_all_matches.py`.

## Best ML Models
The training script automatically selects the **top N models** by metric (default: accuracy) from:
`ML/outputs/reports/final_evaluation_report.json`.

If that report is missing, it falls back to:
`logistic_regression`, `gradient_boosting`, `hist_gradient_boosting`.

## Usage

### 1) Build the ELO-ML dataset
```bash
python data_prep.py \
  --raw ../project/data/raw/all_matches.csv \
  --out data/elo_ml_dataset.csv
```

Optional flags:
- `--no-decay` disable Elo inactivity decay
- `--no-mov` disable margin-of-victory adjustment
- `--no-clean` skip cleaning step

### 2) Train best models on ELO-ML features
```bash
python train_models.py \
  --data data/elo_ml_dataset.csv \
  --top-n 3 \
  --metric accuracy
```

Outputs:
- `outputs/reports/*.json` per-model metrics
- `outputs/summary.csv` leaderboard
- `outputs/models/*.joblib` if `--save-models` is used

## Feature List
Base ML features (from `ML/data_prep.py`):
- `surface`, `round`, `tourney_level`
- `best_of`, `draw_size`, `year`
- `playerA_rank`, `playerB_rank`
- `rank_diff`, `log_playerA_rank`, `log_playerB_rank`, `log_rank_diff`
- `round_code`

ELO-ML additional features:
- `elo_a`, `elo_b`, `elo_diff`

## Notes
- This is a **fair comparison** with DNN/ML because it uses the same raw dataset
  and the same prematch feature definitions, with Elo added as extra inputs.
- Elo is computed only from **past matches**, so no future leakage is introduced.
