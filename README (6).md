# TennisPredictor

Code for the paper **"A Unified Benchmark of Machine Learning and Deep Neural Networks for Tennis Match Prediction"**  
Published in *Analytics* (MDPI) — Manuscript ID: analytics-4362848  
Authors: Khem Poudel, Clifford N. Jones, Saroj Baral, Thuan Nhan, Satish Wagle, Jorge Vargas, Lilly-Sophie Schmidt

---

## Overview

This repository contains all experimental code used to compare Elo ratings, classical machine learning, and deep neural networks for predicting professional men's ATP tennis match outcomes. All models are evaluated on a unified 70/15/15 stratified train/validation/test split across 495,374 pre-match observations (1968 to 2024).

Key results:
- **Best model:** ELO-ML Ridge Classifier — 67.39% accuracy
- **Classical ML best:** AdaBoost — 66.40% accuracy  
- **Best DNN:** Medium Residual (9.5M params) — 66.39% accuracy
- **Elo baseline:** 66.36% accuracy
- Model capacity shows strongly diminishing returns: a 207K-parameter DNN matches a 21M-parameter DNN

---

## Data

Raw match data comes from Jeff Sackmann's publicly available ATP tennis database:  
https://github.com/JeffSackmann/tennis_atp

Download the repository and place the combined match CSV at:
```
project/data/raw/all_matches.csv
```

No data files are included in this repository.

---

## Repository Structure

```
TennisPredictor/
├── README.md
├── requirements.txt
│
├── build_unified_dataset.py       # Build the 70/15/15 stratified split
├── verify_elo_lookahead.py        # Verify Elo ratings have no look-ahead bias
│
├── run_elo_baseline_unified.py    # Evaluate Elo baseline on unified test set
├── run_ml_eloml_unified.py        # Train all classical ML and ELO-ML models
├── run_dnn_unified.py             # Train tiny/small/medium DNN configurations
├── run_dnn_large_only.py          # Train large DNN (separate due to compute time)
├── run_shap_importance.py         # Compute SHAP feature importance (Figure 4)
├── run_inference_timing.py        # Measure wall-clock inference times (Table 9)
├── compute_mcnemar.py             # Compute McNemar pairwise significance tests
├── fill_thesis.py                 # Populate result tables and regenerate figures
│
└── results/                       # Output directory (created by scripts)
    ├── elo/
    ├── ml_eloml/
    ├── dnn/
    ├── shap/
    ├── timing/
    └── mcnemar/
```

---

## Setup

Python 3.9 or higher is recommended.

```bash
git clone https://github.com/LillySchmidt/TennisPredictor.git
cd TennisPredictor
pip install -r requirements.txt
```

---

## Reproducing Results

Run the scripts in this order:

### 1. Build the unified dataset
```bash
python build_unified_dataset.py
```
Creates the 70/15/15 stratified split in `unified_data/`. Requires `project/data/raw/all_matches.csv`.

### 2. Verify Elo look-ahead bias (optional but recommended)
```bash
python verify_elo_lookahead.py
```
Confirms stored Elo ratings are strictly pre-match. Expected output: max absolute deviation = 0.000000.

### 3. Run the Elo baseline
```bash
python run_elo_baseline_unified.py
```
Outputs metrics and bootstrap confidence intervals to `results/elo/`.

### 4. Train classical ML and ELO-ML models
```bash
python run_ml_eloml_unified.py
```
Trains all 10 classical ML algorithms and 3 ELO-ML variants. Saves predictions, models, and bootstrap CIs to `results/ml_eloml/`.

### 5. Train DNN configurations
```bash
python run_dnn_unified.py       # tiny, small, medium
python run_dnn_large_only.py    # large (run separately due to training time)
```
Outputs to `results/dnn/`. Requires a CUDA-capable GPU for reasonable runtime; CPU training is supported but slow for the large configuration (~14 hours).

### 6. Compute SHAP feature importance
```bash
python run_shap_importance.py
```
Generates Figure 4 (SHAP bar chart and beeswarm plot) in `results/shap/`.

### 7. Measure inference times
```bash
python run_inference_timing.py
```
Outputs Table 9 data to `results/timing/`.

### 8. Compute McNemar significance tests
```bash
python compute_mcnemar.py
```
Outputs pairwise p-values to `results/mcnemar/`.

---

## Key Findings

| Approach | Best Model | Accuracy | ROC-AUC | Brier |
|---|---|---|---|---|
| Elo Baseline | Elo expected score | 66.36% | 0.7275 | 0.2108 |
| Classical ML | AdaBoost | 66.40% | 0.7243 | 0.2145 |
| Deep Neural Networks | Medium Residual | 66.39% | 0.7258 | 0.2113 |
| ELO-ML Combined | Ridge Classifier | 67.39% | 0.7413 | 0.2183 |

All results are on the held-out test set (74,307 observations). 95% bootstrap confidence intervals are reported in the paper.

---

## Citation

If you use this code, please cite:

```
Poudel, K.; Jones, C.N.; Baral, S.; Nhan, T.; Wagle, S.; Vargas, J.; Schmidt, L.-S.
A Unified Benchmark of Machine Learning and Deep Neural Networks for Tennis Match Prediction.
Analytics 2026, analytics-4362848.
```

---

## License

MIT License. See LICENSE for details.
