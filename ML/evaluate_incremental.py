"""Incremental ML evaluation with accurate RAM and time tracking.

Features:
- Accurate peak RAM monitoring during training
- Precise timing for each phase (fit, score, hyperparameter tuning)
- Saves results after each model completes
- Generates individual plots for each model
- Creates comparison plots at the end
- Comprehensive hyperparameter tuning
- Always uses n_jobs=-1 for maximum performance
- Default 5-fold CV
"""

from __future__ import annotations

import json
import sys
import warnings
import gc
import psutil
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')

if __package__ is None or __package__ == "":
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import data_prep as dp
    import models_optimized as mdl
else:
    from . import data_prep as dp
    from . import models_optimized as mdl

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = FIGURES_DIR / "individual_models"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_REPORTS_DIR = REPORTS_DIR / "individual_models"


class RAMMonitor:
    """Monitor peak RAM usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_mb = 0
        self.running = False
        self.thread = None
        
    def _monitor(self):
        """Internal monitoring loop."""
        while self.running:
            current_mb = self.process.memory_info().rss / 1024 / 1024
            if current_mb > self.peak_mb:
                self.peak_mb = current_mb
            time.sleep(0.1)  # Check every 100ms
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.peak_mb = self.process.memory_info().rss / 1024 / 1024
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return peak."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_mb
    
    def get_current_mb(self):
        """Get current RAM usage."""
        return self.process.memory_info().rss / 1024 / 1024


def setup_matplotlib():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100


def evaluate_model_with_tracking(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Evaluate model with accurate RAM and timing tracking."""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {name}")
    print(f"{'='*80}")
    
    # Initialize RAM monitor
    ram_monitor = RAMMonitor()
    ram_start = ram_monitor.get_current_mb()
    
    # Setup CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    
    print(f"Running {cv_splits}-fold CV with n_jobs=-1 (parallel)...")
    
    # Start monitoring and timing
    ram_monitor.start()
    cv_start_time = time.time()
    
    # Run CV
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,  # ALWAYS parallel
        return_train_score=True,
        error_score="raise",
    )
    
    # Stop monitoring
    cv_end_time = time.time()
    ram_peak = ram_monitor.stop()
    ram_end = ram_monitor.get_current_mb()
    
    training_time = cv_end_time - cv_start_time
    
    # Aggregate results
    results = {
        "model_name": name,
        "cv_folds": cv_splits,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "timing": {
            "total_training_seconds": float(training_time),
            "fit_time_per_fold_mean": float(np.mean(cv_results["fit_time"])),
            "fit_time_per_fold_std": float(np.std(cv_results["fit_time"])),
            "fit_time_per_fold_min": float(np.min(cv_results["fit_time"])),
            "fit_time_per_fold_max": float(np.max(cv_results["fit_time"])),
            "score_time_per_fold_mean": float(np.mean(cv_results["score_time"])),
            "score_time_per_fold_std": float(np.std(cv_results["score_time"])),
            "fit_times_all_folds": [float(t) for t in cv_results["fit_time"]],
            "score_times_all_folds": [float(t) for t in cv_results["score_time"]],
        },
        "memory": {
            "ram_start_mb": float(ram_start),
            "ram_end_mb": float(ram_end),
            "ram_peak_mb": float(ram_peak),
            "ram_used_mb": float(ram_peak - ram_start),
        }
    }
    
    # Extract metrics
    for metric in scoring.keys():
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        
        results["metrics"][metric] = {
            "test_mean": float(np.mean(cv_results[test_key])),
            "test_std": float(np.std(cv_results[test_key])),
            "test_min": float(np.min(cv_results[test_key])),
            "test_max": float(np.max(cv_results[test_key])),
            "test_scores": [float(x) for x in cv_results[test_key]],
            "train_mean": float(np.mean(cv_results[train_key])),
            "train_std": float(np.std(cv_results[train_key])),
            "train_min": float(np.min(cv_results[train_key])),
            "train_max": float(np.max(cv_results[train_key])),
        }
    
    # Print summary
    print(f"\nResults:")
    print(f"  F1 Score:    {results['metrics']['f1']['test_mean']:.4f} ± {results['metrics']['f1']['test_std']:.4f}")
    print(f"  Accuracy:    {results['metrics']['accuracy']['test_mean']:.4f} ± {results['metrics']['accuracy']['test_std']:.4f}")
    print(f"  Precision:   {results['metrics']['precision']['test_mean']:.4f} ± {results['metrics']['precision']['test_std']:.4f}")
    print(f"  Recall:      {results['metrics']['recall']['test_mean']:.4f} ± {results['metrics']['recall']['test_std']:.4f}")
    print(f"  ROC-AUC:     {results['metrics']['roc_auc']['test_mean']:.4f} ± {results['metrics']['roc_auc']['test_std']:.4f}")
    print(f"\nTiming:")
    print(f"  Total Time:  {training_time:.2f} seconds")
    print(f"  Fit Time:    {results['timing']['fit_time_per_fold_mean']:.2f}s per fold")
    print(f"  Score Time:  {results['timing']['score_time_per_fold_mean']:.2f}s per fold")
    print(f"\nMemory:")
    print(f"  Peak RAM:    {ram_peak:.1f} MB")
    print(f"  RAM Used:    {results['memory']['ram_used_mb']:.1f} MB")
    
    return results


def hyperparameter_tuning_with_tracking(
    name: str,
    model,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """Perform grid search with RAM and timing tracking.
    
    Saves ALL parameter combinations and their performance metrics.
    """
    
    if not param_grid:
        print(f"  No hyperparameter grid for {name}")
        return model, {}
    
    print(f"\n  {'='*76}")
    print(f"  HYPERPARAMETER TUNING: {name}")
    print(f"  {'='*76}")
    
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"  Grid size: {n_combinations} combinations")
    print(f"  CV folds: {cv_splits}")
    print(f"  Total fits: {n_combinations * cv_splits}")
    
    # Initialize RAM monitor
    ram_monitor = RAMMonitor()
    ram_start = ram_monitor.get_current_mb()
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Use multiple scoring metrics for grid search
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit="f1",  # Refit on best F1 score
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    
    # Start monitoring
    ram_monitor.start()
    tuning_start = time.time()
    
    grid_search.fit(X, y)
    
    # Stop monitoring
    tuning_end = time.time()
    ram_peak = ram_monitor.stop()
    ram_end = ram_monitor.get_current_mb()
    
    tuning_time = tuning_end - tuning_start
    
    # Extract ALL results for ALL combinations
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Build comprehensive results with all combinations
    all_combinations = []
    for idx in range(len(cv_results_df)):
        combo_result = {
            "rank": int(cv_results_df.loc[idx, "rank_test_f1"]),
            "params": grid_search.cv_results_["params"][idx],
            "params_str": str(grid_search.cv_results_["params"][idx]),
            "metrics": {
                "accuracy": {
                    "test_mean": float(cv_results_df.loc[idx, "mean_test_accuracy"]),
                    "test_std": float(cv_results_df.loc[idx, "std_test_accuracy"]),
                    "train_mean": float(cv_results_df.loc[idx, "mean_train_accuracy"]),
                    "train_std": float(cv_results_df.loc[idx, "std_train_accuracy"]),
                },
                "precision": {
                    "test_mean": float(cv_results_df.loc[idx, "mean_test_precision"]),
                    "test_std": float(cv_results_df.loc[idx, "std_test_precision"]),
                    "train_mean": float(cv_results_df.loc[idx, "mean_train_precision"]),
                    "train_std": float(cv_results_df.loc[idx, "std_train_precision"]),
                },
                "recall": {
                    "test_mean": float(cv_results_df.loc[idx, "mean_test_recall"]),
                    "test_std": float(cv_results_df.loc[idx, "std_test_recall"]),
                    "train_mean": float(cv_results_df.loc[idx, "mean_train_recall"]),
                    "train_std": float(cv_results_df.loc[idx, "std_train_recall"]),
                },
                "f1": {
                    "test_mean": float(cv_results_df.loc[idx, "mean_test_f1"]),
                    "test_std": float(cv_results_df.loc[idx, "std_test_f1"]),
                    "train_mean": float(cv_results_df.loc[idx, "mean_train_f1"]),
                    "train_std": float(cv_results_df.loc[idx, "std_train_f1"]),
                },
                "roc_auc": {
                    "test_mean": float(cv_results_df.loc[idx, "mean_test_roc_auc"]),
                    "test_std": float(cv_results_df.loc[idx, "std_test_roc_auc"]),
                    "train_mean": float(cv_results_df.loc[idx, "mean_train_roc_auc"]),
                    "train_std": float(cv_results_df.loc[idx, "std_train_roc_auc"]),
                },
            },
            "timing": {
                "mean_fit_time": float(cv_results_df.loc[idx, "mean_fit_time"]),
                "std_fit_time": float(cv_results_df.loc[idx, "std_fit_time"]),
                "mean_score_time": float(cv_results_df.loc[idx, "mean_score_time"]),
                "std_score_time": float(cv_results_df.loc[idx, "std_score_time"]),
            }
        }
        all_combinations.append(combo_result)
    
    # Sort by F1 score (best first)
    all_combinations_sorted = sorted(all_combinations, key=lambda x: x["metrics"]["f1"]["test_mean"], reverse=True)
    
    # Extract results
    results = {
        "best_params": grid_search.best_params_,
        "best_f1_score": float(grid_search.best_score_),
        "best_index": int(grid_search.best_index_),
        "n_combinations_tested": int(n_combinations),
        "cv_folds": cv_splits,
        "total_fits": int(n_combinations * cv_splits),
        "tuning_time_seconds": float(tuning_time),
        "tuning_memory": {
            "ram_start_mb": float(ram_start),
            "ram_peak_mb": float(ram_peak),
            "ram_used_mb": float(ram_peak - ram_start),
        },
        "all_combinations": all_combinations_sorted,  # All combos sorted by F1 score
        "top_10_combinations": all_combinations_sorted[:10],  # Top 10 for quick reference
        "summary_statistics": {
            "best_f1": float(max([c["metrics"]["f1"]["test_mean"] for c in all_combinations])),
            "worst_f1": float(min([c["metrics"]["f1"]["test_mean"] for c in all_combinations])),
            "mean_f1": float(np.mean([c["metrics"]["f1"]["test_mean"] for c in all_combinations])),
            "std_f1": float(np.std([c["metrics"]["f1"]["test_mean"] for c in all_combinations])),
            "best_accuracy": float(max([c["metrics"]["accuracy"]["test_mean"] for c in all_combinations])),
            "best_precision": float(max([c["metrics"]["precision"]["test_mean"] for c in all_combinations])),
            "best_recall": float(max([c["metrics"]["recall"]["test_mean"] for c in all_combinations])),
            "best_roc_auc": float(max([c["metrics"]["roc_auc"]["test_mean"] for c in all_combinations])),
        }
    }
    
    print(f"\n  Results:")
    print(f"    Best F1 Score: {results['best_f1_score']:.4f}")
    print(f"    Best Params: {results['best_params']}")
    print(f"    Combinations Tested: {results['n_combinations_tested']}")
    print(f"    F1 Score Range: {results['summary_statistics']['worst_f1']:.4f} - {results['summary_statistics']['best_f1']:.4f}")
    print(f"    Mean F1 Score: {results['summary_statistics']['mean_f1']:.4f} ± {results['summary_statistics']['std_f1']:.4f}")
    print(f"    Tuning Time: {tuning_time:.2f}s")
    print(f"    Peak RAM: {ram_peak:.1f} MB")
    
    # Print top 3 combinations
    print(f"\n  Top 3 Parameter Combinations:")
    for i, combo in enumerate(all_combinations_sorted[:3], 1):
        print(f"    #{i} - F1: {combo['metrics']['f1']['test_mean']:.4f}, Params: {combo['params_str']}")
    
    return grid_search.best_estimator_, results


def plot_individual_model(
    name: str,
    results: Dict[str, Any],
    tuning_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive plots for a single model including hyperparameter tuning."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if we have tuning results
    has_tuning = tuning_results and "all_combinations" in tuning_results
    
    # Create figure with subplots - more if we have tuning data
    if has_tuning:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    model_title = name.replace('_', ' ').title()
    fig.suptitle(f'{model_title} - Comprehensive Analysis', fontsize=18, fontweight='bold')
    
    # 1. Metrics Bar Chart (larger)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    means = [results['metrics'][m]['test_mean'] for m in metrics]
    stds = [results['metrics'][m]['test_std'] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=2)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics (Test Set with CV Std Dev)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Cross-Validation Scores Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    f1_scores = results['metrics']['f1']['test_scores']
    folds = list(range(1, len(f1_scores) + 1))
    ax2.plot(folds, f1_scores, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax2.axhline(y=results['metrics']['f1']['test_mean'], color='r', linestyle='--', linewidth=2,
                label=f"Mean: {results['metrics']['f1']['test_mean']:.3f}")
    ax2.fill_between(folds, 
                     [results['metrics']['f1']['test_mean'] - results['metrics']['f1']['test_std']] * len(folds),
                     [results['metrics']['f1']['test_mean'] + results['metrics']['f1']['test_std']] * len(folds),
                     alpha=0.2, color='red')
    ax2.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax2.set_title('F1 Score per CV Fold', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks(folds)
    
    # 3. Train vs Test Performance
    ax3 = fig.add_subplot(gs[1, 0])
    train_means = [results['metrics'][m]['train_mean'] for m in metrics]
    test_means = [results['metrics'][m]['test_mean'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax3.bar(x - width/2, train_means, width, label='Train', alpha=0.8, color='#90EE90', edgecolor='darkgreen')
    bars2 = ax3.bar(x + width/2, test_means, width, label='Test', alpha=0.8, color='#FFB6C1', edgecolor='darkred')
    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title('Train vs Test Performance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m[:3].upper() for m in metrics])
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1.05])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Detailed Timing Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    timing_data = [
        results['timing']['fit_time_per_fold_mean'],
        results['timing']['score_time_per_fold_mean'],
        results['timing']['total_training_seconds'] / results['cv_folds']
    ]
    timing_labels = ['Fit\nTime', 'Score\nTime', 'Total\nper Fold']
    colors = ['#FF9999', '#66B3FF', '#99FF99']
    
    bars = ax4.bar(timing_labels, timing_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Timing Breakdown', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, timing_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Memory Usage
    ax5 = fig.add_subplot(gs[1, 2])
    mem_data = [
        results['memory']['ram_start_mb'],
        results['memory']['ram_end_mb'],
        results['memory']['ram_peak_mb']
    ]
    mem_labels = ['Start', 'End', 'Peak']
    colors_mem = ['#90EE90', '#FFD700', '#FF6347']
    
    bars = ax5.bar(mem_labels, mem_data, color=colors_mem, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('RAM (MB)', fontsize=11, fontweight='bold')
    ax5.set_title('Memory Usage', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, mem_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Performance Distribution Box Plot
    ax6 = fig.add_subplot(gs[2, :2])
    metric_scores = [results['metrics'][m]['test_scores'] for m in metrics]
    bp = ax6.boxplot(metric_scores, labels=[m.replace('_', ' ').title() for m in metrics],
                     patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#FFB6C1', '#B6D7A8', '#A4C2F4', '#FFD966', '#EA9999']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='--')
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    for median in bp['medians']:
        median.set(linewidth=2, color='darkred')
    
    ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax6.set_title('Score Distribution Across CV Folds', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    ax6.set_ylim([0, 1.05])
    
    # 7. Summary Statistics Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['Mean F1', f"{results['metrics']['f1']['test_mean']:.4f} ± {results['metrics']['f1']['test_std']:.4f}"],
        ['Mean Accuracy', f"{results['metrics']['accuracy']['test_mean']:.4f}"],
        ['Mean Precision', f"{results['metrics']['precision']['test_mean']:.4f}"],
        ['Mean Recall', f"{results['metrics']['recall']['test_mean']:.4f}"],
        ['Mean ROC-AUC', f"{results['metrics']['roc_auc']['test_mean']:.4f}"],
        ['', ''],
        ['Total Time', f"{results['timing']['total_training_seconds']:.1f}s"],
        ['Fit Time/Fold', f"{results['timing']['fit_time_per_fold_mean']:.2f}s"],
        ['RAM Peak', f"{results['memory']['ram_peak_mb']:.0f} MB"],
        ['RAM Used', f"{results['memory']['ram_used_mb']:.0f} MB"],
        ['CV Folds', f"{results['cv_folds']}"],
    ]
    
    # Add tuning info if available
    if has_tuning:
        summary_data.extend([
            ['', ''],
            ['Tuning Status', 'Completed'],
            ['Combos Tested', f"{tuning_results['n_combinations_tested']}"],
            ['Best F1 Tuned', f"{tuning_results['best_f1_score']:.4f}"],
        ])
    
    table = ax7.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', size=10)
    
    # Alternate row colors
    for i in range(2, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    # 8 & 9. Hyperparameter Tuning Plots (if available)
    if has_tuning:
        # 8. All Combinations F1 Scores
        ax8 = fig.add_subplot(gs[3, :2])
        all_combos = tuning_results['all_combinations']
        f1_scores_all = [c['metrics']['f1']['test_mean'] for c in all_combos]
        
        x_combos = list(range(len(f1_scores_all)))
        ax8.scatter(x_combos, f1_scores_all, alpha=0.6, s=50, c=f1_scores_all, cmap='RdYlGn', 
                   edgecolors='black', linewidth=0.5)
        ax8.axhline(y=tuning_results['best_f1_score'], color='red', linestyle='--', linewidth=2,
                   label=f"Best: {tuning_results['best_f1_score']:.4f}")
        ax8.axhline(y=tuning_results['summary_statistics']['mean_f1'], color='blue', linestyle='--', 
                   linewidth=2, alpha=0.7,
                   label=f"Mean: {tuning_results['summary_statistics']['mean_f1']:.4f}")
        
        ax8.set_xlabel('Parameter Combination Index (sorted by F1)', fontsize=11, fontweight='bold')
        ax8.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax8.set_title(f'All {len(f1_scores_all)} Hyperparameter Combinations', fontsize=12, fontweight='bold')
        ax8.legend(loc='best')
        ax8.grid(alpha=0.3, linestyle='--')
        
        # 9. Top 10 Combinations Details
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        top_10 = tuning_results['top_10_combinations'][:5]  # Show top 5
        top_data = [['Rank', 'F1 Score', 'Params']]
        
        for i, combo in enumerate(top_10, 1):
            # Shorten params string if too long
            params_str = str(combo['params'])
            if len(params_str) > 40:
                params_str = params_str[:37] + '...'
            top_data.append([
                f'#{i}',
                f"{combo['metrics']['f1']['test_mean']:.4f}",
                params_str
            ])
        
        table_top = ax9.table(cellText=top_data, cellLoc='left', loc='center',
                             colWidths=[0.1, 0.15, 0.75])
        table_top.auto_set_font_size(False)
        table_top.set_fontsize(8)
        table_top.scale(1, 2)
        
        # Style header
        for i in range(3):
            table_top[(0, i)].set_facecolor('#FF6347')
            table_top[(0, i)].set_text_props(weight='bold', color='white', size=9)
        
        ax9.set_title('Top 5 Parameter Combinations', fontsize=11, fontweight='bold', pad=10)
    
    # Save figure
    filename = output_dir / f"{name}_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {filename.name}")


def save_model_results(
    name: str,
    results: Dict[str, Any],
    tuning_results: Dict[str, Any],
    reports_dir: Path
) -> None:
    """Save individual model results to JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    combined_results = {
        "cross_validation": results,
        "hyperparameter_tuning": tuning_results,
    }
    
    filename = reports_dir / f"{name}_results.json"
    with open(filename, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"  ✓ Saved JSON: {filename.name}")


def create_comparison_plots(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """Create comprehensive comparison plots for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*80}\n")
    
    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # 1. Metrics Comparison - Enhanced
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle('Model Comparison - All Metrics', fontsize=18, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = [all_results[m]['metrics'][metric]['test_mean'] for m in models]
        stds = [all_results[m]['metrics'][metric]['test_std'] for m in models]
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.75, linewidth=2)
        
        # Color code by performance
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(means)))
        sorted_indices = np.argsort(means)
        for i, bar in enumerate(bars):
            color_idx = np.where(sorted_indices == i)[0][0]
            bar.set_color(colors[color_idx])
            bar.set_edgecolor('black')
            
            # Add value labels
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{means[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} by Model', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
    
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(output_dir / "01_metrics_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 01_metrics_comparison.png")
    
    # 2. Training Time Comparison - Enhanced
    fig, ax = plt.subplots(figsize=(14, 7))
    times = [all_results[m]['timing']['total_training_seconds'] for m in models]
    x_pos = np.arange(len(models))
    
    bars = ax.bar(x_pos, times, alpha=0.8, linewidth=2, edgecolor='black')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(times)))
    sorted_indices = np.argsort(times)
    for idx, bar in enumerate(bars):
        color_idx = np.where(sorted_indices == idx)[0][0]
        bar.set_color(colors[color_idx])
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Total Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_training_times.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02_training_times.png")
    
    # 3. Memory Usage Comparison - Enhanced
    fig, ax = plt.subplots(figsize=(14, 7))
    ram_used = [all_results[m]['memory']['ram_used_mb'] for m in models]
    ram_peak = [all_results[m]['memory']['ram_peak_mb'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ram_used, width, label='RAM Used', alpha=0.8, 
                   color='skyblue', edgecolor='darkblue', linewidth=2)
    bars2 = ax.bar(x + width/2, ram_peak, width, label='RAM Peak', alpha=0.8, 
                   color='orange', edgecolor='darkorange', linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_memory_usage.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 03_memory_usage.png")
    
    # 4. Performance Radar Chart
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for model in models:
        values = [all_results[model]['metrics'][m]['test_mean'] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model.replace('_', ' ').title(), markersize=8)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_performance_radar.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_performance_radar.png")
    
    # 5. Performance Heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    
    data = []
    for model in models:
        row = [all_results[model]['metrics'][m]['test_mean'] for m in metrics]
        data.append(row)
    
    df = pd.DataFrame(data,
                     index=[m.replace('_', ' ').title() for m in models],
                     columns=[m.replace('_', '\n').title() for m in metrics])
    
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Performance Heatmap - All Models', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_performance_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 05_performance_heatmap.png")
    
    # 6. Efficiency Plot (Performance vs Time)
    fig, ax = plt.subplots(figsize=(14, 9))
    
    f1_scores = [all_results[m]['metrics']['f1']['test_mean'] for m in models]
    times = [all_results[m]['timing']['total_training_seconds'] for m in models]
    ram = [all_results[m]['memory']['ram_used_mb'] for m in models]
    
    # Normalize RAM for bubble size
    ram_normalized = [(r / max(ram)) * 1000 + 100 for r in ram]
    
    scatter = ax.scatter(times, f1_scores, s=ram_normalized, alpha=0.6, 
                        c=range(len(models)), cmap='viridis', edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax.annotate(model.replace('_', ' ').title(), 
                   (times[i], f1_scores[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Efficiency: Performance vs Training Time\n(Bubble size = RAM usage)', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Model Index')
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_efficiency_plot.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 06_efficiency_plot.png")


def save_final_report(
    all_results: Dict[str, Dict[str, Any]],
    all_tuning_results: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path
) -> None:
    """Save final comprehensive report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    best_f1 = max(all_results.items(), key=lambda x: x[1]['metrics']['f1']['test_mean'])
    best_acc = max(all_results.items(), key=lambda x: x[1]['metrics']['accuracy']['test_mean'])
    best_auc = max(all_results.items(), key=lambda x: x[1]['metrics']['roc_auc']['test_mean'])
    fastest = min(all_results.items(), key=lambda x: x[1]['timing']['total_training_seconds'])
    most_memory = max(all_results.items(), key=lambda x: x[1]['memory']['ram_peak_mb'])
    least_memory = min(all_results.items(), key=lambda x: x[1]['memory']['ram_peak_mb'])
    
    report = {
        "metadata": metadata,
        "all_model_results": all_results,
        "all_tuning_results": all_tuning_results,
        "summary": {
            "best_model_by_f1": {
                "model": best_f1[0],
                "f1_score": best_f1[1]['metrics']['f1']['test_mean'],
                "f1_std": best_f1[1]['metrics']['f1']['test_std'],
            },
            "best_model_by_accuracy": {
                "model": best_acc[0],
                "accuracy": best_acc[1]['metrics']['accuracy']['test_mean'],
            },
            "best_model_by_roc_auc": {
                "model": best_auc[0],
                "roc_auc": best_auc[1]['metrics']['roc_auc']['test_mean'],
            },
            "fastest_model": {
                "model": fastest[0],
                "time_seconds": fastest[1]['timing']['total_training_seconds'],
            },
            "most_memory_efficient": {
                "model": least_memory[0],
                "ram_peak_mb": least_memory[1]['memory']['ram_peak_mb'],
            },
            "most_memory_intensive": {
                "model": most_memory[0],
                "ram_peak_mb": most_memory[1]['memory']['ram_peak_mb'],
            },
        }
    }
    
    filename = output_dir / "final_evaluation_report.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved final report: {filename}")


def run_incremental_evaluation(
    data_path: Path = None,
    cv_splits: int = 5,
    tune_hyperparameters: bool = True,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Run incremental evaluation with per-model saving."""
    
    setup_matplotlib()
    
    print(f"\n{'='*80}")
    print("INCREMENTAL ML EVALUATION WITH ACCURATE RAM & TIME TRACKING")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    if data_path is None:
        data_path = ROOT / "all_matches.csv"
    
    X, y = dp.load_features_and_target(data_path, clean=True)
    print(f"  Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    
    # Build models
    print("\nBuilding models...")
    models = mdl.make_models(X)
    param_grids = mdl.get_hyperparameter_grids() if tune_hyperparameters else {}
    print(f"  Created {len(models)} models (all use n_jobs=-1)")
    print(f"  Models: {', '.join(models.keys())}\n")
    
    # Evaluate each model incrementally
    all_results = {}
    all_tuning_results = {}
    
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"\n{'#'*80}")
        print(f"# [{i}/{len(models)}] {name.upper()}")
        print(f"{'#'*80}")
        
        try:
            # Cross-validation evaluation with tracking
            cv_results = evaluate_model_with_tracking(name, model, X, y, cv_splits, random_state)
            all_results[name] = cv_results
            
            # Hyperparameter tuning if enabled
            tuning_results = {}
            if tune_hyperparameters and name in param_grids and param_grids[name]:
                best_model, tuning_results = hyperparameter_tuning_with_tracking(
                    name, model, param_grids[name], X, y, cv_splits=3, random_state=random_state
                )
                all_tuning_results[name] = tuning_results
            else:
                all_tuning_results[name] = {}
            
            # Save individual results immediately
            save_model_results(name, cv_results, tuning_results, MODELS_REPORTS_DIR)
            
            # Generate individual plot with tuning results
            plot_individual_model(name, cv_results, tuning_results, MODELS_DIR)
            
            # Force garbage collection
            gc.collect()
            
            print(f"\n✓ Completed {name} successfully\n")
            
        except Exception as e:
            print(f"\n✗ ERROR with {name}: {str(e)}")
            print(f"  Skipping this model and continuing...")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison plots
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"✓ All {len(all_results)} models completed successfully!")
        print(f"{'='*80}")
        create_comparison_plots(all_results, FIGURES_DIR)
    
    # Save final report
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(X),
        "n_features": len(X.columns),
        "n_models_evaluated": len(all_results),
        "cv_folds": cv_splits,
        "hyperparameter_tuning_enabled": tune_hyperparameters,
        "n_jobs": -1,
        "random_state": random_state,
    }
    
    save_final_report(all_results, all_tuning_results, metadata, REPORTS_DIR)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        best_f1 = max(all_results.items(), key=lambda x: x[1]['metrics']['f1']['test_mean'])
        print(f"\n🏆 Best Model (F1): {best_f1[0]}")
        print(f"   F1 Score:  {best_f1[1]['metrics']['f1']['test_mean']:.4f} ± {best_f1[1]['metrics']['f1']['test_std']:.4f}")
        print(f"   Accuracy:  {best_f1[1]['metrics']['accuracy']['test_mean']:.4f}")
        print(f"   Precision: {best_f1[1]['metrics']['precision']['test_mean']:.4f}")
        print(f"   Recall:    {best_f1[1]['metrics']['recall']['test_mean']:.4f}")
        print(f"   ROC-AUC:   {best_f1[1]['metrics']['roc_auc']['test_mean']:.4f}")
        print(f"   Time:      {best_f1[1]['timing']['total_training_seconds']:.1f}s")
        print(f"   RAM Peak:  {best_f1[1]['memory']['ram_peak_mb']:.0f} MB")
        
        if tune_hyperparameters and best_f1[0] in all_tuning_results and all_tuning_results[best_f1[0]]:
            print(f"\n   Best Hyperparameters:")
            for param, value in all_tuning_results[best_f1[0]]['best_params'].items():
                print(f"     {param}: {value}")
    
    print(f"\n📁 Output Locations:")
    print(f"   Individual model reports: {MODELS_REPORTS_DIR}")
    print(f"   Individual model plots:   {MODELS_DIR}")
    print(f"   Comparison plots:         {FIGURES_DIR}")
    print(f"   Final report:             {REPORTS_DIR}/final_evaluation_report.json")
    
    return all_results, all_tuning_results


if __name__ == "__main__":
    run_incremental_evaluation(
        cv_splits=5,
        tune_hyperparameters=True,
        random_state=42
    )
