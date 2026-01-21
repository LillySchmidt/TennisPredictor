"""Comprehensive ML evaluation with cross-validation, hyperparameter tuning, and visualization.

This module provides:
- K-fold cross-validation for all models
- Grid search for hyperparameter optimization
- Detailed performance metrics and comparisons
- Professional visualizations and reports
- JSON output for all results
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_validate, 
    GridSearchCV,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings('ignore')

# Support both package and script imports
if __package__ is None or __package__ == "":
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import data_prep as dp
    import models_enhanced as mdl
    import models_lightweight as mdl_lite
    import models_ultra_lightweight as mdl_ultra
    import data_augmentation as aug
else:
    from . import data_prep as dp
    from . import models_enhanced as mdl
    from . import models_lightweight as mdl_lite
    from . import models_ultra_lightweight as mdl_ultra
    from . import data_augmentation as aug

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"


def setup_matplotlib():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100


def evaluate_model_cv(
    name: str, 
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """Perform comprehensive cross-validation evaluation.
    
    Args:
        name: Model name
        model: Fitted model pipeline
        X: Feature DataFrame
        y: Target Series
        cv_splits: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary with detailed metrics
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    
    print(f"  Running {cv_splits}-fold CV for {name}...")
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    
    # Aggregate metrics
    results = {
        "model_name": name,
        "cv_folds": cv_splits,
        "metrics": {}
    }
    
    for metric in scoring.keys():
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        
        results["metrics"][metric] = {
            "test_mean": float(np.mean(cv_results[test_key])),
            "test_std": float(np.std(cv_results[test_key])),
            "test_scores": [float(x) for x in cv_results[test_key]],
            "train_mean": float(np.mean(cv_results[train_key])),
            "train_std": float(np.std(cv_results[train_key])),
            "train_scores": [float(x) for x in cv_results[train_key]],
        }
    
    # Compute fit times
    results["fit_time"] = {
        "mean": float(np.mean(cv_results["fit_time"])),
        "std": float(np.std(cv_results["fit_time"])),
        "total": float(np.sum(cv_results["fit_time"])),
    }
    
    results["score_time"] = {
        "mean": float(np.mean(cv_results["score_time"])),
        "std": float(np.std(cv_results["score_time"])),
    }
    
    return results


def hyperparameter_tuning(
    name: str,
    model,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """Perform grid search for hyperparameter tuning.
    
    Args:
        name: Model name
        model: Model pipeline
        param_grid: Hyperparameter grid
        X: Feature DataFrame
        y: Target Series
        cv_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (best_model, results_dict)
    """
    if not param_grid:
        print(f"  No hyperparameter grid for {name}, using default parameters")
        return model, {}
    
    print(f"  Tuning hyperparameters for {name}...")
    print(f"    Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    
    grid_search.fit(X, y)
    
    # Extract results
    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_scores": [float(x) for x in grid_search.cv_results_["mean_test_score"]],
            "std_test_scores": [float(x) for x in grid_search.cv_results_["std_test_score"]],
            "mean_train_scores": [float(x) for x in grid_search.cv_results_["mean_train_score"]],
            "params": [str(p) for p in grid_search.cv_results_["params"]],
        }
    }
    
    print(f"    Best F1 score: {results['best_score']:.4f}")
    print(f"    Best params: {results['best_params']}")
    
    return grid_search.best_estimator_, results


def plot_cv_metrics(
    cv_results: Dict[str, Dict[str, Any]], 
    output_dir: Path
) -> None:
    """Create comparison plots for cross-validation metrics.
    
    Args:
        cv_results: Dictionary of CV results for each model
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    models = list(cv_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Metrics comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        means = [cv_results[m]["metrics"][metric]["test_mean"] for m in models]
        stds = [cv_results[m]["metrics"][metric]["test_std"] for m in models]
        
        x_pos = np.arange(len(models))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Model')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(output_dir / "cv_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved CV metrics comparison to {output_dir / 'cv_metrics_comparison.png'}")


def plot_training_times(
    cv_results: Dict[str, Dict[str, Any]], 
    output_dir: Path
) -> None:
    """Plot training time comparison.
    
    Args:
        cv_results: Dictionary of CV results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(cv_results.keys())
    times = [cv_results[m]["fit_time"]["mean"] for m in models]
    stds = [cv_results[m]["fit_time"]["std"] for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(models))
    
    bars = ax.bar(x_pos, times, yerr=stds, capsize=5, alpha=0.7)
    
    # Color bars by time (gradient)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(times)))
    sorted_indices = np.argsort(times)
    for idx, bar in enumerate(bars):
        color_idx = np.where(sorted_indices == idx)[0][0]
        bar.set_color(colors[color_idx])
    
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Model Training Time Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_times.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved training times to {output_dir / 'training_times.png'}")


def plot_hyperparameter_comparison(
    hp_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """Plot hyperparameter tuning results.
    
    Args:
        hp_results: Hyperparameter tuning results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter models with tuning results
    models_with_tuning = {k: v for k, v in hp_results.items() if v}
    
    if not models_with_tuning:
        print("  No hyperparameter tuning results to plot")
        return
    
    # Create subplots for each model
    n_models = len(models_with_tuning)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(models_with_tuning.items()):
        ax = axes[idx]
        
        cv_res = results.get("cv_results", {})
        mean_scores = cv_res.get("mean_test_scores", [])
        
        if mean_scores:
            x = np.arange(len(mean_scores))
            ax.plot(x, mean_scores, 'o-', alpha=0.7)
            ax.axhline(y=results.get("best_score", 0), color='r', linestyle='--', 
                      label=f'Best: {results.get("best_score", 0):.4f}')
            ax.set_xlabel('Parameter Combination')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'{model_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_tuning.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved hyperparameter tuning plot to {output_dir / 'hyperparameter_tuning.png'}")


def plot_model_performance_radar(
    cv_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """Create radar chart comparing model performance across metrics.
    
    Args:
        cv_results: CV results dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(cv_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    for model in models:
        values = [cv_results[model]["metrics"][m]["test_mean"] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model.replace('_', ' ').title())
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison (All Metrics)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved radar chart to {output_dir / 'performance_radar.png'}")


def create_performance_heatmap(
    cv_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """Create heatmap of model performance metrics.
    
    Args:
        cv_results: CV results dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(cv_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Create matrix
    data = []
    for model in models:
        row = [cv_results[model]["metrics"][m]["test_mean"] for m in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, 
                     index=[m.replace('_', ' ').title() for m in models],
                     columns=[m.replace('_', ' ').title() for m in metrics])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Score'}, ax=ax)
    ax.set_title('Model Performance Heatmap', size=14, pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved performance heatmap to {output_dir / 'performance_heatmap.png'}")


def save_json_report(
    cv_results: Dict[str, Dict[str, Any]],
    hp_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save comprehensive JSON report.
    
    Args:
        cv_results: Cross-validation results
        hp_results: Hyperparameter tuning results
        output_dir: Output directory
        metadata: Additional metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "metadata": metadata,
        "cross_validation_results": cv_results,
        "hyperparameter_tuning_results": hp_results,
        "summary": {
            "best_model_by_f1": max(cv_results.items(), 
                                   key=lambda x: x[1]["metrics"]["f1"]["test_mean"])[0],
            "best_model_by_accuracy": max(cv_results.items(), 
                                         key=lambda x: x[1]["metrics"]["accuracy"]["test_mean"])[0],
            "best_model_by_roc_auc": max(cv_results.items(), 
                                        key=lambda x: x[1]["metrics"]["roc_auc"]["test_mean"])[0],
            "fastest_model": min(cv_results.items(), 
                               key=lambda x: x[1]["fit_time"]["mean"])[0],
        }
    }
    
    output_file = output_dir / "evaluation_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  Saved JSON report to {output_file}")


def run_comprehensive_evaluation(
    data_path: Path = None,
    augment_data: bool = True,
    augmentation_factor: float = 2.5,
    cv_splits: int = 10,
    tune_hyperparameters: bool = True,
    random_state: int = 42,
    lightweight_mode: bool = False,
    ultra_lightweight_mode: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run comprehensive ML evaluation pipeline.
    
    Args:
        data_path: Path to raw data
        augment_data: Whether to augment the dataset
        augmentation_factor: Data augmentation multiplier
        cv_splits: Number of CV folds
        tune_hyperparameters: Whether to perform hyperparameter tuning
        random_state: Random seed
        lightweight_mode: Use memory-efficient models for large datasets
        ultra_lightweight_mode: Use minimal memory models (RF=25 estimators, n_jobs=1)
        
    Returns:
        Tuple of (cv_results, hp_results)
    """
    setup_matplotlib()
    
    print("="*80)
    print("COMPREHENSIVE ML EVALUATION PIPELINE")
    print("="*80)
    
    # Load and prepare data
    print("\n[1/6] Loading and preparing data...")
    if data_path is None:
        data_path = ROOT / "all_matches.csv"
    
    X, y = dp.load_features_and_target(data_path, clean=True)
    print(f"  Original dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    
    # Augment data if requested
    if augment_data:
        print(f"\n[2/6] Augmenting dataset (factor={augmentation_factor})...")
        X, y = aug.augment_dataset(X, y, augmentation_factor=augmentation_factor, 
                                   random_state=random_state)
        print(f"  Augmented dataset: {len(X)} samples")
        print(f"  Class distribution: {y.value_counts().to_dict()}")
    else:
        print("\n[2/6] Skipping data augmentation")
    
    # Build models
    print("\n[3/6] Building model pipelines...")
    if ultra_lightweight_mode:
        print("  Using ULTRA-LIGHTWEIGHT models (minimal memory: RF=25 estimators, n_jobs=1)")
        models = mdl_ultra.make_ultra_lightweight_models(X)
        param_grids = mdl_ultra.get_ultra_lightweight_hyperparameter_grids()
    elif lightweight_mode:
        print("  Using LIGHTWEIGHT models (memory-efficient for large datasets)")
        models = mdl_lite.make_lightweight_models(X)
        param_grids = mdl_lite.get_lightweight_hyperparameter_grids()
    else:
        models = mdl.make_models(X)
        param_grids = mdl.get_hyperparameter_grids()
    print(f"  Created {len(models)} models: {', '.join(models.keys())}")
    
    # Cross-validation evaluation
    print(f"\n[4/6] Running {cv_splits}-fold cross-validation...")
    if ultra_lightweight_mode:
        print("  Using sequential processing (n_jobs=1) to minimize memory usage")
    cv_results = {}
    for name, model in models.items():
        cv_results[name] = evaluate_model_cv(
            name, model, X, y, cv_splits, random_state, 
            n_jobs=1 if ultra_lightweight_mode else -1
        )
        print(f"  {name}: F1={cv_results[name]['metrics']['f1']['test_mean']:.4f} "
              f"(±{cv_results[name]['metrics']['f1']['test_std']:.4f})")
    
    # Hyperparameter tuning
    hp_results = {}
    if tune_hyperparameters:
        print(f"\n[5/6] Hyperparameter tuning...")
        
        for name, model in models.items():
            if name in ["stacking_ensemble"]:
                print(f"  Skipping {name} (ensemble model)")
                hp_results[name] = {}
                continue
            
            param_grid = param_grids.get(name, {})
            if param_grid:
                _, hp_results[name] = hyperparameter_tuning(
                    name, model, param_grid, X, y, cv_splits=5, random_state=random_state
                )
            else:
                hp_results[name] = {}
    else:
        print("\n[5/6] Skipping hyperparameter tuning")
        hp_results = {name: {} for name in models.keys()}
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    plot_cv_metrics(cv_results, FIGURES_DIR)
    plot_training_times(cv_results, FIGURES_DIR)
    plot_hyperparameter_comparison(hp_results, FIGURES_DIR)
    plot_model_performance_radar(cv_results, FIGURES_DIR)
    create_performance_heatmap(cv_results, FIGURES_DIR)
    
    # Save comprehensive report
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(X),
        "n_features": len(X.columns),
        "n_models": len(models),
        "cv_folds": cv_splits,
        "data_augmented": augment_data,
        "augmentation_factor": augmentation_factor if augment_data else 1.0,
        "hyperparameter_tuning": tune_hyperparameters,
        "lightweight_mode": lightweight_mode,
        "ultra_lightweight_mode": ultra_lightweight_mode,
        "random_state": random_state,
    }
    
    save_json_report(cv_results, hp_results, REPORTS_DIR, metadata)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    best_f1 = max(cv_results.items(), key=lambda x: x[1]["metrics"]["f1"]["test_mean"])
    print(f"Best model (F1): {best_f1[0]}")
    print(f"  F1 Score: {best_f1[1]['metrics']['f1']['test_mean']:.4f} ± {best_f1[1]['metrics']['f1']['test_std']:.4f}")
    print(f"  Accuracy: {best_f1[1]['metrics']['accuracy']['test_mean']:.4f}")
    print(f"  ROC-AUC: {best_f1[1]['metrics']['roc_auc']['test_mean']:.4f}")
    
    return cv_results, hp_results


if __name__ == "__main__":
    run_comprehensive_evaluation(
        augment_data=True,
        augmentation_factor=1,
        cv_splits=10,
        tune_hyperparameters=True,
        random_state=42
    )
