"""Runner for incremental ML evaluation with comprehensive features.

Usage:
    python run_incremental.py

Features:
- 5-fold CV (faster than 10-fold)
- n_jobs=-1 for all models (maximum parallel performance)
- NO stacking ensemble (removed - too slow)
- NO KNN (removed - too slow even with optimization)
- Accurate RAM tracking with continuous monitoring
- Precise timing for all phases
- Hyperparameter tuning with comprehensive grids (ENABLED by default)
- Saves results after each model completes
- Tracks peak RAM usage per model
- Generates individual plots per model
- Creates comparison plots at the end

10 fast, efficient models evaluated.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate_incremental import run_incremental_evaluation


def main():
    """Run incremental evaluation with all features."""
    
    print("="*80)
    print("TENNIS MATCH PREDICTION - INCREMENTAL ML EVALUATION")
    print("="*80)
    print("\nConfiguration:")
    print("  CV folds: 5 (optimal speed/accuracy trade-off)")
    print("  Parallel processing: n_jobs=-1 (all CPU cores)")
    print("  Models: 10 (NO stacking ensemble, NO KNN)")
    print("  Hyperparameter tuning: ENABLED (comprehensive grids)")
    print("  Incremental saving: YES (after each model)")
    print("  RAM tracking: ACCURATE (continuous monitoring)")
    print("  Timing: PRECISE (per-phase tracking)")
    print("  Per-model analysis plots: YES")
    print()
    print("Models to evaluate:")
    print("  1. Logistic Regression")
    print("  2. Random Forest (50 estimators)")
    print("  3. Gradient Boosting")
    print("  4. Hist Gradient Boosting")
    print("  5. Decision Tree")
    print("  6. AdaBoost")
    print("  7. Ridge Classifier")
    print("  8. Naive Bayes")
    print("  9. SGD Classifier")
    print(" 10. Extra Trees")
    print()
    print("Hyperparameter tuning will test:")
    print("  - Logistic Regression: 20 combinations")
    print("  - Random Forest: 96 combinations")
    print("  - Gradient Boosting: 240 combinations")
    print("  - Hist Gradient Boosting: 108 combinations")
    print("  - Decision Tree: 100 combinations")
    print("  - AdaBoost: 12 combinations")
    print("  - Ridge Classifier: 15 combinations")
    print("  - Naive Bayes: 5 combinations")
    print("  - SGD Classifier: 32 combinations")
    print("  - Extra Trees: 96 combinations")
    print()
    print("Expected total time: 8-15 minutes (depends on dataset size)")
    print()
    input("Press Enter to start evaluation...")
    
    # Run evaluation
    results, tuning_results = run_incremental_evaluation(
        cv_splits=5,
        tune_hyperparameters=True,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nTotal models evaluated: {len(results)}")
    print(f"Total models with hyperparameter tuning: {sum(1 for t in tuning_results.values() if t)}")
    print("\nCheck 'outputs/' directory for:")
    print("  📊 outputs/figures/individual_models/")
    print("     - Comprehensive analysis plot for each model")
    print("     - 7 subplots per model (metrics, timing, memory, distributions)")
    print()
    print("  📈 outputs/figures/")
    print("     - 01_metrics_comparison.png")
    print("     - 02_training_times.png")
    print("     - 03_memory_usage.png")
    print("     - 04_performance_radar.png")
    print("     - 05_performance_heatmap.png")
    print("     - 06_efficiency_plot.png")
    print()
    print("  📝 outputs/reports/individual_models/")
    print("     - JSON results for each model")
    print("     - Includes CV results + hyperparameter tuning results")
    print()
    print("  📄 outputs/reports/final_evaluation_report.json")
    print("     - Complete summary of all models")
    print("     - Best model by each metric")
    print("     - Resource usage statistics")
    

if __name__ == "__main__":
    main()
