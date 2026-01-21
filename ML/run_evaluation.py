"""Main runner script for comprehensive ML evaluation.

Usage:
    python run_evaluation.py [--no-augment] [--no-tune] [--cv-folds N]

This script:
1. Loads and preprocesses tennis match data
2. Optionally augments the dataset for better model training
3. Trains multiple ML models (no SVM - too slow)
4. Performs cross-validation evaluation
5. Conducts hyperparameter tuning
6. Generates visualizations and comparison plots
7. Exports results to JSON format
"""

import argparse
import sys
from pathlib import Path

# Ensure local imports work
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate_comprehensive import run_comprehensive_evaluation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive ML evaluation on tennis match data"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to raw data CSV file (default: all_matches.csv)"
    )
    
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    
    parser.add_argument(
        "--augmentation-factor",
        type=float,
        default=2.5,
        help="Data augmentation multiplier (default: 2.5)"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)"
    )
    
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use memory-efficient models for large datasets (recommended for 400k+ samples)"
    )
    
    parser.add_argument(
        "--ultra-lightweight",
        action="store_true",
        help="Use minimal memory models (RF=25 estimators, sequential processing). For 400k+ samples with limited RAM."
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*80)
    print("TENNIS MATCH PREDICTION - ML EVALUATION")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Data augmentation: {not args.no_augment}")
    if not args.no_augment:
        print(f"  Augmentation factor: {args.augmentation_factor}x")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Hyperparameter tuning: {not args.no_tune}")
    if args.ultra_lightweight:
        print(f"  Mode: ULTRA-LIGHTWEIGHT (RF=25 estimators, sequential processing)")
    elif args.lightweight:
        print(f"  Mode: LIGHTWEIGHT (RF=100 estimators)")
    else:
        print(f"  Mode: STANDARD (RF=400 estimators)")
    print(f"  Random state: {args.random_state}")
    print()
    
    # Run evaluation
    cv_results, hp_results = run_comprehensive_evaluation(
        data_path=args.data_path,
        augment_data=not args.no_augment,
        augmentation_factor=args.augmentation_factor,
        cv_splits=args.cv_folds,
        tune_hyperparameters=not args.no_tune,
        random_state=args.random_state,
        lightweight_mode=args.lightweight,
        ultra_lightweight_mode=args.ultra_lightweight,
    )
    
    print("\n" + "="*80)
    print("SUCCESS: Evaluation complete!")
    print("="*80)
    print("\nCheck the 'outputs' directory for:")
    print("  - figures/: All visualization plots")
    print("  - reports/: JSON evaluation report")
    

if __name__ == "__main__":
    main()
