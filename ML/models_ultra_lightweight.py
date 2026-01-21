"""Ultra-lightweight model configurations for very large datasets (400k+ samples).

This module provides minimal memory footprint models:
- Random Forest: Only 25 estimators (vs 400 original)
- Strict depth limits
- Sequential processing options
- Designed for systems with 8-16GB RAM
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Support both package and script imports
if __package__ is None or __package__ == "":
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from preprocess import build_preprocessor
else:
    from .preprocess import build_preprocessor


def _pipe(preprocessor, estimator):
    """Attach preprocessing to an estimator."""
    return Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("clf", estimator),
        ]
    )


def make_ultra_lightweight_models(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """Return minimal memory footprint models for very large datasets (400k+ samples).
    
    These models are optimized for systems with 8-16GB RAM:
    - Random Forest: Only 25 estimators (75% reduction from lightweight)
    - No standard Gradient Boosting (too slow/memory intensive)
    - Hist Gradient Boosting only (fastest boosting method)
    - Sequential processing (n_jobs=1) to prevent memory spikes
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Dictionary of model name -> pipeline
    """
    preprocessor, _, _ = build_preprocessor(X)

    # Logistic Regression - Very fast and memory efficient
    logreg = _pipe(
        preprocessor,
        LogisticRegression(
            max_iter=500, 
            class_weight="balanced", 
            random_state=42, 
            n_jobs=1,  # Sequential to avoid memory spikes
            solver='saga'  # Memory efficient solver
        ),
    )

    # Random Forest - ULTRA REDUCED: Only 25 estimators
    rf = _pipe(
        preprocessor,
        RandomForestClassifier(
            n_estimators=25,   # CRITICAL: Reduced from 100 to 25
            max_depth=15,      # Strict depth limit
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',  # Reduce feature sampling
            class_weight="balanced_subsample",
            n_jobs=1,          # Sequential processing
            random_state=42,
            max_samples=0.7,   # Use only 70% of data per tree
        ),
    )

    # Histogram Gradient Boosting - Most efficient boosting (ONLY boosting method)
    hgb = _pipe(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=50,       # Reduced from 100
            learning_rate=0.1,
            max_depth=8,       # Reduced depth
            max_leaf_nodes=31, # Limit tree complexity
            random_state=42
        ),
    )

    # K-Nearest Neighbors - Reduced neighbors
    knn = _pipe(
        preprocessor,
        KNeighborsClassifier(
            n_neighbors=10,    # Reduced from 15
            weights="distance",
            p=2,
            n_jobs=1           # Sequential
        ),
    )

    # Decision Tree - Fast and memory efficient
    dt = _pipe(
        preprocessor,
        DecisionTreeClassifier(
            max_depth=12,      # Reduced from 15
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight="balanced",
            random_state=42
        ),
    )

    # AdaBoost - Minimal estimators
    ada = _pipe(
        preprocessor,
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=30,   # Reduced from 50
            learning_rate=1.0,
            random_state=42
        ),
    )

    # Ridge Classifier - Very fast linear model
    ridge = _pipe(
        preprocessor,
        RidgeClassifier(
            alpha=1.0, 
            class_weight="balanced", 
            random_state=42,
            solver='saga'
        ),
    )

    # Naive Bayes - Fastest model
    nb = _pipe(
        preprocessor,
        GaussianNB(),
    )

    # SGD Classifier - Memory efficient, trains on batches
    sgd = _pipe(
        preprocessor,
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    )

    return {
        "logistic_regression": logreg,
        "random_forest_ultra": rf,
        "hist_gradient_boosting": hgb,
        "knn": knn,
        "decision_tree": dt,
        "adaboost": ada,
        "ridge_classifier": ridge,
        "naive_bayes": nb,
        "sgd_classifier": sgd,
    }


def get_ultra_lightweight_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """Return minimal hyperparameter grids for fastest tuning.
    
    Returns:
        Dictionary mapping model names to their hyperparameter grids
    """
    return {
        "logistic_regression": {
            "clf__C": [0.1, 1.0],
            "clf__penalty": ["l2"],
        },
        "random_forest_ultra": {
            "clf__n_estimators": [20, 25, 30],
            "clf__max_depth": [10, 15],
        },
        "hist_gradient_boosting": {
            "clf__max_iter": [30, 50],
            "clf__learning_rate": [0.1, 0.2],
        },
        "knn": {
            "clf__n_neighbors": [8, 10, 12],
        },
        "decision_tree": {
            "clf__max_depth": [10, 12],
        },
        "adaboost": {
            "clf__n_estimators": [20, 30],
        },
        "ridge_classifier": {
            "clf__alpha": [0.1, 1.0],
        },
        "naive_bayes": {},
        "sgd_classifier": {
            "clf__alpha": [0.0001, 0.001],
        },
    }
