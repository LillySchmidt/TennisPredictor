"""Lightweight model configurations for resource-constrained systems.

This module provides memory-efficient model configurations that:
- Use fewer estimators for ensemble methods
- Reduce tree depth to limit memory usage
- Enable parallel processing efficiently
- Provide incremental/batch processing options
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
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


def make_lightweight_models(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """Return memory-efficient model pipelines for large datasets.
    
    These models use reduced complexity to avoid memory issues:
    - Fewer estimators (100-200 vs 300-400)
    - Limited tree depth
    - Efficient algorithms (SGD, HistGradient)
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Dictionary of model name -> pipeline
    """
    preprocessor, _, _ = build_preprocessor(X)

    # Logistic Regression - Very efficient
    logreg = _pipe(
        preprocessor,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=1),
    )

    # Random Forest - REDUCED from 400 to 100 estimators, limited depth
    rf = _pipe(
        preprocessor,
        RandomForestClassifier(
            n_estimators=100,  # Reduced from 400
            max_depth=20,      # Limited depth
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            max_features="sqrt",  # Reduce feature subsampling
            n_jobs=-1,
            random_state=42,
        ),
    )

    # Gradient Boosting - REDUCED estimators
    gb = _pipe(
        preprocessor,
        GradientBoostingClassifier(
            n_estimators=100,  # Reduced from 200
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,     # Sample 80% of data
            random_state=42
        ),
    )

    # Histogram Gradient Boosting - Most efficient boosting method
    hgb = _pipe(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=100,      # Reduced from 200
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        ),
    )

    # K-Nearest Neighbors - Memory efficient with distance weighting
    knn = _pipe(
        preprocessor,
        KNeighborsClassifier(
            n_neighbors=15,    # Reduced from 25
            weights="distance",
            p=2,
            n_jobs=-1
        ),
    )

    # Decision Tree - Fast and memory efficient
    dt = _pipe(
        preprocessor,
        DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
    )

    # AdaBoost - REDUCED estimators
    ada = _pipe(
        preprocessor,
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=50,   # Reduced from 100
            learning_rate=1.0,
            random_state=42
        ),
    )

    # Ridge Classifier - Very fast
    ridge = _pipe(
        preprocessor,
        RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42),
    )

    # Naive Bayes - Fastest method
    nb = _pipe(
        preprocessor,
        GaussianNB(),
    )

    # SGD Classifier - Memory efficient for large datasets
    sgd = _pipe(
        preprocessor,
        SGDClassifier(
            loss="log_loss",  # Logistic regression with SGD
            penalty="l2",
            alpha=0.0001,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

    return {
        "logistic_regression": logreg,
        "random_forest_lite": rf,
        "gradient_boosting_lite": gb,
        "hist_gradient_boosting": hgb,
        "knn_lite": knn,
        "decision_tree": dt,
        "adaboost_lite": ada,
        "ridge_classifier": ridge,
        "naive_bayes": nb,
        "sgd_classifier": sgd,
    }


def get_lightweight_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """Return smaller hyperparameter grids for faster tuning.
    
    Returns:
        Dictionary mapping model names to their hyperparameter grids
    """
    return {
        "logistic_regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
        "random_forest_lite": {
            "clf__n_estimators": [50, 100, 150],
            "clf__max_depth": [15, 20, 25],
            "clf__min_samples_split": [10, 20],
        },
        "gradient_boosting_lite": {
            "clf__n_estimators": [50, 100, 150],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        },
        "hist_gradient_boosting": {
            "clf__max_iter": [50, 100, 150],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [5, 10],
        },
        "knn_lite": {
            "clf__n_neighbors": [10, 15, 20],
            "clf__weights": ["uniform", "distance"],
        },
        "decision_tree": {
            "clf__max_depth": [10, 15, 20],
            "clf__min_samples_split": [20, 50],
        },
        "adaboost_lite": {
            "clf__n_estimators": [30, 50, 70],
            "clf__learning_rate": [0.5, 1.0],
        },
        "ridge_classifier": {
            "clf__alpha": [0.1, 1.0, 10.0],
        },
        "naive_bayes": {
            "clf__var_smoothing": [1e-9, 1e-8],
        },
        "sgd_classifier": {
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__penalty": ["l2", "elasticnet"],
        },
    }
