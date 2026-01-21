"""Optimized models with comprehensive hyperparameter grids.

Features:
- All models use n_jobs=-1 for maximum performance
- No stacking ensemble (too slow)
- KNN optimized for speed with algorithm='ball_tree'
- Comprehensive hyperparameter grids for tuning
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
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

if __package__ is None or __package__ == "":
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from preprocess import build_preprocessor
else:
    from .preprocess import build_preprocessor


def _pipe(preprocessor, estimator):
    """Attach preprocessing to an estimator."""
    return Pipeline([("prep", clone(preprocessor)), ("clf", estimator)])


def make_models(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """Return optimized model pipelines. NO STACKING ENSEMBLE. NO KNN (too slow).
    
    All models use n_jobs=-1 for parallel processing.
    Returns 10 fast, efficient models.
    """
    preprocessor, _, _ = build_preprocessor(X)

    logreg = _pipe(
        preprocessor,
        LogisticRegression(
            max_iter=1000, 
            class_weight="balanced", 
            random_state=42, 
            n_jobs=-1,
            solver='saga'
        ),
    )

    rf = _pipe(
        preprocessor,
        RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    )

    gb = _pipe(
        preprocessor,
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
    )

    hgb = _pipe(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        ),
    )

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

    ada = _pipe(
        preprocessor,
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
    )

    ridge = _pipe(
        preprocessor,
        RidgeClassifier(
            alpha=1.0, 
            class_weight="balanced", 
            random_state=42
        ),
    )

    nb = _pipe(
        preprocessor,
        GaussianNB(),
    )

    sgd = _pipe(
        preprocessor,
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
    )
    
    # Extra Trees - Similar to Random Forest but faster
    et = _pipe(
        preprocessor,
        ExtraTreesClassifier(
            n_estimators=50,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    )

    return {
        "logistic_regression": logreg,
        "random_forest": rf,
        "gradient_boosting": gb,
        "hist_gradient_boosting": hgb,
        "decision_tree": dt,
        "adaboost": ada,
        "ridge_classifier": ridge,
        "naive_bayes": nb,
        "sgd_classifier": sgd,
        "extra_trees": et,
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """Return comprehensive hyperparameter grids for tuning.
    
    Expanded grids with more parameter combinations for better optimization.
    """
    return {
        "logistic_regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["saga"],  # saga supports both l1 and l2
            "clf__max_iter": [1000, 2000],
        },
        "random_forest": {
            "clf__n_estimators": [30, 50, 70, 100],
            "clf__max_depth": [10, 15, 20, None],
            "clf__min_samples_split": [5, 10, 20],
            "clf__min_samples_leaf": [2, 4, 8],
            "clf__max_features": ["sqrt", "log2"],
        },
        "gradient_boosting": {
            "clf__n_estimators": [50, 100, 150, 200],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__min_samples_split": [2, 5, 10],
        },
        "hist_gradient_boosting": {
            "clf__max_iter": [50, 100, 150, 200],
            "clf__learning_rate": [0.05, 0.1, 0.15, 0.2],
            "clf__max_depth": [5, 7, 10, None],
            "clf__min_samples_leaf": [10, 20, 30],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        },
        "decision_tree": {
            "clf__max_depth": [5, 10, 15, 20, None],
            "clf__min_samples_split": [10, 20, 30, 50],
            "clf__min_samples_leaf": [5, 10, 15, 20],
            "clf__criterion": ["gini", "entropy"],
            "clf__max_features": ["sqrt", "log2", None],
        },
        "adaboost": {
            "clf__n_estimators": [30, 50, 70, 100],
            "clf__learning_rate": [0.5, 1.0, 1.5, 2.0],
            "clf__algorithm": ["SAMME", "SAMME.R"],
        },
        "ridge_classifier": {
            "clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__solver": ["auto", "saga", "lsqr"],
            "clf__max_iter": [1000, 2000],
        },
        "naive_bayes": {
            "clf__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        },
        "sgd_classifier": {
            "clf__alpha": [0.00001, 0.0001, 0.001, 0.01],
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__learning_rate": ["optimal", "constant", "adaptive"],
            "clf__max_iter": [1000, 2000],
        },
        "extra_trees": {
            "clf__n_estimators": [30, 50, 70, 100],
            "clf__max_depth": [10, 15, 20, None],
            "clf__min_samples_split": [5, 10, 20],
            "clf__min_samples_leaf": [2, 4, 8],
            "clf__max_features": ["sqrt", "log2"],
        },
    }
