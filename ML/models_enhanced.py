"""Enhanced model factories with hyperparameter grids and no SVM models.

This module provides:
- Model pipelines without computationally expensive SVM variants
- Hyperparameter grids for each model type
- Professional model configurations following ML best practices
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
    StackingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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


def make_models(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """Return a dict of model pipelines keyed by human-readable names.
    
    SVM models removed due to computational cost.
    """
    preprocessor, _, _ = build_preprocessor(X)

    # Logistic Regression - Simple baseline
    logreg = _pipe(
        preprocessor,
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )

    # Random Forest - Ensemble of decision trees
    rf = _pipe(
        preprocessor,
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    )

    # Gradient Boosting - Sequential ensemble
    gb = _pipe(
        preprocessor,
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    )

    # Histogram Gradient Boosting - Faster gradient boosting
    hgb = _pipe(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    )

    # K-Nearest Neighbors - Distance-based classifier
    knn = _pipe(
        preprocessor,
        KNeighborsClassifier(n_neighbors=25, weights="distance", p=2, n_jobs=-1),
    )

    # Decision Tree - Single tree for interpretability
    dt = _pipe(
        preprocessor,
        DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42
        ),
    )

    # AdaBoost - Adaptive boosting
    ada = _pipe(
        preprocessor,
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
    )

    # Ridge Classifier - Linear model with L2 regularization
    ridge = _pipe(
        preprocessor,
        RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42),
    )

    # Naive Bayes - Probabilistic classifier
    nb = _pipe(
        preprocessor,
        GaussianNB(),
    )

    # Stacking Ensemble - Meta-learner combining multiple models (no SVM)
    stacking = StackingClassifier(
        estimators=[
            ("logreg", _pipe(preprocessor, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))),
            ("rf", _pipe(preprocessor, RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ))),
            ("gb", _pipe(preprocessor, GradientBoostingClassifier(n_estimators=150, random_state=42))),
            ("hgb", _pipe(preprocessor, HistGradientBoostingClassifier(max_iter=150, random_state=42))),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        n_jobs=-1,
        stack_method="auto",
    )

    return {
        "logistic_regression": logreg,
        "random_forest": rf,
        "gradient_boosting": gb,
        "hist_gradient_boosting": hgb,
        "knn": knn,
        "decision_tree": dt,
        "adaboost": ada,
        "ridge_classifier": ridge,
        "naive_bayes": nb,
        "stacking_ensemble": stacking,
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """Return hyperparameter grids for each model type.
    
    Returns:
        Dictionary mapping model names to their hyperparameter grids
    """
    return {
        "logistic_regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "saga"],
        },
        "random_forest": {
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_split": [2, 4, 8],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.8, 1.0],
        },
        "hist_gradient_boosting": {
            "clf__max_iter": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7, None],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        },
        "knn": {
            "clf__n_neighbors": [5, 15, 25, 35, 50],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # 1: Manhattan, 2: Euclidean
        },
        "decision_tree": {
            "clf__max_depth": [5, 10, 15, 20, None],
            "clf__min_samples_split": [2, 10, 20, 50],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__criterion": ["gini", "entropy"],
        },
        "adaboost": {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.5, 1.0, 1.5],
        },
        "ridge_classifier": {
            "clf__alpha": [0.1, 1.0, 10.0, 100.0],
            "clf__solver": ["auto", "saga"],
        },
        "naive_bayes": {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
    }
