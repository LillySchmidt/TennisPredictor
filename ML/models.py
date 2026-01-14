"""Model factories replicating the classic ML notebook pipelines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

# Support both package and script imports
if __package__ is None or __package__ == "":
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from preprocess import build_preprocessor  # type: ignore
else:
    from .preprocess import build_preprocessor  # type: ignore


def _pipe(preprocessor, estimator):
    """Attach preprocessing to an estimator."""
    return Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("clf", estimator),
        ]
    )


def make_models(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """Return a dict of model pipelines keyed by human-readable names."""
    preprocessor, _, _ = build_preprocessor(X)

    logreg = _pipe(
        preprocessor,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

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

    gb = _pipe(
        preprocessor,
        GradientBoostingClassifier(random_state=42),
    )

    hgb = _pipe(
        preprocessor,
        HistGradientBoostingClassifier(random_state=42),
    )

    svm_rbf = _pipe(
        preprocessor,
        CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=1.0, class_weight="balanced"),
            cv=3,
        ),
    )

    svm_poly = _pipe(
        preprocessor,
        CalibratedClassifierCV(
            estimator=SVC(kernel="poly", degree=3, C=1.0, class_weight="balanced"),
            cv=3,
        ),
    )

    linear_svm = _pipe(
        preprocessor,
        CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced"),
            cv=3,
        ),
    )

    knn = _pipe(
        preprocessor,
        KNeighborsClassifier(n_neighbors=25, weights="distance", p=2),
    )

    stacking = StackingClassifier(
        estimators=[
            ("logreg", _pipe(preprocessor, LogisticRegression(max_iter=1000, class_weight="balanced"))),
            ("rf", _pipe(preprocessor, RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ))),
            ("gb", _pipe(preprocessor, GradientBoostingClassifier(random_state=42))),
            ("hgb", _pipe(preprocessor, HistGradientBoostingClassifier(random_state=42))),
            ("svm_rbf", _pipe(preprocessor, CalibratedClassifierCV(
                estimator=SVC(kernel="rbf", C=1.0, class_weight="balanced"),
                cv=3,
            ))),
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        n_jobs=-1,
        stack_method="auto",
    )

    return {
        "logistic_regression": logreg,
        "random_forest": rf,
        "gradient_boosting": gb,
        "hist_gradient_boosting": hgb,
        "svm_rbf": svm_rbf,
        "svm_poly": svm_poly,
        "linear_svm": linear_svm,
        "knn": knn,
        "stacking_ensemble": stacking,
    }
