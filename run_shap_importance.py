#!/usr/bin/env python3
"""
Compute SHAP values and permutation importance for a Random Forest trained on
the unified ML training split. Produces the updated Figure 4 feature-importance plot.
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
UNIFIED_DIR = ROOT / "unified_data"
OUT_DIR = ROOT / "results" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ml_preprocess = load_module("ml_preprocess", REPO / "ML" / "preprocess.py")
build_preprocessor = ml_preprocess.build_preprocessor


def get_feature_names(preprocessor, X):
    """Extract final feature names from sklearn ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return feature_names


def main():
    print("Loading data...")
    train = pd.read_csv(UNIFIED_DIR / "ml_train.csv")
    test = pd.read_csv(UNIFIED_DIR / "ml_test.csv")
    feature_cols = [c for c in train.columns if c != "label"]
    X_train, y_train = train[feature_cols], train["label"].values
    X_test, y_test = test[feature_cols], test["label"].values

    print("Building preprocessor...")
    preprocessor, _, _ = build_preprocessor(X_train)
    X_train_tx = preprocessor.fit_transform(X_train)
    X_test_tx = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, X_train)
    print(f"Features after preprocessing ({len(feature_names)}): {feature_names}")

    print("Training Random Forest (100 trees for SHAP speed)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_tx, y_train)

    # Native feature importance
    native_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    native_imp.to_csv(OUT_DIR / "random_forest_native_importance.csv", index=False)

    # Permutation importance on a subset for speed
    print("Computing permutation importance on subset (3 repeats)...")
    np.random.seed(42)
    perm_sample_idx = np.random.choice(len(X_test_tx), size=min(5000, len(X_test_tx)), replace=False)
    perm = permutation_importance(
        rf, X_test_tx[perm_sample_idx], y_test[perm_sample_idx],
        n_repeats=3, random_state=42, n_jobs=-1, scoring="roc_auc"
    )
    perm_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False)
    perm_imp.to_csv(OUT_DIR / "random_forest_permutation_importance.csv", index=False)

    # SHAP values using TreeExplainer
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(rf)
    sample_size = min(500, len(X_test_tx))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test_tx), size=sample_size, replace=False)
    X_sample = X_test_tx[sample_idx]
    shap_values = explainer.shap_values(X_sample)
    print(f"SHAP values type: {type(shap_values)}, shape info will be printed below")
    if isinstance(shap_values, list):
        print(f"List of {len(shap_values)} arrays, first shape: {shap_values[0].shape}")
        shap_values = shap_values[1]
    elif hasattr(shap_values, "values"):
        shap_values = shap_values.values
    print(f"Using SHAP array shape: {getattr(shap_values, 'shape', 'N/A')}")

    # Ensure 2D: (n_samples, n_features)
    shap_values = np.atleast_2d(np.asarray(shap_values))
    if shap_values.ndim > 2:
        shap_values = shap_values[..., 1]  # take positive class if last dim is class

    shap_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False)
    shap_imp.to_csv(OUT_DIR / "random_forest_shap_importance.csv", index=False)

    # Save SHAP values matrix
    np.savez(OUT_DIR / "shap_values.npz", shap_values=shap_values, sample_idx=sample_idx, feature_names=np.array(feature_names))

    # Plot top 10 SHAP feature importances
    plt.figure(figsize=(8, 6))
    top_n = 10
    top = shap_imp.head(top_n).sort_values("importance", ascending=True)
    plt.barh(top["feature"], top["importance"], color="steelblue")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 10 Feature Importances (Random Forest, SHAP)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure4_shap_importance.png", dpi=300)
    plt.savefig(OUT_DIR / "figure4_shap_importance.pdf")
    plt.close()

    # SHAP summary plot (beeswarm)
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False, plot_size=(8, 8))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure4_shap_beeswarm.png", dpi=300)
    plt.close()

    # Plot comparison of native, permutation, SHAP
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, df, title in zip(axes, [native_imp, perm_imp, shap_imp],
                             ["Native Gini", "Permutation (AUC)", "SHAP |value|"]):
        top = df.head(top_n).sort_values("importance", ascending=True)
        ax.barh(top["feature"], top["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_comparison.png", dpi=300)
    plt.close()

    # Summary JSON
    summary = {
        "model": "random_forest",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "shap_sample_size": sample_size,
        "top_features_shap": shap_imp.head(10).to_dict("records"),
        "top_features_native": native_imp.head(10).to_dict("records"),
        "top_features_permutation": perm_imp.head(10).to_dict("records"),
    }
    with open(OUT_DIR / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"SHAP results saved to {OUT_DIR}")
    print("Top 10 SHAP features:")
    print(shap_imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
