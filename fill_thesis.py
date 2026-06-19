#!/usr/bin/env python3
"""
Read experimental results and substitute placeholders in updated_master_thesis.md,
then generate tables/figures and convert to PDF.
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TEMPLATE = ROOT / "updated_master_thesis.md"
FILLED = ROOT / "updated_master_thesis_filled.md"


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(x, digits=2):
    if x is None:
        return "N/A"
    return f"{x*100:.{digits}f}"


def fmt_auc(x, digits=4):
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_brier(x, digits=4):
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_ci(low, high, digits=2):
    if low is None or high is None:
        return "N/A"
    return f"[{low*100:.{digits}f}, {high*100:.{digits}f}]"


def fmt_auc_ci(low, high, digits=4):
    if low is None or high is None:
        return "N/A"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def fmt_brier_ci(low, high, digits=4):
    if low is None or high is None:
        return "N/A"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def collect_results():
    data = {}
    data["elo"] = load_json(RESULTS / "elo" / "elo_baseline.json") or {}

    ml_dir = RESULTS / "ml_eloml"
    ml = {}
    pred_ml = ml_dir / "predictions" / "ml"
    if pred_ml.exists():
        for f in sorted(pred_ml.glob("*_test.csv")):
            name = f.stem.replace("_test", "")
            ml[name] = load_json(ml_dir / f"{name}.json") or {}
    data["ml"] = ml

    eloml = {}
    pred_eloml = ml_dir / "predictions" / "eloml"
    if pred_eloml.exists():
        for f in sorted(pred_eloml.glob("*_test.csv")):
            name = f.stem.replace("_test", "")
            eloml[name] = load_json(ml_dir / f"{name}.json") or {}
    data["eloml"] = eloml

    dnn = {}
    dnn_dir = RESULTS / "dnn"
    if dnn_dir.exists():
        for f in sorted(dnn_dir.glob("*.json")):
            val = load_json(f)
            if isinstance(val, dict):
                dnn[f.stem] = val
    data["dnn"] = dnn

    data["shap"] = load_json(RESULTS / "shap" / "shap_summary.json") or {}
    data["timing"] = load_json(RESULTS / "timing" / "inference_timing.json") or []
    data["mcnemar"] = load_json(RESULTS / "mcnemar" / "mcnemar_results.json") or {}
    return data


def metric(model, key):
    if not model:
        return None
    return model.get("metrics", {}).get(key)


def ci(model, key):
    if not model:
        return None, None
    c = model.get("bootstrap_ci", {}).get(key, {})
    return c.get("low"), c.get("high")


def build_tables(data):
    elo = data.get("elo", {})
    ml = data.get("ml", {})
    eloml = data.get("eloml", {})
    dnn = data.get("dnn", {})

    best_ml = max(ml.values(), key=lambda x: x.get("metrics", {}).get("accuracy", 0)) if ml else {}
    best_eloml = max(eloml.values(), key=lambda x: x.get("metrics", {}).get("accuracy", 0)) if eloml else {}
    best_dnn = max(dnn.values(), key=lambda x: x.get("metrics", {}).get("accuracy", 0)) if dnn else {}

    # Overall table
    overall = r"""\begin{table}[H]
\centering
\caption{Overall Performance Summary Across All Approaches}
\label{tab:overall}
\begin{tabular}{lccc}
\toprule
Approach & Accuracy (\%) & ROC-AUC & Brier Score \\
\midrule
"""
    rows = [
        ("Elo Rating System", elo, "elo_baseline"),
        ("Classical ML (Best)", best_ml, "ml_best"),
        ("Deep Neural Networks (Best)", best_dnn, "dnn_best"),
        ("ELO-ML Combined (Best)", best_eloml, "eloml_best"),
    ]
    for label, model, _ in rows:
        acc = fmt_pct(metric(model, "accuracy"))
        auc = fmt_auc(metric(model, "roc_auc"))
        brier = fmt_brier(metric(model, "brier"))
        overall += f"{label} & {acc} & {auc} & {brier} \\\\\n"
    overall += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Elo table
    elo_table = r"""\begin{table}[H]
\centering
\caption{Elo Rating System Performance Metrics on Test Set}
\label{tab:elo}
\begin{tabular}{lc}
\toprule
Metric & Value (95\% CI) \\
\midrule
"""
    for mname in ["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"]:
        v = metric(elo, mname)
        l, h = ci(elo, mname)
        if mname in ["accuracy", "precision", "recall", "f1"]:
            val = fmt_pct(v)
            interval = fmt_ci(l, h)
        else:
            val = fmt_auc(v) if mname == "roc_auc" else fmt_brier(v)
            interval = fmt_auc_ci(l, h) if mname == "roc_auc" else fmt_brier_ci(l, h)
        elo_table += f"{mname.replace('_', ' ').title()} & {val} ({interval}) \\\\\n"
    elo_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # ML table
    ml_table = r"""\begin{table}[H]
\centering
\caption{Classical Machine Learning Algorithm Performance on Test Set}
\label{tab:ml}
\begin{tabular}{lcccc}
\toprule
Algorithm & Accuracy (\%) & ROC-AUC & F1 & Brier \\
\midrule
"""
    for name, model in sorted(ml.items(), key=lambda kv: kv[1].get("metrics", {}).get("accuracy", 0), reverse=True):
        acc = fmt_pct(metric(model, "accuracy"))
        auc = fmt_auc(metric(model, "roc_auc"))
        f1 = fmt_pct(metric(model, "f1"))
        brier = fmt_brier(metric(model, "brier"))
        ml_table += f"{name.replace('_', ' ').title()} & {acc} & {auc} & {f1} & {brier} \\\\\n"
    ml_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # DNN table
    dnn_table = r"""\begin{table}[H]
\centering
\caption{Deep Neural Network Configuration Performance on Test Set}
\label{tab:dnn}
\begin{tabular}{lccccc}
\toprule
Configuration & Params & Acc (\%) & AUC & Brier & ECE \\
\midrule
"""
    for name, model in sorted(dnn.items(), key=lambda kv: kv[1].get("metrics", {}).get("accuracy", 0), reverse=True):
        params = model.get("params", 0)
        params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.0f}K"
        acc = fmt_pct(metric(model, "accuracy"))
        auc = fmt_auc(metric(model, "roc_auc"))
        brier = fmt_brier(metric(model, "brier"))
        ece = fmt_brier(model.get("ece"))
        dnn_table += f"{name.replace('_', ' ').title()} & {params_str} & {acc} & {auc} & {brier} & {ece} \\\\\n"
    dnn_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # ELO-ML table
    eloml_table = r"""\begin{table}[H]
\centering
\caption{Performance Improvement from Elo Feature Augmentation}
\label{tab:eloml}
\begin{tabular}{lccccc}
\toprule
Algorithm & Without Elo & With Elo & $\Delta$ Acc & AUC (with) & Brier (with) \\
\midrule
"""
    name_map = {"eloml_logistic_regression": "Logistic Regression", "eloml_ridge_classifier": "Ridge Classifier", "eloml_adaboost": "AdaBoost"}
    for key, label in name_map.items():
        base = ml.get(key, {})
        aug = eloml.get(key, {})
        base_key = key.replace("eloml_", "")
        base = ml.get(base_key, {})
        without = fmt_pct(metric(base, "accuracy"))
        with_ = fmt_pct(metric(aug, "accuracy"))
        delta = f"{metric(aug, 'accuracy')*100 - metric(base, 'accuracy')*100:+.2f}" if metric(aug, "accuracy") and metric(base, "accuracy") else "N/A"
        auc = fmt_auc(metric(aug, "roc_auc"))
        brier = fmt_brier(metric(aug, "brier"))
        eloml_table += f"{label} & {without} & {with_} & {delta} pp & {auc} & {brier} \\\\\n"
    eloml_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Calibration table
    calib_table = r"""\begin{table}[H]
\centering
\caption{Calibration Quality Comparison Across Approaches}
\label{tab:calibration}
\begin{tabular}{lcccc}
\toprule
Approach & ECE & Brier & Log Loss & Calibration \\
\midrule
"""
    for label, model in [
        ("DNN (Best)", best_dnn),
        ("ELO-ML Combined (Best)", best_eloml),
        ("Classical ML (Best)", best_ml),
        ("Elo System", elo),
    ]:
        ece = fmt_brier(model.get("ece")) if isinstance(model.get("ece"), float) else fmt_brier(metric(model, "brier"))
        if model.get("ece") is not None:
            ece = fmt_brier(model.get("ece"))
        brier = fmt_brier(metric(model, "brier"))
        ll = fmt_brier(metric(model, "log_loss"))
        quality = "Excellent" if (model.get("ece") or 1) < 0.01 else "Very Good" if (model.get("ece") or 1) < 0.015 else "Good"
        if model == elo:
            quality = "Good"
        calib_table += f"{label} & {ece} & {brier} & {ll} & {quality} \\\\\n"
    calib_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # McNemar table from computed results
    mcnemar_rows = data.get("mcnemar", {}).get("comparisons", [])
    mcnemar = r"""\begin{table}[H]
\centering
\caption{Statistical Significance Testing Results (McNemar's Test)}
\label{tab:mcnemar}
\begin{tabular}{lcc}
\toprule
Comparison & Acc. Diff. & p-value \\
\midrule
"""
    for row in mcnemar_rows[:6]:
        comp = row.get("comparison", "")
        diff = row.get("acc_diff_pp", 0)
        p = row.get("p_value", 1)
        diff_str = f"{diff:+.2f}"
        if p < 0.001:
            p_str = "$<0.001$"
        elif p == 1.0:
            p_str = "$1.000$"
        else:
            p_str = f"${p:.4f}$"
        mcnemar += f"{comp} & {diff_str} pp & {p_str} \\\\\n"
    mcnemar += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Efficiency table from timing results
    timing_rows = []
    for t in data.get("timing", []):
        name = t.get("model", "")
        approach = t.get("approach", "")
        mean = t.get("timing_seconds", {}).get("mean", 0)
        std = t.get("timing_seconds", {}).get("std", 0)
        timing_rows.append((approach, name, mean, std))

    eff_table = r"""\begin{table}[H]
\centering
\caption{Measured Computational Requirements Comparison Across Approaches}
\label{tab:efficiency}
\begin{tabular}{lcc}
\toprule
Approach & Inference Time (s) & Per 1,000 Pred. (ms) \\
\midrule
"""
    for _, name, mean, std in timing_rows:
        per1k = mean / 74307 * 1000 * 1000 if mean else 0
        eff_table += f"{name.replace('_', ' ').title()} & {mean:.4f} $\\pm$ {std:.4f} & {per1k:.2f} \\\\\n"
    eff_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return {
        "overall_table": overall,
        "elo_table": elo_table,
        "ml_table": ml_table,
        "dnn_table": dnn_table,
        "eloml_table": eloml_table,
        "calibration_table": calib_table,
        "mcnemar_table": mcnemar,
        "efficiency_table": eff_table,
        "best_ml_name": (max(ml.items(), key=lambda kv: kv[1].get("metrics", {}).get("accuracy", 0))[0].replace("_", " ").title() if ml else "N/A"),
        "best_eloml_name": (max(eloml.items(), key=lambda kv: kv[1].get("metrics", {}).get("accuracy", 0))[0].replace("eloml_", "").replace("_", " ").title() if eloml else "N/A"),
        "best_dnn_name": (max(dnn.items(), key=lambda kv: kv[1].get("metrics", {}).get("accuracy", 0))[0].replace("dnn_", "").replace("_", " ").title() if dnn else "N/A"),
        "best_ml_accuracy": fmt_pct(metric(best_ml, "accuracy")),
        "best_eloml_accuracy": fmt_pct(metric(best_eloml, "accuracy")),
        "best_dnn_accuracy": fmt_pct(metric(best_dnn, "accuracy")),
        "elo_accuracy": fmt_pct(metric(elo, "accuracy")),
    }


def generate_accuracy_figure(data):
    """Generate Figure: accuracy comparison with bootstrap CIs."""
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    models = []
    for name, model in data.get("ml", {}).items():
        models.append((name.replace("_", " ").title(), model, "ML"))
    for name, model in data.get("eloml", {}).items():
        models.append((name.replace("_", " ").title() + " +Elo", model, "ELO-ML"))
    for name, model in data.get("dnn", {}).items():
        models.append((name.replace("_", " ").title(), model, "DNN"))
    models.append(("Elo Baseline", data.get("elo", {}), "Elo"))

    # Sort by accuracy
    models.sort(key=lambda x: x[1].get("metrics", {}).get("accuracy", 0), reverse=True)

    names = [m[0] for m in models]
    accs = [m[1].get("metrics", {}).get("accuracy", 0) * 100 for m in models]
    lows = [(m[1].get("bootstrap_ci", {}).get("accuracy", {}).get("low", 0) or 0) * 100 for m in models]
    highs = [(m[1].get("bootstrap_ci", {}).get("accuracy", {}).get("high", 0) or 0) * 100 for m in models]
    errors = [[a - l for a, l in zip(accs, lows)], [h - a for a, h in zip(accs, highs)]]

    colors = ["#1f77b4" if g == "ML" else "#ff7f0e" if g == "ELO-ML" else "#2ca02c" if g == "DNN" else "#d62728" for _, _, g in models]

    plt.figure(figsize=(10, 7))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, accs, xerr=errors, color=colors, capsize=3)
    plt.yticks(y_pos, names)
    plt.xlabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison Across All Approaches (95% Bootstrap CI)")
    plt.xlim([64, 69])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_comparison.png", dpi=300)
    plt.close()


def main():
    data = collect_results()
    subs = build_tables(data)

    # Generate figure
    generate_accuracy_figure(data)

    # Substitute placeholders
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, val in subs.items():
        text = text.replace("{{" + key + "}}", str(val))

    FILLED.write_text(text, encoding="utf-8")
    print(f"Wrote {FILLED}")

    # Convert to PDF
    pdf_path = ROOT / "updated_master_thesis.pdf"
    xelatex = Path.home() / ".TinyTeX" / "bin" / "x86_64-linux" / "xelatex"
    cmd = [
        "pandoc",
        str(FILLED),
        "-o", str(pdf_path),
        "--pdf-engine", str(xelatex),
        "--toc",
        "-V", "toc-title=Contents",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
