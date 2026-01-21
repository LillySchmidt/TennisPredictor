"""
Comprehensive visualization module for ML/DNN training results.
Generates publication-quality plots for classification and regression tasks.
"""
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})


# =============================================================================
# TRAINING HISTORY PLOTS
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training History",
    metric_name: str = "Loss"
) -> plt.Figure:
    """
    Plot training and validation loss/metric curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', and optionally metric keys
        save_path: Path to save the figure
        title: Plot title
        metric_name: Name of the primary metric
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metric plot (if available)
    ax2 = axes[1]
    metric_keys = [k for k in history.keys() if k not in ['train_loss', 'val_loss', 'lr']]
    
    if metric_keys:
        for key in metric_keys[:4]:  # Limit to 4 metrics
            values = history[key]
            label = key.replace('_', ' ').title()
            ax2.plot(epochs, values, '-', label=label, linewidth=2)
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Validation Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No additional metrics recorded', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Validation Metrics')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# CLASSIFICATION VISUALIZATIONS
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    class_names: List[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with counts and percentages.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save figure
        title: Plot title
        class_names: Names for classes
        
    Returns:
        matplotlib Figure object
    """
    if class_names is None:
        class_names = ['Negative (0)', 'Positive (1)']
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum() * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Count')
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)
    
    # Add summary statistics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    stats_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
    ax.text(0.5, -0.15, stats_text, ha='center', va='top', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Plot ROC curve with AUC and optimal threshold.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark optimal point
    ax.scatter([best_fpr], [best_tpr], c='red', s=100, zorder=5, 
              label=f'Optimal (Threshold={best_threshold:.3f})')
    ax.annotate(f'TPR={best_tpr:.3f}\nFPR={best_fpr:.3f}',
               xy=(best_fpr, best_tpr), xytext=(best_fpr + 0.1, best_tpr - 0.1),
               fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add AUC text box
    ax.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve"
) -> plt.Figure:
    """
    Plot Precision-Recall curve with Average Precision.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    # Baseline (proportion of positive class)
    baseline = np.mean(y_true)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    # Find F1-optimal point
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    ax.scatter([recall[best_idx]], [precision[best_idx]], c='red', s=100, zorder=5,
              label=f'Best F1 = {f1_scores[best_idx]:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Calibration Curve"
) -> plt.Figure:
    """
    Plot calibration (reliability) curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Calculate calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    mean_predicted_values = []
    fraction_of_positives = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_predicted_values.append(np.mean(y_prob[mask]))
            fraction_of_positives.append(np.mean(y_true[mask]))
            bin_counts.append(np.sum(mask))
        else:
            mean_predicted_values.append(np.nan)
            fraction_of_positives.append(np.nan)
            bin_counts.append(0)
    
    # Calculate ECE
    ece = 0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / len(y_true)) * abs(fraction_of_positives[i] - mean_predicted_values[i])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax1.plot(mean_predicted_values, fraction_of_positives, 's-', color='blue',
            markersize=8, linewidth=2, label=f'Model (ECE={ece:.4f})')
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Reliability Diagram', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Predictions', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Probability Distribution by Class"
) -> plt.Figure:
    """
    Plot distribution of predicted probabilities by actual class.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prob_neg = y_prob[y_true == 0]
    prob_pos = y_prob[y_true == 1]
    
    bins = np.linspace(0, 1, 51)
    
    ax.hist(prob_neg, bins=bins, alpha=0.6, label=f'Negative (n={len(prob_neg)})', color='blue')
    ax.hist(prob_pos, bins=bins, alpha=0.6, label=f'Positive (n={len(prob_pos)})', color='red')
    
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    if len(prob_neg) > 1:
        kde_neg = gaussian_kde(prob_neg)
        x_range = np.linspace(0, 1, 200)
        ax.plot(x_range, kde_neg(x_range) * len(prob_neg) * 0.02, 'b-', linewidth=2)
    if len(prob_pos) > 1:
        kde_pos = gaussian_kde(prob_pos)
        x_range = np.linspace(0, 1, 200)
        ax.plot(x_range, kde_pos(x_range) * len(prob_pos) * 0.02, 'r-', linewidth=2)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# REGRESSION VISUALIZATIONS
# =============================================================================

def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Regression Analysis"
) -> plt.Figure:
    """
    Plot regression predictions analysis with multiple subplots.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    residuals = y_true - y_pred
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax1.plot(x_line, p(x_line), 'g-', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    r2 = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Predicted vs Actual (R² = {r2:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(residuals), color='g', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(residuals):.3f}')
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Residual Distribution (std={np.std(residuals):.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Error Analysis"
) -> plt.Figure:
    """
    Detailed error analysis visualization.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Binned MAE by target value
    ax1 = fig.add_subplot(gs[0, 0])
    n_bins = 10
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_true, percentiles)
    bin_indices = np.digitize(y_true, bin_edges[1:-1])
    
    bin_maes = []
    bin_centers = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_maes.append(np.mean(abs_errors[mask]))
            bin_centers.append(np.mean(y_true[mask]))
    
    ax1.bar(range(len(bin_maes)), bin_maes, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Target Value Bin')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('MAE by Target Value Range')
    ax1.set_xticks(range(len(bin_maes)))
    ax1.set_xticklabels([f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(len(bin_maes))],
                        rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative error distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    
    # Mark percentiles
    for p in [50, 90, 95, 99]:
        val = np.percentile(abs_errors, p)
        ax2.axvline(x=val, color='gray', linestyle='--', alpha=0.7)
        ax2.text(val, p/100, f' P{p}={val:.2f}', fontsize=9)
    
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Cumulative Proportion')
    ax2.set_title('Cumulative Error Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of errors by decile
    ax3 = fig.add_subplot(gs[1, 0])
    box_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            box_data.append(abs_errors[mask])
    ax3.boxplot(box_data, labels=[f'D{i+1}' for i in range(len(box_data))])
    ax3.set_xlabel('Decile of Target Value')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Error Distribution by Decile')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error percentiles table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    percentile_values = [25, 50, 75, 90, 95, 99]
    error_percentiles = [np.percentile(abs_errors, p) for p in percentile_values]
    
    table_data = [
        ['Metric', 'Value'],
        ['Mean Absolute Error', f'{np.mean(abs_errors):.4f}'],
        ['Median Absolute Error', f'{np.median(abs_errors):.4f}'],
        ['RMSE', f'{np.sqrt(np.mean(residuals**2)):.4f}'],
        ['', ''],
        ['Percentile', 'Error'],
    ]
    for p, v in zip(percentile_values, error_percentiles):
        table_data.append([f'P{p}', f'{v:.4f}'])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Error Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# COMPARISON VISUALIZATIONS
# =============================================================================

def plot_model_comparison_bar(
    models_metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    save_path: Optional[Path] = None,
    title: str = None,
    higher_is_better: bool = True
) -> plt.Figure:
    """
    Bar chart comparing a single metric across models.
    
    Args:
        models_metrics: Dictionary {model_name: {metric_name: value}}
        metric_name: Name of metric to compare
        save_path: Path to save figure
        title: Plot title
        higher_is_better: Whether higher values are better
        
    Returns:
        matplotlib Figure object
    """
    model_names = list(models_metrics.keys())
    values = [models_metrics[m].get(metric_name, 0) for m in model_names]
    
    # Sort by value
    sorted_indices = np.argsort(values)
    if higher_is_better:
        sorted_indices = sorted_indices[::-1]
    
    model_names = [model_names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.barh(model_names, values, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01 * max(values), bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=10)
    
    # Highlight best
    best_idx = 0 if higher_is_better else len(values) - 1
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)
    
    ax.set_xlabel(metric_name.replace('_', ' ').title())
    ax.set_title(title or f'Model Comparison: {metric_name.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_metrics_heatmap(
    models_metrics: Dict[str, Dict[str, float]],
    metrics_to_show: List[str] = None,
    save_path: Optional[Path] = None,
    title: str = "Model Comparison Heatmap"
) -> plt.Figure:
    """
    Heatmap comparing multiple metrics across models.
    
    Args:
        models_metrics: Dictionary {model_name: {metric_name: value}}
        metrics_to_show: List of metric names to include
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    model_names = list(models_metrics.keys())
    
    if metrics_to_show is None:
        # Get common metrics
        all_metrics = set()
        for m in models_metrics.values():
            all_metrics.update(m.keys())
        metrics_to_show = sorted(list(all_metrics))[:10]  # Limit to 10 metrics
    
    # Build data matrix
    data = np.zeros((len(model_names), len(metrics_to_show)))
    for i, model in enumerate(model_names):
        for j, metric in enumerate(metrics_to_show):
            data[i, j] = models_metrics[model].get(metric, np.nan)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(model_names) * 0.6)))
    
    # Normalize each column for coloring
    data_norm = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-10)
    
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics_to_show)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_show], rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metrics_to_show)):
            val = data[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                             fontsize=9, color='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Normalized Value')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Feature Importance",
    top_n: int = 20
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    
    Args:
        feature_names: Names of features
        importances: Importance values
        save_path: Path to save figure
        title: Plot title
        top_n: Number of top features to show
        
    Returns:
        matplotlib Figure object
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Take top N
    indices = indices[:top_n]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_names)))
    bars = ax.barh(range(len(top_names)), top_importances, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig
