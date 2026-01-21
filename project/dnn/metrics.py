"""
Comprehensive metrics module for ML/DNN evaluation.
Includes classification, regression, calibration, and bootstrap CI metrics.
"""
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, max_error, median_absolute_error
)

warnings.filterwarnings('ignore')


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def classification_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        threshold: Classification threshold
        
    Returns:
        Dictionary with all classification metrics
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # ROC-AUC and PR-AUC
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics['roc_auc'] = float('nan')
    
    try:
        metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics['pr_auc'] = float('nan')
    
    # Probabilistic metrics
    try:
        metrics['log_loss'] = float(log_loss(y_true, y_prob))
    except ValueError:
        metrics['log_loss'] = float('nan')
    
    metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
    
    # Advanced metrics
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = metrics['recall']
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        metrics['positive_predictive_value'] = metrics['precision']
        metrics['negative_predictive_value'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    # Class distribution
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true == 1))
    metrics['n_negative'] = int(np.sum(y_true == 0))
    metrics['positive_rate'] = float(np.mean(y_true))
    
    # Prediction statistics
    metrics['pred_positive_rate'] = float(np.mean(y_pred))
    metrics['prob_mean'] = float(np.mean(y_prob))
    metrics['prob_std'] = float(np.std(y_prob))
    metrics['prob_min'] = float(np.min(y_prob))
    metrics['prob_max'] = float(np.max(y_prob))
    
    metrics['threshold_used'] = threshold
    
    return metrics


def compute_optimal_thresholds(
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Find optimal classification thresholds using different criteria.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    
    results = {}
    
    # Youden's J statistic
    try:
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        results['youden_j'] = {
            'threshold': float(thresholds_roc[best_idx]),
            'sensitivity': float(tpr[best_idx]),
            'specificity': float(1 - fpr[best_idx]),
            'j_score': float(j_scores[best_idx])
        }
    except:
        results['youden_j'] = {'threshold': 0.5, 'sensitivity': 0.0, 'specificity': 0.0, 'j_score': 0.0}
    
    # F1-optimal threshold
    try:
        precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_true, y_prob)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1])
            f1_scores = np.nan_to_num(f1_scores)
        best_idx = np.argmax(f1_scores)
        results['f1_optimal'] = {
            'threshold': float(thresholds_pr[best_idx]),
            'precision': float(precision_arr[best_idx]),
            'recall': float(recall_arr[best_idx]),
            'f1_score': float(f1_scores[best_idx])
        }
    except:
        results['f1_optimal'] = {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    # Balanced accuracy optimal
    try:
        thresholds = np.linspace(0.01, 0.99, 99)
        best_ba = 0
        best_thresh = 0.5
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                ba = (sens + spec) / 2
                if ba > best_ba:
                    best_ba = ba
                    best_thresh = thresh
        results['balanced_accuracy_optimal'] = {
            'threshold': float(best_thresh),
            'balanced_accuracy': float(best_ba)
        }
    except:
        results['balanced_accuracy_optimal'] = {'threshold': 0.5, 'balanced_accuracy': 0.0}
    
    return results


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics (ECE, MCE) and binned calibration data.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        
        bin_count = np.sum(mask)
        
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            bin_error = abs(bin_accuracy - bin_confidence)
            
            ece += (bin_count / total_samples) * bin_error
            mce = max(mce, bin_error)
        else:
            bin_accuracy = 0.0
            bin_confidence = 0.0
            bin_error = 0.0
        
        bin_data.append({
            'bin_lower': float(lower),
            'bin_upper': float(upper),
            'bin_center': float((lower + upper) / 2),
            'count': int(bin_count),
            'accuracy': float(bin_accuracy),
            'confidence': float(bin_confidence),
            'error': float(bin_error)
        })
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'n_bins': n_bins,
        'bins': bin_data
    }


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive regression metrics.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred
    
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['r2'] = float(r2_score(y_true, y_pred))
    metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
    metrics['max_error'] = float(max_error(y_true, y_pred))
    metrics['median_absolute_error'] = float(median_absolute_error(y_true, y_pred))
    
    # Percentage errors
    nonzero_mask = np.abs(y_true) > 1e-10
    if np.sum(nonzero_mask) > 0:
        ape = np.abs(residuals[nonzero_mask] / y_true[nonzero_mask]) * 100
        metrics['mape'] = float(np.mean(ape))
        metrics['median_ape'] = float(np.median(ape))
    else:
        metrics['mape'] = float('nan')
        metrics['median_ape'] = float('nan')
    
    # Symmetric MAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    nonzero_denom = denominator > 1e-10
    if np.sum(nonzero_denom) > 0:
        sape = np.abs(residuals[nonzero_denom]) / denominator[nonzero_denom] * 100
        metrics['smape'] = float(np.mean(sape))
    else:
        metrics['smape'] = float('nan')
    
    # Residual statistics
    metrics['residual_mean'] = float(np.mean(residuals))
    metrics['residual_std'] = float(np.std(residuals))
    metrics['residual_skewness'] = float(stats.skew(residuals))
    metrics['residual_kurtosis'] = float(stats.kurtosis(residuals))
    
    # Error percentiles
    abs_errors = np.abs(residuals)
    metrics['error_percentile_25'] = float(np.percentile(abs_errors, 25))
    metrics['error_percentile_50'] = float(np.percentile(abs_errors, 50))
    metrics['error_percentile_75'] = float(np.percentile(abs_errors, 75))
    metrics['error_percentile_90'] = float(np.percentile(abs_errors, 90))
    metrics['error_percentile_95'] = float(np.percentile(abs_errors, 95))
    metrics['error_percentile_99'] = float(np.percentile(abs_errors, 99))
    
    # Correlation metrics
    if len(y_true) > 2:
        metrics['pearson_correlation'] = float(np.corrcoef(y_true, y_pred)[0, 1])
        metrics['spearman_correlation'] = float(stats.spearmanr(y_true, y_pred)[0])
    else:
        metrics['pearson_correlation'] = float('nan')
        metrics['spearman_correlation'] = float('nan')
    
    # Sample statistics
    metrics['n_samples'] = len(y_true)
    metrics['y_true_mean'] = float(np.mean(y_true))
    metrics['y_true_std'] = float(np.std(y_true))
    metrics['y_pred_mean'] = float(np.mean(y_pred))
    metrics['y_pred_std'] = float(np.std(y_pred))
    
    return metrics


def compute_binned_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5
) -> Dict[str, Any]:
    """
    Compute regression metrics binned by target value ranges.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    
    try:
        bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([y_true.min(), y_true.max()])
    except:
        bin_edges = np.array([y_true.min(), y_true.max()])
    
    bin_data = []
    
    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        
        if i == len(bin_edges) - 2:
            mask = (y_true >= lower) & (y_true <= upper)
        else:
            mask = (y_true >= lower) & (y_true < upper)
        
        bin_count = np.sum(mask)
        
        if bin_count > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            bin_rmse = float(np.sqrt(mean_squared_error(bin_true, bin_pred)))
            bin_mae = float(mean_absolute_error(bin_true, bin_pred))
            bin_r2 = float(r2_score(bin_true, bin_pred)) if bin_count > 1 else 0.0
        else:
            bin_rmse = 0.0
            bin_mae = 0.0
            bin_r2 = 0.0
        
        bin_data.append({
            'bin_lower': float(lower),
            'bin_upper': float(upper),
            'bin_center': float((lower + upper) / 2),
            'count': int(bin_count),
            'rmse': bin_rmse,
            'mae': bin_mae,
            'r2': bin_r2
        })
    
    return {
        'n_bins': len(bin_data),
        'bins': bin_data
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for classification metrics.
    """
    np.random.seed(random_state)
    
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    n_samples = len(y_true)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    metric_funcs = {
        'accuracy': lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
        'roc_auc': lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
        'f1_score': lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0),
        'precision': lambda yt, yp: precision_score(yt, (yp >= 0.5).astype(int), zero_division=0),
        'recall': lambda yt, yp: recall_score(yt, (yp >= 0.5).astype(int), zero_division=0),
        'brier_score': lambda yt, yp: brier_score_loss(yt, yp),
    }
    
    bootstrap_results = {name: [] for name in metric_funcs}
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        for name, func in metric_funcs.items():
            try:
                value = func(y_true_boot, y_prob_boot)
                if not np.isnan(value):
                    bootstrap_results[name].append(value)
            except:
                pass
    
    results = {}
    point_estimates = classification_metrics(y_true, y_prob)
    
    for name in metric_funcs:
        values = bootstrap_results[name]
        if len(values) > 0:
            results[name] = {
                'point_estimate': point_estimates.get(name, float(np.mean(values))),
                'ci_lower': float(np.percentile(values, lower_percentile)),
                'ci_upper': float(np.percentile(values, upper_percentile)),
                'std_error': float(np.std(values))
            }
        else:
            results[name] = {
                'point_estimate': point_estimates.get(name, np.nan),
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'std_error': np.nan
            }
    
    return results


def bootstrap_regression_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for regression metrics.
    """
    np.random.seed(random_state)
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n_samples = len(y_true)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    metric_funcs = {
        'rmse': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        'mae': lambda yt, yp: mean_absolute_error(yt, yp),
        'r2': lambda yt, yp: r2_score(yt, yp),
        'explained_variance': lambda yt, yp: explained_variance_score(yt, yp),
    }
    
    bootstrap_results = {name: [] for name in metric_funcs}
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        for name, func in metric_funcs.items():
            try:
                value = func(y_true_boot, y_pred_boot)
                if not np.isnan(value):
                    bootstrap_results[name].append(value)
            except:
                pass
    
    results = {}
    point_estimates = regression_metrics(y_true, y_pred)
    
    for name in metric_funcs:
        values = bootstrap_results[name]
        if len(values) > 0:
            results[name] = {
                'point_estimate': point_estimates.get(name, float(np.mean(values))),
                'ci_lower': float(np.percentile(values, lower_percentile)),
                'ci_upper': float(np.percentile(values, upper_percentile)),
                'std_error': float(np.std(values))
            }
        else:
            results[name] = {
                'point_estimate': point_estimates.get(name, np.nan),
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'std_error': np.nan
            }
    
    return results
