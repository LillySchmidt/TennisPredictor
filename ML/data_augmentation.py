"""Data augmentation module for generating additional training data.

This module provides intelligent data augmentation techniques:
- SMOTE (Synthetic Minority Over-sampling Technique) for class balance
- Feature perturbation with noise injection
- Match context variation (surface, round, tournament level)
- Rank-based synthetic match generation
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.utils import resample


def augment_with_noise(
    X: pd.DataFrame, 
    y: pd.Series, 
    noise_factor: float = 0.05, 
    n_augmented: int = 1000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate augmented samples by adding Gaussian noise to numerical features.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        noise_factor: Standard deviation of noise relative to feature std
        n_augmented: Number of augmented samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    rng = np.random.default_rng(random_state)
    
    # Identify numerical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    # Sample indices to augment
    sample_indices = rng.choice(len(X), size=n_augmented, replace=True)
    
    X_aug = X.iloc[sample_indices].copy()
    y_aug = y.iloc[sample_indices].copy()
    
    # Add noise to numerical features
    for col in num_cols:
        if X[col].std() > 0:  # Only add noise if there's variation
            noise = rng.normal(0, X[col].std() * noise_factor, size=n_augmented)
            X_aug[col] = X_aug[col] + noise
            
            # Ensure non-negative for rank columns
            if 'rank' in col.lower():
                X_aug[col] = np.maximum(X_aug[col], 1.0)
    
    return X_aug, y_aug


def augment_with_resampling(
    X: pd.DataFrame, 
    y: pd.Series, 
    target_size: int = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate augmented samples through strategic resampling.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        target_size: Desired total dataset size (None = 2x original)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    if target_size is None:
        target_size = len(X) * 2
    
    n_augmented = target_size - len(X)
    
    if n_augmented <= 0:
        return X.copy(), y.copy()
    
    # Resample with replacement
    X_resampled, y_resampled = resample(
        X, y, 
        n_samples=n_augmented, 
        random_state=random_state,
        replace=True,
        stratify=y
    )
    
    return X_resampled, y_resampled


def generate_synthetic_matches(
    X: pd.DataFrame,
    y: pd.Series,
    n_synthetic: int = 5000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic tennis matches based on rank distributions.
    
    Creates realistic matches by:
    - Sampling ranks from observed distributions
    - Assigning realistic match contexts (surface, round, etc.)
    - Computing outcome probabilities based on rank differences
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_synthetic: Number of synthetic matches to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (synthetic_X, synthetic_y)
    """
    rng = np.random.default_rng(random_state)
    
    synthetic_data = []
    
    # Get feature distributions
    surfaces = X['surface'].value_counts(normalize=True)
    rounds = X['round'].value_counts(normalize=True) if 'round' in X.columns else None
    levels = X['tourney_level'].value_counts(normalize=True) if 'tourney_level' in X.columns else None
    
    # Rank statistics
    rank_mean = X['playerA_rank'].mean()
    rank_std = X['playerA_rank'].std()
    
    for _ in range(n_synthetic):
        # Generate player ranks from truncated normal distribution
        playerA_rank = max(1.0, rng.normal(rank_mean, rank_std))
        playerB_rank = max(1.0, rng.normal(rank_mean, rank_std))
        
        # Sample categorical features
        surface = rng.choice(surfaces.index, p=surfaces.values)
        round_val = rng.choice(rounds.index, p=rounds.values) if rounds is not None else X['round'].mode()[0]
        level = rng.choice(levels.index, p=levels.values) if levels is not None else X['tourney_level'].mode()[0]
        
        # Determine outcome based on rank difference (better rank = lower number = higher win probability)
        rank_diff = playerA_rank - playerB_rank
        # Sigmoid-like probability: better player (lower rank) more likely to win
        win_prob = 1 / (1 + np.exp(rank_diff / 50))  # Normalized by typical rank difference
        label = 1 if rng.random() < win_prob else 0
        
        # Create synthetic match
        match = {
            'surface': surface,
            'round': round_val,
            'tourney_level': level,
            'best_of': X['best_of'].mode()[0] if 'best_of' in X.columns else 3,
            'draw_size': X['draw_size'].median() if 'draw_size' in X.columns else 64,
            'year': rng.choice(X['year'].unique()) if 'year' in X.columns else 2024,
            'playerA_rank': playerA_rank,
            'playerB_rank': playerB_rank,
            'rank_diff': playerA_rank - playerB_rank,
            'log_playerA_rank': np.log1p(playerA_rank),
            'log_playerB_rank': np.log1p(playerB_rank),
            'log_rank_diff': np.log1p(playerA_rank) - np.log1p(playerB_rank),
        }
        
        # Add round_code if it exists
        if 'round_code' in X.columns:
            from data_prep import ROUND_ORDER
            match['round_code'] = ROUND_ORDER.get(round_val, 4)  # Default to R16
        
        synthetic_data.append((match, label))
    
    # Convert to DataFrame and Series
    X_synthetic = pd.DataFrame([m[0] for m in synthetic_data])
    y_synthetic = pd.Series([m[1] for m in synthetic_data], dtype=int)
    
    # Ensure column order matches original
    X_synthetic = X_synthetic[X.columns]
    
    return X_synthetic, y_synthetic


def augment_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    augmentation_factor: float = 2.0,
    use_noise: bool = True,
    use_synthetic: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Comprehensive data augmentation combining multiple techniques.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        augmentation_factor: Target size as multiple of original (e.g., 2.0 = double)
        use_noise: Whether to include noise-based augmentation
        use_synthetic: Whether to include synthetic match generation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (augmented_X, augmented_y) including original data
    """
    target_size = int(len(X) * augmentation_factor)
    n_to_generate = target_size - len(X)
    
    if n_to_generate <= 0:
        return X.copy(), y.copy()
    
    augmented_X_parts = [X]
    augmented_y_parts = [y]
    
    # Split augmentation between techniques
    n_per_method = n_to_generate // (int(use_noise) + int(use_synthetic))
    
    if use_noise and n_per_method > 0:
        X_noise, y_noise = augment_with_noise(
            X, y, 
            noise_factor=0.03,
            n_augmented=n_per_method,
            random_state=random_state
        )
        augmented_X_parts.append(X_noise)
        augmented_y_parts.append(y_noise)
    
    if use_synthetic and n_per_method > 0:
        X_synthetic, y_synthetic = generate_synthetic_matches(
            X, y,
            n_synthetic=n_per_method,
            random_state=random_state + 1
        )
        augmented_X_parts.append(X_synthetic)
        augmented_y_parts.append(y_synthetic)
    
    # Combine all parts
    X_final = pd.concat(augmented_X_parts, ignore_index=True)
    y_final = pd.concat(augmented_y_parts, ignore_index=True)
    
    # Shuffle
    shuffle_idx = np.random.default_rng(random_state).permutation(len(X_final))
    X_final = X_final.iloc[shuffle_idx].reset_index(drop=True)
    y_final = y_final.iloc[shuffle_idx].reset_index(drop=True)
    
    return X_final, y_final


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    
    import data_prep as dp
    
    print("Loading original data...")
    X, y = dp.load_features_and_target()
    print(f"Original size: {len(X)} samples")
    
    print("\nAugmenting data...")
    X_aug, y_aug = augment_dataset(X, y, augmentation_factor=3.0)
    print(f"Augmented size: {len(X_aug)} samples")
    print(f"Class distribution: {y_aug.value_counts().to_dict()}")
