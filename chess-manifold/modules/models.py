"""
Machine learning utilities for expert vs novice classification.

This module provides functions for training classifiers to distinguish between
expert and novice chess players based on their neural participation ratios,
and for visualizing the results using PCA.

Functions
---------
train_logreg_on_pr : Train logistic regression on PR features
compute_pca_2d : Compute 2D PCA projection for visualization
compute_2d_decision_boundary : Precompute decision boundary grid for plotting
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, permutation_test_score
from scipy.stats import binomtest

logger = logging.getLogger(__name__)


def train_logreg_on_pr(
    pr_df: pd.DataFrame,
    participants: pd.DataFrame,
    roi_labels: np.ndarray,
    random_seed: int = 42
) -> Tuple[LogisticRegression, StandardScaler, np.ndarray, np.ndarray]:
    """
    Train a logistic regression classifier to distinguish experts from novices.

    The classifier uses PR values across all ROIs as features. Feature weights
    indicate which ROIs are most discriminative between groups.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR results (columns: subject_id, ROI_Label, PR, n_voxels)
    participants : pd.DataFrame
        Participant metadata (columns: participant_id, group)
    roi_labels : np.ndarray
        ROI labels defining feature order
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    clf : LogisticRegression
        Trained classifier (clf.coef_ contains feature importance weights)
    scaler : StandardScaler
        Fitted scaler for transforming new data
    all_pr_scaled : np.ndarray
        Standardized PR data (shape: n_subjects × n_rois)
    labels : np.ndarray
        Binary labels (1=expert, 0=novice)

    Notes
    -----
    Data is standardized before training because different ROIs have different
    PR scales. Without standardization, high-variance ROIs would dominate.

    The classifier is trained on ALL data (no cross-validation) because the goal
    is feature importance analysis, not generalization performance.
    """
    logger.info("Training logistic regression on PR features (expert vs novice)...")

    # Merge PR data with group labels
    from common.bids_utils import merge_group_labels
    from .utils import ensure_roi_order
    pr_with_group = merge_group_labels(pr_df, participants, subject_col='subject_id')

    # Convert to wide format (subjects × ROIs)
    expert_pr = pr_with_group[pr_with_group['group'] == 'expert'].pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )
    expert_pr = ensure_roi_order(expert_pr, roi_labels)[roi_labels].values

    novice_pr = pr_with_group[pr_with_group['group'] == 'novice'].pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )
    novice_pr = ensure_roi_order(novice_pr, roi_labels)[roi_labels].values

    # Stack data: experts first, then novices
    all_pr = np.vstack([expert_pr, novice_pr])
    labels = np.array([1] * len(expert_pr) + [0] * len(novice_pr))

    # Standardize features
    scaler = StandardScaler()
    all_pr_scaled = scaler.fit_transform(all_pr)

    # Train classifier
    clf = LogisticRegression(random_state=random_seed, max_iter=1000)
    clf.fit(all_pr_scaled, labels)

    logger.info(f"  Trained on {len(all_pr)} subjects "
                f"({len(expert_pr)} experts, {len(novice_pr)} novices)")
    logger.info(f"  Using {len(roi_labels)} ROI features")

    return clf, scaler, all_pr_scaled, labels


def compute_pca_2d(
    data_scaled: np.ndarray,
    n_components: int = 2,
    random_seed: int = 42
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Compute PCA embedding for 2D visualization.

    PCA finds directions of maximum variance and projects the high-dimensional
    PR data into 2D for visualization.

    Parameters
    ----------
    data_scaled : np.ndarray
        Standardized data (shape: n_subjects × n_features)
    n_components : int, default=2
        Number of components (2 for 2D visualization)
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pca : PCA
        Fitted PCA object (pca.components_ shows ROI loadings on each PC)
    coords_2d : np.ndarray
        2D coordinates (shape: n_subjects × 2)
    explained_variance_pct : np.ndarray
        Percentage of variance explained by each PC

    Notes
    -----
    PC1 captures the direction of maximum variance, PC2 captures the second-
    maximum variance orthogonal to PC1. Together they often capture major
    group differences.
    """
    logger.info("Computing 2D PCA embedding...")

    # Fit PCA and transform data
    pca = PCA(n_components=n_components, random_state=random_seed)
    coords_2d = pca.fit_transform(data_scaled)

    # Get variance explained
    explained_variance_pct = (pca.explained_variance_ratio_ * 100).astype(float)

    logger.info(f"  PC1 explains {explained_variance_pct[0]:.1f}% variance")
    logger.info(f"  PC2 explains {explained_variance_pct[1]:.1f}% variance")

    return pca, coords_2d, explained_variance_pct


def compute_2d_decision_boundary(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    random_seed: int = 42,
    grid_resolution: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute decision boundary grid for 2D visualization.

    Trains a simple classifier in 2D space and evaluates it on a dense grid,
    creating a smooth decision boundary for plotting.

    Parameters
    ----------
    coords_2d : np.ndarray
        2D coordinates from PCA (shape: n_subjects × 2)
    labels : np.ndarray
        Binary labels (1=expert, 0=novice)
    random_seed : int, default=42
        Random seed for reproducibility
    grid_resolution : int, default=200
        Grid points per axis (higher = smoother boundary)

    Returns
    -------
    xx : np.ndarray
        X-coordinates of grid (shape: grid_resolution × grid_resolution)
    yy : np.ndarray
        Y-coordinates of grid (shape: grid_resolution × grid_resolution)
    Z : np.ndarray
        Predicted class at each grid point (for contour plotting)

    Notes
    -----
    The boundary shows how well groups separate in the 2D PCA space. It's for
    visualization only - actual group discrimination happens in full-dimensional
    space (see train_logreg_on_pr).
    """
    logger.info("Computing 2D decision boundary for visualization...")

    # Train simple classifier in 2D
    clf_2d = LogisticRegression(random_state=random_seed, max_iter=1000)
    clf_2d.fit(coords_2d, labels)

    # Create grid with 1-unit margins
    x_min, x_max = coords_2d[:, 0].min() - 1, coords_2d[:, 0].max() + 1
    y_min, y_max = coords_2d[:, 1].min() - 1, coords_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )

    # Predict at each grid point
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    logger.info(f"  Computed {grid_resolution}×{grid_resolution} grid")

    return xx, yy, Z


__all__ = [
    'train_logreg_on_pr',
    'compute_pca_2d',
    'compute_2d_decision_boundary',
]


def _build_feature_matrix(
    pr_df: pd.DataFrame,
    participants: pd.DataFrame,
    roi_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Construct X (subjects × ROIs) and y (1=expert, 0=novice) from long PR data.

    Returns
    -------
    X : np.ndarray
        Feature matrix ordered by [experts, novices]
    y : np.ndarray
        Binary labels (1=expert, 0=novice)
    n_expert : int
        Number of expert subjects
    n_novice : int
        Number of novice subjects
    """
    pr_with_group = pr_df.merge(
        participants[['participant_id', 'group']],
        left_on='subject_id',
        right_on='participant_id',
        how='left'
    )

    expert_pr = pr_with_group[pr_with_group['group'] == 'expert'].pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )[roi_labels].values

    novice_pr = pr_with_group[pr_with_group['group'] == 'novice'].pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )[roi_labels].values

    X = np.vstack([expert_pr, novice_pr])
    y = np.array([1] * len(expert_pr) + [0] * len(novice_pr))
    return X, y, len(expert_pr), len(novice_pr)


def evaluate_classification_significance(
    pr_df: pd.DataFrame,
    participants: pd.DataFrame,
    roi_labels: np.ndarray,
    space: str = 'roi',
    random_seed: int = 42,
    n_splits: int = None,
    n_permutations: int = 1000,
) -> Dict[str, Any]:
    """
    Test whether classification accuracy is above chance (50%).

    Uses stratified K-fold CV to estimate accuracy and two significance tests:
    - Permutation test (label shuffling within CV)
    - Binomial test on total correct predictions across CV folds

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR results.
    participants : pd.DataFrame
        Participant metadata with group labels.
    roi_labels : np.ndarray
        ROI labels defining feature order.
    space : {'roi', 'pca2d'}, default='roi'
        Feature space to evaluate. 'roi' uses all ROIs as features; 'pca2d'
        uses a pipeline with PCA(2) fit within each CV fold.
    random_seed : int, default=42
        Random seed for reproducibility.
    n_splits : int or None, default=None
        Number of CV folds (StratifiedKFold). If None, chooses the maximum
        feasible up to 5 given class counts.
    n_permutations : int, default=1000
        Number of permutations for permutation test.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'space': 'roi' or 'pca2d'
        - 'cv_accuracy_mean', 'cv_accuracy_std'
        - 'n_splits', 'n_subjects', 'n_experts', 'n_novices'
        - 'perm_pvalue', 'perm_null_mean', 'perm_null_std', 'n_permutations'
        - 'binom_pvalue', 'n_correct', 'n_trials'
    """
    from sklearn.pipeline import Pipeline

    logger.info(f"Evaluating classification significance in '{space}' space...")

    X, y, n_expert, n_novice = _build_feature_matrix(pr_df, participants, roi_labels)
    n_subjects = X.shape[0]

    # Determine feasible number of splits
    if n_splits is None:
        max_splits = max(2, min(5, n_expert, n_novice))
        n_splits = max_splits
    if n_splits < 2 or n_splits > min(n_expert, n_novice):
        n_splits = min(max(2, n_splits), n_expert, n_novice)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    steps = [('scaler', StandardScaler())]
    if space.lower() in ['pca2d', 'pca_2d', '2d', 'pca']:
        steps.append(('pca', PCA(n_components=2, random_state=random_seed)))
    steps.append(('clf', LogisticRegression(random_state=random_seed, max_iter=1000)))
    estimator = Pipeline(steps)

    # Cross-validated accuracy (mean ± std across folds)
    cv_scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy')
    cv_acc_mean = float(np.mean(cv_scores))
    cv_acc_std = float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0

    # Predictions for binomial test
    y_pred = cross_val_predict(estimator, X, y, cv=cv, method='predict')
    n_correct = int((y_pred == y).sum())
    n_trials = int(len(y))
    binom_p = float(binomtest(n_correct, n_trials, p=0.5, alternative='greater').pvalue)

    # Permutation test (scikit-learn handles CV internally)
    score, perm_scores, pvalue = permutation_test_score(
        estimator, X, y,
        scoring='accuracy',
        cv=cv,
        n_permutations=n_permutations,
        random_state=random_seed,
        n_jobs=None,
    )

    results = {
        'space': 'pca2d' if ('pca' in [name for name, _ in steps]) else 'roi',
        'cv_accuracy_mean': cv_acc_mean,
        'cv_accuracy_std': cv_acc_std,
        'n_splits': int(n_splits),
        'n_subjects': int(n_subjects),
        'n_experts': int(n_expert),
        'n_novices': int(n_novice),
        'perm_pvalue': float(pvalue),
        'perm_null_mean': float(np.mean(perm_scores)),
        'perm_null_std': float(np.std(perm_scores, ddof=1)) if len(perm_scores) > 1 else 0.0,
        'n_permutations': int(n_permutations),
        'binom_pvalue': binom_p,
        'n_correct': n_correct,
        'n_trials': n_trials,
    }

    logger.info(
        f"  CV accuracy: {cv_acc_mean:.3f} ± {cv_acc_std:.3f} (n_splits={n_splits})\n"
        f"  Permutation p={pvalue:.4g} (null mean={np.mean(perm_scores):.3f})\n"
        f"  Binomial p={binom_p:.4g} (n_correct={n_correct}/{n_trials})"
    )

    return results


__all__.extend(['evaluate_classification_significance'])
