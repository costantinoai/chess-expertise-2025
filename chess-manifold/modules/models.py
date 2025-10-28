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
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    pr_with_group = pr_df.merge(
        participants[['participant_id', 'group']],
        left_on='subject_id',
        right_on='participant_id',
        how='left'
    )

    # Convert to wide format (subjects × ROIs)
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
