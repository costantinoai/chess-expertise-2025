"""
Participation Ratio (PR) computation for manifold dimensionality analysis.

The Participation Ratio measures the effective dimensionality of neural population
activity by quantifying how evenly variance is distributed across principal components.

    PR = (sum(λ))² / sum(λ²)

where λ are the eigenvalues (explained variance) from PCA.

Properties:
- PR = 1: All variance in one dimension (minimum dimensionality)
- PR = N: Variance evenly distributed across N dimensions (maximum dimensionality)
- Higher PR = more distributed, higher-dimensional population activity
"""

import logging
from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def participation_ratio(roi_matrix: np.ndarray) -> float:
    """
    Compute Participation Ratio from PCA spectrum of neural activity matrix (Conditions x Voxels).

    Parameters
    ----------
    roi_matrix : np.ndarray
        Neural activity matrix of shape (n_conditions, n_voxels).
        Rows are experimental conditions, columns are voxels/neurons.

    Returns
    -------
    float
        Participation ratio value. Returns np.nan if matrix is degenerate
        (empty, all NaN columns, or insufficient data for PCA).

    Notes
    -----
    The function handles degenerate cases gracefully:
    - Empty matrices return NaN
    - Columns with all NaN values are removed before PCA
    - Matrices with <2 valid columns return NaN

    Examples
    --------
    >>> # High-dimensional activity (variance spread across components)
    >>> activity = np.random.randn(10, 100)  # 10 conditions, 100 voxels
    >>> pr = participation_ratio(activity)
    >>> print(f"PR = {pr:.2f}")  # Should be close to number of conditions

    >>> # Low-dimensional activity (variance in few components)
    >>> low_dim = np.outer(np.arange(10), np.random.randn(100))
    >>> pr_low = participation_ratio(low_dim)
    >>> print(f"Low-dim PR = {pr_low:.2f}")  # Should be close to 1
    """
    if roi_matrix.size == 0:
        logger.debug("Empty matrix provided to participation_ratio")
        return float("nan")

    # Remove columns that are all NaN (invalid voxels)
    valid_voxels = ~np.isnan(roi_matrix).all(axis=0)
    cleaned_matrix = roi_matrix[:, valid_voxels]

    if cleaned_matrix.size == 0:
        logger.debug("No valid voxels after NaN removal")
        return float("nan")

    if cleaned_matrix.shape[1] < 2:
        logger.debug(f"Insufficient voxels for PCA: {cleaned_matrix.shape[1]}")
        return float("nan")

    try:
        # Perform PCA on the neural activity matrix (conditions × voxels).
        # PCA decomposes the covariance structure into orthogonal components,
        # ordered by variance explained. Eigenvalues (explained_variance_) quantify
        # how much variance each component captures.
        pca = PCA()
        pca.fit(cleaned_matrix)
        eigenvalues = pca.explained_variance_

        if eigenvalues.size == 0 or np.sum(eigenvalues) == 0:
            logger.debug("PCA returned zero or empty eigenvalues")
            return float("nan")

        # Compute participation ratio using the formula: PR = (Σλ)² / Σ(λ²)
        # This formula measures effective dimensionality:
        # - If all variance is in one dimension: λ = [V, 0, 0, ...] → PR = V²/V² = 1
        # - If variance evenly distributed: λ = [V/n, V/n, ...] → PR = (V)²/(n*V²/n²) = n
        # Essentially, PR quantifies how many dimensions are "actively participating"
        # in representing the data, accounting for variance distribution evenness.
        sum_eigenvals = np.sum(eigenvalues)
        sum_squared_eigenvals = np.sum(eigenvalues ** 2)

        if sum_squared_eigenvals == 0:
            logger.debug("Sum of squared eigenvalues is zero")
            return float("nan")

        pr = (sum_eigenvals ** 2) / sum_squared_eigenvals
        return float(pr)

    except Exception as e:
        logger.warning(f"PCA computation failed: {e}")
        return float("nan")


def compute_subject_roi_prs(
    subject_id: str,
    atlas_data: np.ndarray,
    roi_labels: np.ndarray,
    base_path: Path,
    spm_filename: str = "SPM.mat"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute participation ratios for all ROIs in a single subject.

    This function orchestrates the complete pipeline for one subject:
    1. Load SPM beta coefficients for all experimental conditions
    2. Extract voxel-wise activity matrices for each ROI
    3. Compute participation ratio for each ROI
    4. Return results with quality control metadata

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '03'). Used to construct data paths.
    atlas_data : np.ndarray
        3D atlas volume with integer ROI labels.
    roi_labels : np.ndarray
        1D array of ROI labels to process.
    base_path : Path
        Base path to SPM GLM results directory.
    spm_filename : str, optional
        Name of SPM.mat file (default: 'SPM.mat').

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - pr_values: PR values per ROI (length = n_rois). NaN for failed computations.
        - voxel_counts: Number of valid voxels per ROI (length = n_rois).

    Raises
    ------
    FileNotFoundError
        If SPM data cannot be found for the subject.
    ValueError
        If atlas or data are malformed.

    Notes
    -----
    The function is designed to be robust to individual ROI failures - if
    PR computation fails for one ROI, others will still be processed.
    Check the returned arrays for NaN values to identify failed ROIs.

    Examples
    --------
    >>> from common.neuro_utils import load_atlas
    >>> atlas, rois = load_atlas("atlas.nii")
    >>> pr_vals, vox_counts = compute_subject_roi_prs(
    ...     "03", atlas, rois, Path("/data/glm")
    ... )
    >>> print(f"Subject 03: {np.sum(~np.isnan(pr_vals))} valid ROIs")
    """
    # Import here to avoid circular imports
    from common.spm_utils import extract_roi_voxel_matrices

    logger.info(f"[Subject {subject_id}] Computing participation ratios for {len(roi_labels)} ROIs")

    try:
        # Extract voxel-wise activity matrices for all ROIs
        roi_matrices = extract_roi_voxel_matrices(
            subject_id, atlas_data, roi_labels, base_path
        )
    except Exception as e:
        logger.error(f"[Subject {subject_id}] Failed to extract ROI matrices: {e}")
        # Return arrays filled with NaN to indicate complete failure
        return (np.full(len(roi_labels), np.nan, dtype=np.float32),
                np.zeros(len(roi_labels), dtype=int))

    # Initialize output arrays
    pr_values = np.full(len(roi_labels), np.nan, dtype=np.float32)
    voxel_counts = np.zeros(len(roi_labels), dtype=int)

    # Compute PR for each ROI individually (robust to individual failures)
    successful_rois = 0
    for idx, roi_label in enumerate(roi_labels):
        # Let any exception propagate — caller should handle failures explicitly.
        roi_matrix = roi_matrices[int(roi_label)]
        pr_val = participation_ratio(roi_matrix)
        pr_values[idx] = pr_val
        voxel_counts[idx] = roi_matrix.shape[1]

        if np.isnan(pr_val):
            logger.error(f"[Subject {subject_id}] ROI {roi_label}: PR computation returned NaN")

    logger.info(f"[Subject {subject_id}] PR computation completed: "
               f"{successful_rois}/{len(roi_labels)} ROIs successful")

    return pr_values, voxel_counts
