"""
Group-level descriptive statistics helpers shared across analyses.

Provides a minimal, reusable API to compute per-ROI (mean, CI) tuples for a
group matrix shaped (n_subjects, n_rois). Centralizing this logic avoids
duplication between MVPA and PR modules while preserving existing behavior.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .stats_utils import compute_group_mean_and_ci


def get_descriptives_per_roi(
    data: np.ndarray,
    confidence_level: float = 0.95,
) -> List[Tuple[float, float, float]]:
    """
    Compute mean and CI for each ROI column in a 2D array.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_rois)
        Group data per ROI.
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI).

    Returns
    -------
    list of tuple
        One (mean, ci_low, ci_high) tuple per ROI. If a column has fewer
        than 2 valid observations, returns (NaN, NaN, NaN) for that ROI.
    """
    descriptives: List[Tuple[float, float, float]] = []
    if data is None or data.size == 0:
        return descriptives

    for roi_idx in range(data.shape[1]):
        x = data[:, roi_idx]
        x = x[~np.isnan(x)]
        if x.size > 1:
            descriptives.append(
                compute_group_mean_and_ci(x, confidence_level=confidence_level)
            )
        else:
            descriptives.append((np.nan, np.nan, np.nan))

    return descriptives


__all__ = [
    'get_descriptives_per_roi',
]

