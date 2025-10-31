"""
Split-half reliability utilities for behavioral RDM analysis.

This module provides focused functions for split-half reliability analysis.
It reuses existing RDM computation and correlation functions from
chess-behavioral and common modules - no duplication, no thin wrappers.

Functions
---------
spearman_brown_correction : Apply Spearman-Brown prophecy formula
bootstrap_split_half_reliability : Bootstrap random splits for CI estimation

Notes
-----
This module imports dependencies inside functions rather than at module level
to avoid sys.path issues when imported from different contexts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def spearman_brown_correction(r_half: float) -> float:
    """
    Apply Spearman-Brown prophecy formula for split-half reliability.

    The Spearman-Brown formula estimates full-sample reliability from
    the correlation between two half-samples:

        r_full = (2 × r_half) / (1 + r_half)

    This adjusts for the reduction in measurement precision when using
    only half the sample (Spearman, 1910; Brown, 1910).

    Parameters
    ----------
    r_half : float
        Correlation between two half-sample measurements

    Returns
    -------
    r_full : float
        Estimated full-sample reliability

    Notes
    -----
    - Formula assumes equal-length halves and equal measurement precision
    - Valid for r_half in (-1, 1); returns NaN if r_half = -1 (denominator zero)
    - Typically increases reliability estimate (r_full > r_half) for positive r_half

    References
    ----------
    Brown, W. (1910). Some experimental results in the correlation of mental
      abilities. British Journal of Psychology, 3, 296-322.
    Spearman, C. (1910). Correlation calculated from faulty data. British
      Journal of Psychology, 3, 271-295.

    Examples
    --------
    >>> r_half = 0.80
    >>> r_full = spearman_brown_correction(r_half)
    >>> print(f"Half-sample r = {r_half:.2f}, Full-sample r = {r_full:.2f}")
    Half-sample r = 0.80, Full-sample r = 0.89
    """
    if r_half == -1:
        return np.nan

    r_full = (2.0 * r_half) / (1.0 + r_half)
    return r_full


def bootstrap_split_half_reliability(
    pairwise_data: pd.DataFrame,
    participant_ids: List[str],
    n_iterations: int = 10000,
    method: str = 'spearman',
    random_state: int = 42,
    participant_col: str = 'sub_id'
) -> Dict:
    """
    Bootstrap split-half reliability across multiple random splits.

    For each iteration:
    1. Randomly split participants in half
    2. Aggregate pairwise counts for each half (reuse chess-behavioral)
    3. Compute RDM for each half (reuse chess-behavioral)
    4. Correlate the two RDMs (reuse common)
    5. Apply Spearman-Brown correction

    This provides distribution-based confidence intervals for reliability
    estimates, accounting for uncertainty in the specific split.

    Parameters
    ----------
    pairwise_data : pd.DataFrame
        Pairwise preference data with columns: better, worse, sub_id
    participant_ids : list of str
        List of all participant IDs to split
    n_iterations : int, default=10000
        Number of random splits to perform
    method : str, default='spearman'
        Correlation method ('pearson' or 'spearman')
    random_state : int, default=42
        Random seed for reproducibility
    participant_col : str, default='sub_id'
        Column name containing participant IDs

    Returns
    -------
    dict with keys:
        'r_half' : np.ndarray, shape (n_iterations,)
            Half-sample correlations for each iteration
        'r_full' : np.ndarray, shape (n_iterations,)
            Spearman-Brown corrected reliabilities for each iteration
        'mean_r_half' : float
            Mean half-sample correlation across iterations
        'mean_r_full' : float
            Mean corrected reliability across iterations
        'ci_r_half' : tuple of float
            95% CI for half-sample correlation [lower, upper]
        'ci_r_full' : tuple of float
            95% CI for corrected reliability [lower, upper]

    Notes
    -----
    - Uses percentile method for CI (2.5th and 97.5th percentiles)
    - Each iteration uses a different random split
    - Results are reproducible given the same random_state
    - Inline implementation - no wrapper functions
    - Imports dependencies inside function to avoid sys.path issues
    """
    # Import dependencies inside function to avoid sys.path issues at module import time
    from modules.rdm_utils import (
        aggregate_pairwise_counts,
        compute_symmetric_rdm,
    )
    from common.rsa_utils import correlate_rdms

    n_participants = len(participant_ids)

    if n_participants % 2 != 0:
        raise ValueError(f"Cannot split {n_participants} participants evenly")

    half_size = n_participants // 2

    rng = np.random.RandomState(random_state)

    r_half_values = []
    r_full_values = []

    for i in range(n_iterations):
        # Random split - shuffle and take first/second halves
        shuffled_indices = rng.permutation(n_participants)
        half1_indices = shuffled_indices[:half_size]
        half2_indices = shuffled_indices[half_size:]

        half1_ids = [participant_ids[idx] for idx in half1_indices]
        half2_ids = [participant_ids[idx] for idx in half2_indices]

        # Filter pairwise data by participant IDs
        half1_data = pairwise_data[pairwise_data[participant_col].isin(half1_ids)]
        half2_data = pairwise_data[pairwise_data[participant_col].isin(half2_ids)]

        # Aggregate pairwise counts within each half (REUSE)
        # chess-behavioral.modules.rdm_utils.aggregate_pairwise_counts expects a list
        half1_agg = aggregate_pairwise_counts([half1_data])
        half2_agg = aggregate_pairwise_counts([half2_data])

        # Compute RDM for each half (REUSE)
        rdm_half1 = compute_symmetric_rdm(half1_agg)
        rdm_half2 = compute_symmetric_rdm(half2_agg)

        # Correlate the two RDMs (REUSE)
        r, p, ci_l, ci_u = correlate_rdms(rdm_half1, rdm_half2, method=method)

        # Apply Spearman-Brown correction
        r_corrected = spearman_brown_correction(r)

        r_half_values.append(r)
        r_full_values.append(r_corrected)

    # Convert to arrays
    r_half_arr = np.array(r_half_values)
    r_full_arr = np.array(r_full_values)

    # Compute summary statistics
    results = {
        'r_half': r_half_arr,
        'r_full': r_full_arr,
        'mean_r_half': float(np.mean(r_half_arr)),
        'mean_r_full': float(np.mean(r_full_arr)),
        'ci_r_half': (
            float(np.percentile(r_half_arr, 2.5)),
            float(np.percentile(r_half_arr, 97.5))
        ),
        'ci_r_full': (
            float(np.percentile(r_full_arr, 2.5)),
            float(np.percentile(r_full_arr, 97.5))
        ),
    }

    return results
