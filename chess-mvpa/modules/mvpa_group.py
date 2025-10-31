"""
Shared MVPA group statistics routines.

Provides focused functions for common MVPA group-level comparisons:
  - Between-group tests (expert vs novice) with descriptive statistics
  - Within-group tests (vs chance level) with descriptive statistics
  - Data splitting helpers to avoid duplication

Each function has a clear, single responsibility. Analysis scripts should
explicitly show the workflow steps (loop over targets, perform tests, save results)
rather than hiding complexity in orchestration functions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from common.stats_utils import (
    per_roi_welch_and_fdr,
    per_roi_one_sample_vs_value,
)
from common.group_stats import get_descriptives_per_roi


def _compute_descriptives_per_roi(
    data: np.ndarray,
    confidence_level: float = 0.95,
) -> List[Tuple[float, float, float]]:
    """Compat shim: delegates to common.group_stats.get_descriptives_per_roi."""
    return get_descriptives_per_roi(data, confidence_level=confidence_level)


def compute_per_roi_group_comparison(
    expert_data: np.ndarray,
    novice_data: np.ndarray,
    roi_names: List[str],
    alpha: float,
    confidence_level: float = 0.95,
) -> Dict[str, object]:
    """
    Compare experts vs novices per ROI with Welch's t-test (FDR corrected) and descriptives.

    Performs Welch's two-sample t-test (does not assume equal variances) comparing
    expert and novice groups for each ROI. FDR correction is applied across all ROIs.
    Also computes descriptive statistics (mean and confidence intervals) for both
    groups, which describe the comparison being tested.

    Parameters
    ----------
    expert_data : ndarray, shape (n_experts, n_rois)
        Expert group data; each row is a subject, each column is an ROI
    novice_data : ndarray, shape (n_novices, n_rois)
        Novice group data; same structure as expert_data
    roi_names : list of str
        ROI labels (must match number of columns in data arrays)
    alpha : float
        FDR alpha threshold (e.g., 0.05)
    confidence_level : float, default=0.95
        Confidence level for descriptive intervals (e.g., 0.95 for 95% CI)

    Returns
    -------
    dict
        Dictionary with keys:
          - 'test_results': DataFrame with columns ROI_Name, ROI_id, t_stat,
            p_val, p_fdr, reject_fdr (one row per ROI)
          - 'expert_desc': list of (mean, ci_low, ci_high) tuples per ROI
          - 'novice_desc': list of (mean, ci_low, ci_high) tuples per ROI
    """
    # Perform Welch's t-test with FDR correction
    roi_ids = np.arange(1, len(roi_names) + 1)
    welch_results = per_roi_welch_and_fdr(expert_data, novice_data, roi_ids, alpha=alpha)
    welch_results.insert(0, "ROI_Name", roi_names)

    # Compute descriptive statistics for both groups
    expert_desc = _compute_descriptives_per_roi(expert_data, confidence_level)
    novice_desc = _compute_descriptives_per_roi(novice_data, confidence_level)

    return {
        'test_results': welch_results,
        'expert_desc': expert_desc,
        'novice_desc': novice_desc,
    }


def compute_per_roi_vs_chance_tests(
    expert_data: np.ndarray,
    novice_data: np.ndarray,
    roi_names: List[str],
    chance_level: float,
    alpha: float,
    alternative: str = 'greater',
    confidence_level: float = 0.95,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Test each group vs chance per ROI with one-sample t-tests (FDR corrected) and descriptives.

    Performs one-sample t-tests to assess whether each group's mean significantly
    differs from a specified chance level. Common use cases:
      - SVM decoding: test if accuracy > 0.5 (binary) or > 1/n_classes (multiclass)
      - RSA correlations: test if correlation > 0 (no relationship)

    FDR correction is applied separately within each group across all ROIs.
    Also computes descriptive statistics (mean and confidence intervals) for each
    group, which describe the values being tested against chance.

    Parameters
    ----------
    expert_data : ndarray, shape (n_experts, n_rois)
        Expert group data
    novice_data : ndarray, shape (n_novices, n_rois)
        Novice group data
    roi_names : list of str
        ROI labels
    chance_level : float
        Null hypothesis value (e.g., 0.5 for binary SVM, 0.0 for RSA)
    alpha : float
        FDR alpha threshold
    alternative : {'greater', 'two-sided', 'less'}, default='greater'
        Alternative hypothesis direction:
          - 'greater': mean > chance (most common for decoding/RSA)
          - 'two-sided': mean ≠ chance
          - 'less': mean < chance
    confidence_level : float, default=0.95
        Confidence level for descriptive intervals (e.g., 0.95 for 95% CI)

    Returns
    -------
    tuple of dicts
        (experts_result, novices_result)
        Each dict contains:
          - 'test_results': DataFrame with columns ROI_Name, t_stat, p_val,
            p_fdr, reject_fdr (one row per ROI)
          - 'desc': list of (mean, ci_low, ci_high) tuples per ROI
    """
    # Perform one-sample t-tests with FDR correction
    experts_vs_chance = per_roi_one_sample_vs_value(
        expert_data, roi_names, chance_level, alpha=alpha, alternative=alternative
    )
    novices_vs_chance = per_roi_one_sample_vs_value(
        novice_data, roi_names, chance_level, alpha=alpha, alternative=alternative
    )

    # Compute descriptive statistics for both groups
    expert_desc = _compute_descriptives_per_roi(expert_data, confidence_level)
    novice_desc = _compute_descriptives_per_roi(novice_data, confidence_level)

    return (
        {'test_results': experts_vs_chance, 'desc': expert_desc},
        {'test_results': novices_vs_chance, 'desc': novice_desc},
    )


def split_data_by_target_and_group(
    df: pd.DataFrame,
    target: str,
    roi_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract expert and novice ROI data for a specific target (DRY helper).

    This helper eliminates duplicated data-splitting logic. It filters the
    dataframe for a specific target, separates experts from novices, and
    returns their ROI data as numpy arrays ready for statistical tests.

    Parameters
    ----------
    df : pd.DataFrame
        Group dataframe with columns: subject, expert (bool), target, and ROI columns
    target : str
        Target name to filter (e.g., 'checkmate', 'categories')
    roi_names : list of str
        ROI column names to extract

    Returns
    -------
    tuple of ndarray
        (expert_data, novice_data)
        Each array has shape (n_subjects, n_rois)
    """
    df_tgt = df[df['target'] == target].copy()
    expert_data = df_tgt[df_tgt['expert'] == True][roi_names].values
    novice_data = df_tgt[df_tgt['expert'] == False][roi_names].values
    return expert_data, novice_data


__all__ = [
    'compute_per_roi_group_comparison',
    'compute_per_roi_vs_chance_tests',
    'split_data_by_target_and_group',
]
