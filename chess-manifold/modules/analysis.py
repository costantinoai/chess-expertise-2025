"""
High-level analysis orchestration for manifold dimensionality analysis.

This module provides the main analysis functions for computing participation
ratios, summarizing results by group, and performing statistical comparisons
between experts and novices.

The participation ratio (PR) quantifies the effective dimensionality of neural
population activity. Higher PR values indicate that activity is spread across
more dimensions (more distributed), while lower PR values indicate activity
is concentrated in fewer dimensions (more specialized).

Functions
---------
summarize_pr_by_group : Compute descriptive statistics per group and ROI
compare_groups_welch_fdr : Perform expert vs novice comparisons with FDR correction
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


 


def summarize_pr_by_group(
    pr_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    roi_labels: np.ndarray,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, CI, SEM) per group and ROI.

    This function summarizes the PR data by computing the mean and variability
    for each ROI within each group (experts and novices). These statistics are
    used for plotting and reporting group differences.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR results with columns: subject_id, ROI_Label, PR, n_voxels
    participants_df : pd.DataFrame
        Participant metadata with columns: participant_id, group
    roi_labels : np.ndarray
        1D array of ROI labels (defines which ROIs to summarize)
    confidence_level : float, default=0.95
        Confidence level for intervals (0.95 = 95% CI)

    Returns
    -------
    pd.DataFrame
        Summary statistics with one row per ROI-group combination:
        - ROI_Label: ROI identifier (integer)
        - group: 'expert' or 'novice'
        - mean_PR: Mean participation ratio across subjects
        - ci_low, ci_high: Lower and upper bounds of confidence interval
        - sem: Standard error of the mean
        - n: Number of valid subjects (excludes NaN values)

    Notes
    -----
    Confidence intervals are computed using the t-distribution, which accounts
    for the uncertainty due to finite sample size. The CI gets narrower as
    sample size increases.

    NaN values are excluded on a per-ROI basis, so different ROIs may have
    different numbers of subjects if some subjects had missing data.

    Examples
    --------
    >>> summary_stats = summarize_pr_by_group(
    ...     pr_df, participants_df, roi_labels
    ... )
    >>> # View expert statistics for first ROI
    >>> expert_stats = summary_stats[
    ...     (summary_stats['group'] == 'expert') &
    ...     (summary_stats['ROI_Label'] == 1)
    ... ]
    >>> print(f"ROI 1 experts: mean={expert_stats['mean_PR'].values[0]:.2f}")
    """
    from common.stats_utils import compute_group_mean_and_ci

    # Step 1: Merge PR data with group labels (expert/novice)
    pr_with_group = pr_df.merge(
        participants_df[['participant_id', 'group']],
        left_on='subject_id',
        right_on='participant_id',
        how='left'
    )

    summary_records = []

    # Step 2: Loop over each group and ROI to compute statistics
    for group in ['expert', 'novice']:
        # Extract data for this group only
        group_data = pr_with_group[pr_with_group['group'] == group]

        for roi_label in roi_labels:
            # Extract PR values for this ROI
            roi_data = group_data[group_data['ROI_Label'] == roi_label]['PR'].values

            # Remove NaN values (some subjects may have failed for this ROI)
            roi_clean = roi_data[~np.isnan(roi_data)]

            if roi_clean.size > 0:
                # Compute mean and 95% CI using t-distribution
                mean, ci_low, ci_high = compute_group_mean_and_ci(
                    roi_clean, confidence_level=confidence_level
                )

                # Compute standard error of the mean
                sem = np.std(roi_clean, ddof=1) / np.sqrt(len(roi_clean))

                summary_records.append({
                    'ROI_Label': int(roi_label),
                    'group': group,
                    'mean_PR': mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'sem': sem,
                    'n': len(roi_clean)
                })
            else:
                # No valid data for this ROI/group combination
                logger.warning(f"No valid PR data for ROI {roi_label}, group {group}")
                summary_records.append({
                    'ROI_Label': int(roi_label),
                    'group': group,
                    'mean_PR': np.nan,
                    'ci_low': np.nan,
                    'ci_high': np.nan,
                    'sem': np.nan,
                    'n': 0
                })

    summary_df = pd.DataFrame(summary_records)

    logger.info(f"Computed summary statistics for {len(roi_labels)} ROIs, 2 groups")

    return summary_df


def compare_groups_welch_fdr(
    pr_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    roi_labels: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform statistical comparison of experts vs novices with FDR correction.

    This function runs Welch's t-test for each ROI to compare expert and novice
    PR values. Welch's t-test is used because it doesn't assume equal variances
    between groups. The Benjamini-Hochberg FDR procedure corrects for multiple
    comparisons across ROIs.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR results with columns: subject_id, ROI_Label, PR
    participants_df : pd.DataFrame
        Participant metadata with columns: participant_id, group
    roi_labels : np.ndarray
        1D array of ROI labels (defines which ROIs to test)
    alpha : float, default=0.05
        False discovery rate threshold (0.05 = 5% FDR)

    Returns
    -------
    pd.DataFrame
        Statistical test results with one row per ROI:
        - ROI_Label: ROI identifier
        - t_stat: Welch's t-statistic
        - p_val: Uncorrected p-value
        - dof: Degrees of freedom (Welch-Satterthwaite approximation)
        - cohen_d: Effect size (standardized mean difference)
        - mean_diff: Raw mean difference (expert - novice)
        - ci95_low, ci95_high: 95% CI for mean difference
        - p_val_fdr: FDR-corrected p-value
        - significant: True if p_val < alpha (uncorrected)
        - significant_fdr: True if p_val_fdr < alpha (FDR-corrected)

    Notes
    -----
    **Welch's t-test** is preferred over Student's t-test because it doesn't
    assume equal variances. This is more robust when group sizes or variances
    differ.

    **FDR correction** (Benjamini-Hochberg) controls the expected proportion of
    false discoveries among rejected hypotheses. It's less conservative than
    Bonferroni but still provides protection against false positives.

    Examples
    --------
    >>> stats_df = compare_groups_welch_fdr(
    ...     pr_df, participants_df, roi_labels, alpha=0.05
    ... )
    >>> # Find significant ROIs after FDR correction
    >>> sig_rois = stats_df[stats_df['significant_fdr']]['ROI_Label'].values
    >>> print(f"Found {len(sig_rois)} significant ROIs (FDR < 0.05)")
    """
    from common.stats_utils import per_roi_welch_and_fdr

    logger.info(f"Running statistical analysis: expert vs novice, {len(roi_labels)} ROIs")

    # Step 1: Merge PR data with group assignment
    pr_with_group = pr_df.merge(
        participants_df[['participant_id', 'group']],
        left_on='subject_id',
        right_on='participant_id',
        how='left'
    )

    # Step 2: Pivot to wide format (subjects × ROIs) separately for each group
    # This makes it easy to pass to the statistical testing function
    expert_data = pr_with_group[pr_with_group['group'] == 'expert']
    novice_data = pr_with_group[pr_with_group['group'] == 'novice']

    expert_pivot = expert_data.pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )
    novice_pivot = novice_data.pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )

    # Step 3: Ensure ROIs are in the correct order (matching roi_labels)
    expert_vals = expert_pivot[roi_labels].values  # Shape: (n_experts, n_rois)
    novice_vals = novice_pivot[roi_labels].values  # Shape: (n_novices, n_rois)

    # Step 4: Run Welch t-tests with FDR correction
    # This function handles all the statistical testing internally
    stats_df = per_roi_welch_and_fdr(
        expert_vals, novice_vals, roi_labels, alpha=alpha
    )

    # Step 5: Log summary of results
    n_sig = stats_df['significant'].sum()
    n_sig_fdr = stats_df['significant_fdr'].sum()
    logger.info(f"Results: {n_sig} ROIs significant (uncorrected), "
                f"{n_sig_fdr} ROIs significant (FDR-corrected)")

    return stats_df


__all__ = [
    'summarize_pr_by_group',
    'compare_groups_welch_fdr',
]
