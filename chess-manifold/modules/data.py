"""
Data loading and transformation utilities for manifold analysis.

This module provides functions for loading, transforming, and preparing data
for PR visualization and analysis.

Functions
---------
load_atlas_and_metadata : Load atlas, ROI info, and participants
pivot_pr_long_to_subject_roi : Reshape PR data for heatmap visualization
correlate_pr_with_roi_size : Analyze PR vs ROI size relationships
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from typing import Tuple

from common.bids_utils import load_participants_tsv, load_roi_metadata, merge_group_labels
from .utils import ensure_roi_order

logger = logging.getLogger(__name__)


def load_atlas_and_metadata(
    atlas_path: Path,
    roi_info_path: Path,
    participants_path: Path,
    load_atlas_func
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load atlas volume, ROI metadata, and participant information.

    This is the entry point for loading all ground truth data needed for
    manifold analysis.

    Parameters
    ----------
    atlas_path : Path
        Path to atlas NIfTI file with integer ROI labels
    roi_info_path : Path
        Path to ROI metadata TSV (region_info.tsv)
    participants_path : Path
        Path to participants.tsv file
    load_atlas_func : callable
        Function to load atlas (from common.neuro_utils.load_atlas)

    Returns
    -------
    atlas_data : np.ndarray
        3D volume with ROI labels at each voxel
    roi_labels : np.ndarray
        Unique ROI labels in the atlas
    roi_info : pd.DataFrame
        ROI metadata (names, colors, etc.)
    participants : pd.DataFrame
        Participant information (group, ELO, etc.)
    """
    logger.info("Loading atlas and metadata...")

    # Load atlas volume and extract ROI labels
    atlas_data, roi_labels = load_atlas_func(atlas_path)
    logger.info(f"  Loaded atlas: {len(roi_labels)} ROIs")

    # Load ROI metadata (use centralized loader; path → directory)
    roi_info = load_roi_metadata(Path(roi_info_path).parent)
    logger.info(f"  Loaded ROI metadata: {len(roi_info)} regions")

    # Load participant data
    participants = load_participants_tsv(participants_path)
    logger.info(f"  Loaded participants: {len(participants)} subjects")

    return atlas_data, roi_labels, roi_info, participants


def pivot_pr_long_to_subject_roi(
    pr_df: pd.DataFrame,
    participants: pd.DataFrame,
    roi_labels: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Reshape PR data from long format to subject × ROI matrix for heatmaps.

    Converts one-row-per-subject-ROI to one-row-per-subject, with experts
    sorted to appear first in the matrix.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR data (columns: subject_id, ROI_Label, PR, n_voxels)
    participants : pd.DataFrame
        Participant metadata (columns: participant_id, group)
    roi_labels : np.ndarray
        ROI labels defining column order

    Returns
    -------
    pr_matrix : np.ndarray
        2D array (n_subjects × n_rois) with experts in top rows
    n_experts : int
        Number of expert subjects (marks boundary in heatmap)

    Notes
    -----
    Experts are sorted first so heatmaps show expert/novice separation clearly.
    Each row is one subject, each column is one ROI.
    """
    logger.info("Reshaping PR data to subject × ROI matrix...")

    # Add group labels
    pr_with_group = merge_group_labels(pr_df, participants, subject_col='subject_id')

    # Pivot: rows=subjects, columns=ROIs
    pr_pivot = pr_with_group.pivot(
        index='subject_id',
        columns='ROI_Label',
        values='PR'
    )
    pr_pivot = ensure_roi_order(pr_pivot, roi_labels)

    # Add group labels and ELO ratings, then sort (experts first, then by ELO descending within each group)
    subject_info = participants[['participant_id', 'group', 'rating']].copy()
    subject_info = subject_info.rename(columns={'participant_id': 'subject_id'})
    subject_info = subject_info.drop_duplicates().set_index('subject_id')

    pr_pivot = pr_pivot.join(subject_info)

    # Explicitly separate experts and novices, sort each by ELO, then concatenate (experts first)
    expert_rows = pr_pivot[pr_pivot['group'] == 'expert'].sort_values('rating', ascending=False)
    novice_rows = pr_pivot[pr_pivot['group'] == 'novice'].sort_values('rating', ascending=False)
    pr_pivot = pd.concat([expert_rows, novice_rows])

    # Clean up temporary columns
    pr_pivot = pr_pivot.drop(columns=['rating'])

    # Extract matrix and count experts
    pr_matrix = pr_pivot.drop(columns='group').values
    n_experts = len(expert_rows)

    logger.info(f"  Created matrix: {pr_matrix.shape[0]} subjects × {pr_matrix.shape[1]} ROIs")
    logger.info(f"  Experts: rows 0-{n_experts-1}, Novices: rows {n_experts}-{pr_matrix.shape[0]-1}")

    return pr_matrix, n_experts


def correlate_pr_with_roi_size(
    pr_df: pd.DataFrame,
    participants: pd.DataFrame,
    roi_info: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Analyze relationship between PR and ROI size (voxel count).

    Computes correlations to check whether PR differences are confounded by
    ROI size. Non-significant correlations suggest PR differences are real,
    not artifacts of ROI size.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR data (columns: subject_id, ROI_Label, PR, n_voxels)
    participants : pd.DataFrame
        Participant metadata (columns: participant_id, group)
    roi_info : pd.DataFrame
        ROI metadata (columns: roi_id, pretty_name, color)

    Returns
    -------
    group_avg : pd.DataFrame
        Average PR and voxel count per ROI and group (for plotting)
    diff_data : pd.DataFrame
        Expert-novice PR differences with voxel counts
    stats_vox : dict
        Regression statistics:
        - 'expert': {slope, intercept, r, p} for expert PR vs voxels
        - 'novice': {slope, intercept, r, p} for novice PR vs voxels
        - 'diff': {slope, intercept, r, p} for (E-N) difference vs voxels

    Notes
    -----
    If PR differences correlate strongly with voxel count, ROI size might
    confound the results. Weak/non-significant correlations suggest PR
    differences are independent of ROI size.
    """
    logger.info("Computing PR vs voxel count correlations...")

    # Add group labels
    pr_with_group = merge_group_labels(pr_df, participants, subject_col='subject_id')

    # Average by ROI and group
    group_avg = pr_with_group.groupby(['ROI_Label', 'group']).agg({
        'PR': 'mean',
        'n_voxels': 'mean'
    }).reset_index()

    # Add ROI colors/names for plotting
    group_avg = group_avg.merge(
        roi_info[['roi_id', 'pretty_name', 'color']],
        left_on='ROI_Label',
        right_on='roi_id',
        how='left'
    )

    # Per-group correlations
    stats_vox = {}
    for grp in ['expert', 'novice']:
        gd = group_avg[group_avg['group'] == grp]
        slope, intercept, r, p, _ = linregress(gd['n_voxels'], gd['PR'])
        stats_vox[grp] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r': float(r),
            'p': float(p)
        }
        logger.info(f"  {grp.capitalize()}: r={r:.3f}, p={p:.4f}")

    # Expert-novice difference vs average voxels
    expert_avg = group_avg[group_avg['group'] == 'expert'][['ROI_Label', 'PR', 'n_voxels', 'color']]
    novice_avg = group_avg[group_avg['group'] == 'novice'][['ROI_Label', 'PR', 'n_voxels']]
    diff_data = expert_avg.merge(novice_avg, on='ROI_Label', suffixes=('_expert', '_novice'))
    diff_data['PR_diff'] = diff_data['PR_expert'] - diff_data['PR_novice']
    diff_data['n_voxels_avg'] = (diff_data['n_voxels_expert'] + diff_data['n_voxels_novice']) / 2

    slope, intercept, r, p, _ = linregress(diff_data['n_voxels_avg'], diff_data['PR_diff'])
    stats_vox['diff'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r': float(r),
        'p': float(p)
    }
    logger.info(f"  PR difference (E-N): r={r:.3f}, p={p:.4f}")

    return group_avg, diff_data, stats_vox


__all__ = [
    'load_atlas_and_metadata',
    'pivot_pr_long_to_subject_roi',
    'correlate_pr_with_roi_size',
]
