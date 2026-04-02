"""
MVPA plotting utilities for extracting per-ROI bar data from group stats.

Provides a single function to extract mean/CI series for Experts/Novices
and associated per-ROI metadata (names, colors, label colors, p-values)
for both RSA correlations and SVM decoding.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from common.plotting import format_roi_labels_and_colors


def extract_mvpa_bar_data(
    group_stats: dict,
    roi_info,
    targets: List[str],
    method: str,
    subtract_chance: bool = False,
) -> Dict[str, dict]:
    """
    Extract bar plot data for MVPA RSA or SVM results.

    Parameters
    ----------
    group_stats : dict
        Artifact index loaded from mvpa_group_stats.pkl
    roi_info : DataFrame
        ROI metadata table consumed by format_roi_labels_and_colors
    targets : list[str]
        Target names to extract (e.g., ['visual_similarity','strategy','checkmate'])
    method : {'rsa_corr','svm'}
        Which method block to read from group_stats
    subtract_chance : bool, default=False
        If True, subtract method-specific chance from means and CIs (for SVM)

    Returns
    -------
    dict
        target -> dict with keys: exp_means, exp_cis, nov_means, nov_cis,
        pvals, roi_names, roi_colors, label_colors
    """
    out: Dict[str, dict] = {}
    if method not in group_stats:
        return out

    for tgt in targets:
        if tgt not in group_stats[method]:
            continue
        blocks = group_stats[method][tgt]
        welch = blocks["welch_expert_vs_novice"]
        exp_desc = blocks["experts_desc"]
        nov_desc = blocks["novices_desc"]
        chance = float(blocks.get("chance", 0.0)) if subtract_chance else 0.0

        exp_means = np.array([m - chance for (m, _, _) in exp_desc])
        exp_cis = [(lo - chance, hi - chance) for (_, lo, hi) in exp_desc]
        nov_means = np.array([m - chance for (m, _, _) in nov_desc])
        nov_cis = [(lo - chance, hi - chance) for (_, lo, hi) in nov_desc]

        roi_names, roi_colors, label_colors = format_roi_labels_and_colors(
            welch, roi_info, alpha=0.05
        )
        pvals = welch["p_val_fdr"].values

        out[tgt] = dict(
            exp_means=exp_means,
            exp_cis=exp_cis,
            nov_means=nov_means,
            nov_cis=nov_cis,
            pvals=pvals,
            roi_names=roi_names,
            roi_colors=roi_colors,
            label_colors=label_colors,
        )
    return out


__all__ = [
    'extract_mvpa_bar_data',
]

