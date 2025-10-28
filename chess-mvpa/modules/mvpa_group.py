"""
Shared MVPA group statistics routines (analysis-specific glue).

Provides a single function to compute per-ROI group comparisons and
within-group vs-chance tests for a given method (svm or rsa_corr).
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from common import CONFIG
from common.stats_utils import (
    per_roi_welch_and_fdr,
    compute_group_mean_and_ci,
    per_roi_one_sample_vs_value,
)


def compute_group_stats_for_method(
    df_method: pd.DataFrame,
    roi_names: List[str],
    method: str,
    chance_map: Dict[str, float],
    alpha: float,
) -> Dict[str, Dict[str, pd.DataFrame | float | List[Tuple[float, float, float]]]]:
    """
    Compute per-ROI group stats for all targets within a method.

    Parameters
    ----------
    df_method : pd.DataFrame
        Columns: subject, expert(bool), target, <ROI 1> ... <ROI N>
    roi_names : list of str
        Display names for ROI columns in df_method
    method : {'svm','rsa_corr'}
        Analysis method; determines chance handling
    chance_map : dict
        Mapping target -> chance value (used if method == 'svm')
    alpha : float
        FDR alpha

    Returns
    -------
    dict
        target -> blocks dict with keys:
          - welch_expert_vs_novice: DataFrame
          - experts_vs_chance: DataFrame
          - novices_vs_chance: DataFrame
          - chance: float
          - experts_desc: list[(mean, ci_low, ci_high)] per ROI
          - novices_desc: list[(mean, ci_low, ci_high)] per ROI
    """
    targets = sorted(df_method["target"].dropna().unique())
    results: Dict[str, Dict[str, object]] = {}

    for tgt in targets:
        df_tgt = df_method[df_method["target"] == tgt].copy()
        df_exp = df_tgt[df_tgt["expert"] == True].drop(columns=["subject", "expert", "target"])
        df_nov = df_tgt[df_tgt["expert"] == False].drop(columns=["subject", "expert", "target"])
        df_exp = df_exp[roi_names]
        df_nov = df_nov[roi_names]

        welch = per_roi_welch_and_fdr(
            df_exp.values, df_nov.values, np.arange(1, len(roi_names) + 1), alpha=alpha
        )
        welch.insert(0, "ROI_Name", roi_names)

        if method == 'svm':
            chance = float(chance_map.get(tgt, np.nan))
        else:
            # RSA correlations: chance is 0
            chance = float(CONFIG.get('CHANCE_LEVEL_RSA', 0.0))

        vs_chance_exp = per_roi_one_sample_vs_value(
            df_exp.values, roi_names, chance, alpha=alpha, alternative='greater'
        )
        vs_chance_nov = per_roi_one_sample_vs_value(
            df_nov.values, roi_names, chance, alpha=alpha, alternative='greater'
        )

        # Descriptive stats (mean and 95% CI)
        exp_desc: List[Tuple[float, float, float]] = []
        nov_desc: List[Tuple[float, float, float]] = []
        for roi in roi_names:
            x_exp = df_exp[roi].values
            x_exp = x_exp[~np.isnan(x_exp)]
            x_nov = df_nov[roi].values
            x_nov = x_nov[~np.isnan(x_nov)]

            if x_exp.size > 1:
                exp_desc.append(compute_group_mean_and_ci(x_exp, confidence_level=0.95))
            else:
                exp_desc.append((np.nan, np.nan, np.nan))

            if x_nov.size > 1:
                nov_desc.append(compute_group_mean_and_ci(x_nov, confidence_level=0.95))
            else:
                nov_desc.append((np.nan, np.nan, np.nan))

        results[tgt] = {
            'welch_expert_vs_novice': welch,
            'experts_vs_chance': vs_chance_exp,
            'novices_vs_chance': vs_chance_nov,
            'chance': chance,
            'experts_desc': exp_desc,
            'novices_desc': nov_desc,
        }

    return results


__all__ = [
    'compute_group_stats_for_method',
]

