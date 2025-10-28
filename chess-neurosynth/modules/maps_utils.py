"""
Neurosynth mapping utilities: transforms and spatial correlations (local).

Contains neurosynth-specific helpers built on top of reusable common utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from nilearn import image
from scipy.stats import t as _t
from scipy.stats import norm as _norm

from common.neuro_utils import get_gray_matter_mask, clean_voxels
from common.stats_utils import apply_fdr_correction, correlate_vectors_bootstrap


def t_to_two_tailed_z(t_map: np.ndarray, dof: int) -> np.ndarray:
    """
    Convert a t-map to a signed two-tailed z-map, retaining original sign.
    """
    t_abs = np.abs(t_map)
    p_two = 2 * _t.sf(t_abs, df=int(dof))
    z_abs = _norm.isf(p_two / 2)
    z = np.sign(t_map) * z_abs
    z[t_abs == 0] = 0.0
    return z


def split_zmap_by_sign(z_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a z-map into positive and negative magnitude maps.
    Returns (z_pos, z_neg) where z_neg is positive magnitude of negative voxels.
    """
    z_pos = np.where(z_map > 0, z_map, 0.0)
    z_neg = np.where(z_map < 0, -z_map, 0.0)
    return z_pos, z_neg


def bootstrap_corr_diff(
    term_vec: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ci_alpha: float,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap Δr = r(term, x) − r(term, y) with percentile CI and two-sided p.
    """
    n = term_vec.shape[0]
    seeds = rng.integers(0, 2**32 - 1, size=int(n_boot))
    diffs = np.empty(int(n_boot), dtype=float)
    for i, s in enumerate(seeds):
        sub_rng = np.random.default_rng(int(s))
        idx = sub_rng.integers(0, n, size=n)
        r_pos = np.corrcoef(term_vec[idx], x_vec[idx])[0, 1]
        r_neg = np.corrcoef(term_vec[idx], y_vec[idx])[0, 1]
        diffs[i] = r_pos - r_neg

    diffs.sort()
    lo = float(np.percentile(diffs, 100 * ci_alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - ci_alpha / 2)))
    mean_diff = float(np.mean(diffs))
    # two-sided around 0
    p_low = float(np.mean(diffs <= 0))
    p_high = float(np.mean(diffs >= 0))
    p_val = 2 * min(p_low, p_high)
    return mean_diff, lo, hi, float(p_val)


def compute_all_zmap_correlations(
    z_pos: np.ndarray,
    z_neg: np.ndarray,
    term_maps: Dict[str, str],
    ref_img,
    n_boot: int = 10000,
    fdr_alpha: float = 0.05,
    ci_alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Correlate directional z-maps (pos/neg) with term maps and estimate Δr.

    Returns three DataFrames: df_pos, df_neg, df_diff with FDR columns.
    """
    rng = np.random.default_rng(int(random_state))

    # Build flat GM mask in ref space
    gm_mask = get_gray_matter_mask(ref_img)
    mask_flat = gm_mask.get_fdata().ravel().astype(bool)

    flat_pos = z_pos.ravel()
    flat_neg = z_neg.ravel()

    rec_pos = []
    rec_neg = []
    rec_diff = []

    for term, path in term_maps.items():
        term_img = image.resample_to_img(image.load_img(str(path)), ref_img, force_resample=True, copy_header=True)
        flat_t = term_img.get_fdata().ravel()

        # Clean voxels across the three vectors
        stacked = np.vstack([flat_pos, flat_neg, flat_t])
        stacked_clean, keep = clean_voxels(stacked, brain_mask_flat=mask_flat, var_thresh=1e-5)
        x, y, t = stacked_clean

        # POS
        r_pos, p_pos, ci_lo_pos, ci_hi_pos = correlate_vectors_bootstrap(t, x, method='pearson', n_bootstraps=n_boot)
        rec_pos.append((term, r_pos, ci_lo_pos, ci_hi_pos, p_pos))

        # NEG
        r_neg, p_neg, ci_lo_neg, ci_hi_neg = correlate_vectors_bootstrap(t, y, method='pearson', n_bootstraps=n_boot)
        rec_neg.append((term, r_neg, ci_lo_neg, ci_hi_neg, p_neg))

        # DIFF
        diff, lo, hi, p_diff = bootstrap_corr_diff(t, x, y, n_boot=n_boot, rng=rng, ci_alpha=ci_alpha)
        rec_diff.append((term, r_pos, r_neg, diff, lo, hi, p_diff))

    df_pos = pd.DataFrame(rec_pos, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_neg = pd.DataFrame(rec_neg, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_diff = pd.DataFrame(rec_diff, columns=['term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw'])

    # FDR per table
    for df in (df_pos, df_neg, df_diff):
        rej, p_fdr = apply_fdr_correction(df['p_raw'].values, alpha=fdr_alpha)
        df['p_fdr'] = p_fdr
        df['sig'] = rej

    return df_pos, df_neg, df_diff

