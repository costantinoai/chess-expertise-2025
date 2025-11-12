"""
Neurosynth mapping utilities: transforms and spatial correlations (local).

Contains neurosynth-specific helpers built on top of reusable common utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from nilearn import image
from scipy.stats import t, norm

from common.neuro_utils import get_gray_matter_mask, clean_voxels
from common.stats_utils import apply_fdr_correction, correlate_vectors_bootstrap


def t_to_two_tailed_z(t_map: np.ndarray, dof: int) -> np.ndarray:
    """
    Convert a t-map to a signed two-tailed z-map, retaining original sign.

    Transforms t-statistics to z-scores using two-tailed p-values, which
    standardizes effect sizes across different sample sizes while preserving
    directionality. This is useful for comparing maps with different degrees
    of freedom or for correlation with reference maps.

    Parameters
    ----------
    t_map : np.ndarray
        3D array of t-statistics (can be positive or negative)
    dof : int
        Degrees of freedom for the t-distribution

    Returns
    -------
    np.ndarray
        3D array of z-scores with same shape and sign as input t_map

    Notes
    -----
    Conversion procedure:
    1. Take absolute value for two-tailed test
    2. Compute two-tailed p-value: P(|T| > |t|) = 2 * P(T > |t|)
    3. Convert p-value to z-score using inverse survival function
    4. Restore original sign (positive t -> positive z, negative t -> negative z)

    Examples
    --------
    >>> # Convert group comparison t-map to z-map
    >>> import numpy as np
    >>> t_map = np.random.randn(10, 10, 10) * 3  # Simulated t-map
    >>> dof = 42  # n_experts + n_novices - 2
    >>> z_map = t_to_two_tailed_z(t_map, dof)
    >>> # z_map has same shape and signs as t_map, but standardized scale
    """
    # Take absolute value for two-tailed testing (we want magnitude, not direction yet)
    t_abs = np.abs(t_map)

    # Compute two-tailed p-value: P(|T| > |t_obs|) = 2 * P(T > |t_obs|)
    # sf() is the survival function: P(X > x) = 1 - CDF(x), more accurate for tail probabilities
    p_two = 2 * t.sf(t_abs, df=int(dof))

    # Convert two-tailed p-value back to z-score using inverse survival function.
    # isf(p/2) gives the z-value where P(|Z| > z) = p (two-tailed)
    # This standardizes effect sizes from t-distribution to standard normal distribution.
    z_abs = norm.isf(p_two / 2)

    # Restore original sign: positive t → positive z, negative t → negative z
    z = np.sign(t_map) * z_abs
    z[t_abs == 0] = 0.0  # Handle exactly zero values explicitly
    return z


def split_zmap_by_sign(z_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a z-map into positive and negative magnitude maps.

    Separates a signed z-map into two maps representing regions where the
    target group shows enhanced (positive) or reduced (negative) activation
    relative to the control group. This enables separate correlation analyses
    for positive and negative effects.

    Parameters
    ----------
    z_map : np.ndarray
        3D array of signed z-scores from group comparison

    Returns
    -------
    z_pos : np.ndarray
        Positive z-values (negative voxels set to 0)
    z_neg : np.ndarray
        Absolute value of negative z-values (positive voxels set to 0)

    Notes
    -----
    - z_pos contains only voxels where z > 0 (e.g., expert > novice)
    - z_neg contains absolute values where z < 0 (e.g., novice > expert)
    - Both maps have same shape as input, with non-selected regions as 0

    Examples
    --------
    >>> import numpy as np
    >>> z_map = np.array([2.5, -1.8, 0.5, -3.2, 1.0])
    >>> z_pos, z_neg = split_zmap_by_sign(z_map)
    >>> print(z_pos)
    [2.5  0.   0.5  0.   1.0]
    >>> print(z_neg)
    [0.   1.8  0.   3.2  0. ]
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
    ci_alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap correlation difference Δr = r(term, x) − r(term, y) with CI and p-value.

    Computes the difference between two correlations and estimates its confidence
    interval using bootstrap resampling. This is used to test whether a meta-analytic
    term shows differential association with expert-enhanced versus expert-reduced
    brain regions.

    The bootstrap procedure:
    1. Resample voxels with replacement n_boot times
    2. For each resample, compute r_x = corr(term, x) and r_y = corr(term, y)
    3. Compute difference: diff_i = r_x - r_y
    4. Use percentile method for CI and two-sided test around 0 for p-value

    Parameters
    ----------
    term_vec : np.ndarray
        1D array of Neurosynth term map values across voxels
    x_vec : np.ndarray
        1D array of z-values for positive map (expert-enhanced regions)
    y_vec : np.ndarray
        1D array of z-values for negative map (expert-reduced regions)
    n_boot : int
        Number of bootstrap samples (typically 10,000)
    rng : np.random.Generator
        Random number generator for reproducibility
    ci_alpha : float, default=0.05
        Significance level for CI (0.05 → 95% CI)

    Returns
    -------
    mean_diff : float
        Bootstrap mean of correlation differences (Δr)
    ci_low : float
        Lower bound of percentile CI
    ci_high : float
        Upper bound of percentile CI
    p_val : float
        Two-sided p-value testing H₀: Δr = 0

    Notes
    -----
    - P-value is computed as 2 * min(P(Δr ≤ 0), P(Δr ≥ 0))
    - Positive Δr indicates term is more associated with expert-enhanced regions
    - Negative Δr indicates term is more associated with expert-reduced regions
    - All inputs must have the same length (already cleaned/masked)

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> term = np.random.randn(1000)
    >>> x = term + np.random.randn(1000) * 0.3  # Stronger correlation
    >>> y = term + np.random.randn(1000) * 0.7  # Weaker correlation
    >>> mean_diff, ci_low, ci_high, p = bootstrap_corr_diff(term, x, y, 1000, rng)
    >>> # mean_diff should be positive (x has stronger correlation than y)
    """
    n = term_vec.shape[0]
    # Generate independent seeds for each bootstrap iteration (for parallel safety)
    seeds = rng.integers(0, 2**32 - 1, size=int(n_boot))
    diffs = np.empty(int(n_boot), dtype=float)

    # Bootstrap resampling: sample with replacement for each iteration
    for i, seed in enumerate(seeds):
        sub_rng = np.random.default_rng(int(seed))
        idx = sub_rng.integers(0, n, size=n)

        # Compute correlations on resampled data
        r_x = np.corrcoef(term_vec[idx], x_vec[idx])[0, 1]
        r_y = np.corrcoef(term_vec[idx], y_vec[idx])[0, 1]
        diffs[i] = r_x - r_y

    # Compute confidence interval using percentile method
    diffs.sort()
    ci_low = float(np.percentile(diffs, 100 * ci_alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - ci_alpha / 2)))
    mean_diff = float(np.mean(diffs))

    # Two-sided p-value: probability of observing Δr as extreme as observed under H₀
    p_low = float(np.mean(diffs <= 0))
    p_high = float(np.mean(diffs >= 0))
    p_val = 2 * min(p_low, p_high)

    return mean_diff, ci_low, ci_high, p_val


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
    Correlate directional z-maps (pos/neg) with Neurosynth term maps with bootstrap CIs.

    This function computes spatial correlations between group comparison maps
    (expert vs novice) and Neurosynth meta-analytic term association maps,
    estimating confidence intervals via bootstrap resampling and applying FDR
    correction for multiple comparisons.

    By splitting the comparison map by sign, we can identify which cognitive
    terms are associated with expert-enhanced regions (Z+) versus expert-reduced
    regions (Z-). Bootstrap resampling provides robust confidence intervals
    that account for spatial dependencies in the data.

    For each term map, compute:
    - r_pos: correlation between term map and Z+ (expert-enhanced regions)
    - r_neg: correlation between term map and Z- (expert-reduced regions)
    - r_diff: difference (r_pos - r_neg) with bootstrap CI

    A positive r_diff indicates the term is more strongly associated with
    expert-enhanced regions; a negative r_diff indicates stronger association
    with expert-reduced regions.

    Parameters
    ----------
    z_pos : np.ndarray
        3D array of positive z-values (zeros elsewhere)
    z_neg : np.ndarray
        3D array of absolute negative z-values (zeros elsewhere)
    term_maps : dict
        Dictionary mapping term names to file paths
    ref_img : Nifti1Image
        Reference image for resampling term maps
    n_boot : int, default=10000
        Number of bootstrap samples for CI estimation
    fdr_alpha : float, default=0.05
        Significance level for FDR correction (0.05 → 5% FDR)
    ci_alpha : float, default=0.05
        Significance level for CIs (0.05 → 95% CI)
    random_state : int, default=42
        Random seed for reproducible bootstrap sampling

    Returns
    -------
    df_pos : pd.DataFrame
        Columns: term, r, CI_low, CI_high, p_raw, p_fdr, sig
        Correlations with expert-enhanced regions
    df_neg : pd.DataFrame
        Columns: term, r, CI_low, CI_high, p_raw, p_fdr, sig
        Correlations with expert-reduced regions
    df_diff : pd.DataFrame
        Columns: term, r_pos, r_neg, r_diff, CI_low, CI_high, p_raw, p_fdr, sig
        Correlation differences with bootstrap CIs

    Notes
    -----
    - Gray matter masking ensures correlation is computed only within brain
    - Voxel cleaning removes NaNs and zero-variance voxels
    - All term maps are resampled to reference image geometry before correlation
    - Bootstrap CIs are computed using pingouin for individual correlations
    - Bootstrap correlation differences use custom percentile CI method
    - FDR correction is applied separately to each DataFrame (pos, neg, diff)

    Statistical Approach
    --------------------
    1. Individual correlations (pos/neg): Bootstrap resampling via pingouin
       - Resamples voxels with replacement n_boot times
       - Computes percentile-based 95% CI
       - Two-sided p-value from permutation distribution

    2. Correlation differences: Custom bootstrap
       - For each resample: compute Δr = r_pos - r_neg
       - Percentile CI from bootstrap distribution of Δr
       - Two-sided p-value: P(|Δr| ≥ |observed|) under H₀: Δr = 0

    3. Multiple testing: FDR correction (Benjamini-Hochberg)
       - Applied within each table (pos, neg, diff)
       - Controls false discovery rate at fdr_alpha level

    Examples
    --------
    >>> # After computing z-map from group comparison
    >>> z_pos, z_neg = split_zmap_by_sign(z_map)
    >>> term_maps = load_term_maps('neurosynth_maps/')
    >>> df_pos, df_neg, df_diff = compute_all_zmap_correlations(
    ...     z_pos, z_neg, term_maps, ref_img,
    ...     n_boot=10000, fdr_alpha=0.05
    ... )
    >>> # Find significant terms after FDR correction
    >>> sig_pos = df_pos[df_pos['sig']]
    >>> # Find terms with largest difference and narrow CI
    >>> top_diff = df_diff.nlargest(5, 'r_diff')
    >>> print(top_diff[['term', 'r_diff', 'CI_low', 'CI_high', 'p_fdr']])
    """
    # Initialize random number generator for reproducible bootstrap sampling
    rng = np.random.default_rng(int(random_state))

    # Build gray matter mask to restrict analysis to brain voxels
    gm_mask = get_gray_matter_mask(ref_img)
    mask_flat = gm_mask.get_fdata().ravel().astype(bool)

    # Flatten 3D maps to 1D vectors for correlation
    flat_pos = z_pos.ravel()
    flat_neg = z_neg.ravel()

    rec_pos = []
    rec_neg = []
    rec_diff = []

    for term, path in term_maps.items():
        # Load and resample term map to match reference image geometry
        term_img = image.resample_to_img(
            image.load_img(str(path)), ref_img,
            force_resample=True, copy_header=True
        )
        flat_t = term_img.get_fdata().ravel()

        # Clean voxels: remove NaNs, zero-variance voxels, and apply gray matter mask.
        # Stack all three vectors for joint cleaning to ensure consistent voxel sets.
        stacked = np.vstack([flat_pos, flat_neg, flat_t])
        stacked_clean, keep = clean_voxels(
            stacked, brain_mask_flat=mask_flat, var_thresh=1e-5
        )
        x, y, t = stacked_clean

        # === Correlation: term vs Z+ (expert-enhanced regions) ===
        # Use pingouin's bootstrap correlation with percentile CI
        r_pos, p_pos, ci_lo_pos, ci_hi_pos = correlate_vectors_bootstrap(
            t, x, method='pearson', n_bootstraps=n_boot
        )
        rec_pos.append((term, r_pos, ci_lo_pos, ci_hi_pos, p_pos))

        # === Correlation: term vs Z− (expert-reduced regions) ===
        r_neg, p_neg, ci_lo_neg, ci_hi_neg = correlate_vectors_bootstrap(
            t, y, method='pearson', n_bootstraps=n_boot
        )
        rec_neg.append((term, r_neg, ci_lo_neg, ci_hi_neg, p_neg))

        # === Correlation difference: Δr = r_pos − r_neg ===
        # Custom bootstrap for difference with percentile CI and two-sided p-value
        diff, ci_lo_diff, ci_hi_diff, p_diff = bootstrap_corr_diff(
            t, x, y, n_boot=n_boot, rng=rng, ci_alpha=ci_alpha
        )
        rec_diff.append((term, r_pos, r_neg, diff, ci_lo_diff, ci_hi_diff, p_diff))

    # Build DataFrames with all statistical results
    df_pos = pd.DataFrame(
        rec_pos, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw']
    )
    df_neg = pd.DataFrame(
        rec_neg, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw']
    )
    df_diff = pd.DataFrame(
        rec_diff, columns=['term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw']
    )

    # Apply FDR correction separately to each table to control false discoveries
    # Uses Benjamini-Hochberg procedure to maintain fdr_alpha rate
    for df in (df_pos, df_neg, df_diff):
        rej, p_fdr = apply_fdr_correction(df['p_raw'].values, alpha=fdr_alpha)
        df['p_fdr'] = p_fdr
        df['sig'] = rej

    return df_pos, df_neg, df_diff

