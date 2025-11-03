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


def compute_all_zmap_correlations(
    z_pos: np.ndarray,
    z_neg: np.ndarray,
    term_maps: Dict[str, str],
    ref_img,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Correlate directional z-maps (pos/neg) with Neurosynth term maps.

    This function computes spatial correlations between group comparison maps
    (expert vs novice) and Neurosynth meta-analytic term association maps.
    By splitting the comparison map by sign, we can identify which cognitive
    terms are associated with expert-enhanced regions (Z+) versus expert-reduced
    regions (Z-).

    For each term map, compute:
    - r_pos: correlation between term map and Z+ (expert-enhanced regions)
    - r_neg: correlation between term map and Z- (expert-reduced regions)
    - r_diff: difference (r_pos - r_neg)

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

    Returns
    -------
    df_pos : pd.DataFrame
        Columns: term, r (correlation with expert-enhanced regions)
    df_neg : pd.DataFrame
        Columns: term, r (correlation with expert-reduced regions)
    df_diff : pd.DataFrame
        Columns: term, r_pos, r_neg, r_diff (difference score)

    Notes
    -----
    - Gray matter masking ensures correlation is computed only within brain
    - Voxel cleaning removes NaNs and zero-variance voxels
    - All term maps are resampled to reference image geometry before correlation

    Examples
    --------
    >>> # After computing z-map from group comparison
    >>> z_pos, z_neg = split_zmap_by_sign(z_map)
    >>> term_maps = load_term_maps('neurosynth_maps/')
    >>> df_pos, df_neg, df_diff = compute_all_zmap_correlations(
    ...     z_pos, z_neg, term_maps, ref_img
    ... )
    >>> # Find terms most associated with expert enhancements
    >>> top_pos = df_pos.nlargest(5, 'r')
    >>> # Find terms with largest difference
    >>> top_diff = df_diff.nlargest(5, 'r_diff')
    """
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
        # This ensures we only correlate across valid brain voxels with non-zero values.
        stacked = np.vstack([flat_pos, flat_neg, flat_t])
        stacked_clean, keep = clean_voxels(
            stacked, brain_mask_flat=mask_flat, var_thresh=1e-5
        )
        x, y, t = stacked_clean

        # Compute Pearson correlations between term map and Z+/Z− maps
        # np.corrcoef returns 2×2 correlation matrix; [0,1] extracts r between vectors
        r_pos = float(np.corrcoef(t, x)[0, 1])
        r_neg = float(np.corrcoef(t, y)[0, 1])

        # Compute difference: r_pos - r_neg
        # Positive r_diff means term is more associated with expert-enhanced regions
        # Negative r_diff means term is more associated with expert-reduced regions
        r_diff = r_pos - r_neg

        rec_pos.append((term, r_pos))
        rec_neg.append((term, r_neg))
        rec_diff.append((term, r_pos, r_neg, r_diff))

    df_pos = pd.DataFrame(rec_pos, columns=['term', 'r'])
    df_neg = pd.DataFrame(rec_neg, columns=['term', 'r'])
    df_diff = pd.DataFrame(rec_diff, columns=['term', 'r_pos', 'r_neg', 'r_diff'])

    return df_pos, df_neg, df_diff

