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
    # Take absolute value for two-tailed testing (we want magnitude, not direction yet)
    t_abs = np.abs(t_map)

    # Compute two-tailed p-value: P(|T| > |t_obs|) = 2 * P(T > |t_obs|)
    # sf() is the survival function: P(X > x) = 1 - CDF(x), more accurate for tail probabilities
    p_two = 2 * _t.sf(t_abs, df=int(dof))

    # Convert two-tailed p-value back to z-score using inverse survival function.
    # isf(p/2) gives the z-value where P(|Z| > z) = p (two-tailed)
    # This standardizes effect sizes from t-distribution to standard normal distribution.
    z_abs = _norm.isf(p_two / 2)

    # Restore original sign: positive t → positive z, negative t → negative z
    z = np.sign(t_map) * z_abs
    z[t_abs == 0] = 0.0  # Handle exactly zero values explicitly
    return z


def split_zmap_by_sign(z_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a z-map into positive and negative magnitude maps.
    Returns (z_pos, z_neg) where z_neg is positive magnitude of negative voxels.
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

    For each term map, compute:
    - r_pos: correlation between term map and Z+ (expert-enhanced regions)
    - r_neg: correlation between term map and Z− (expert-reduced regions)
    - r_diff: difference (r_pos - r_neg)

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
        Columns: term, r
    df_neg : pd.DataFrame
        Columns: term, r
    df_diff : pd.DataFrame
        Columns: term, r_pos, r_neg, r_diff
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

