"""
SPM I/O utilities for loading GLM beta images and extracting ROI data.

This module provides functions for:
- Loading SPM.mat files
- Extracting condition-specific beta images
- Averaging betas across runs
- Extracting voxel data from ROIs
"""

import re
import numpy as np
import nibabel as nib
import scipy.io as sio
from pathlib import Path
from typing import Dict, Sequence, Optional
import logging

logger = logging.getLogger(__name__)


def load_spm_beta_images(
    subject_id: str,
    glm_dir: Path,
    spm_filename: str = "SPM.mat"
) -> Dict[str, nib.Nifti1Image]:
    """
    Load SPM.mat for one subject and return averaged beta image per condition.

    Identifies condition-specific beta images (e.g., 'C1', 'C2', ..., 'C40')
    across multiple runs and averages them into a single image per condition.

    Parameters
    ----------
    subject_id : str
        Subject code (e.g., '03')
    glm_dir : Path
        Base directory containing GLM results (e.g., CONFIG['SPM_GLM_UNSMOOTHED'])
    spm_filename : str, default='SPM.mat'
        Name of the SPM file

    Returns
    -------
    averaged_betas : dict
        Mapping from condition label (e.g., 'C1') to averaged beta NIfTI image

    Raises
    ------
    FileNotFoundError
        If SPM.mat file is not found
    AttributeError
        If expected fields are missing from SPM structure

    Example
    -------
    >>> from common import CONFIG
    >>> betas = load_spm_beta_images('sub-03', CONFIG['SPM_GLM_UNSMOOTHED'])
    >>> print(list(betas.keys()))  # ['C1', 'C2', ..., 'C40']
    """
    # Normalize path input to support str/Path
    glm_dir = Path(glm_dir)
    # Normalize subject id to 'sub-XX' form
    subject_id_norm = _normalize_subject_id(subject_id)
    # Construct path to SPM.mat
    spm_mat_path = glm_dir / subject_id_norm / "exp" / spm_filename

    if not spm_mat_path.is_file():
        raise FileNotFoundError(f"SPM.mat not found: {spm_mat_path}")

    logger.debug(f"Loading SPM.mat from: {spm_mat_path}")

    # Load SPM structure
    spm_dict = sio.loadmat(spm_mat_path.as_posix(), struct_as_record=False, squeeze_me=True)
    SPM = spm_dict["SPM"]

    beta_info = SPM.Vbeta
    regressor_names: Sequence[str] = SPM.xX.name

    # Regex to match condition names: "Sn(1) C1*bf(1)" -> "C1"
    pattern = re.compile(r"Sn\(\d+\)\s+(.*?)\*bf\(1\)")

    # Map condition -> beta indices (across runs)
    condition_to_indices: Dict[str, list] = {}
    for i, reg_name in enumerate(regressor_names):
        m = pattern.match(reg_name)
        if m:
            cond = m.group(1)
            condition_to_indices.setdefault(cond, []).append(i)

    # Get SPM working directory
    spm_dir = _get_spm_dir(SPM, spm_mat_path.parent)

    # Average beta images per condition
    averaged: Dict[str, nib.Nifti1Image] = {}

    for cond, idxs in condition_to_indices.items():
        if not idxs:
            continue

        sum_data: Optional[np.ndarray] = None
        affine = header = None

        for idx in idxs:
            # Get beta filename (handle different SPM versions)
            beta_fname = _get_beta_filename(beta_info[idx])
            beta_path = spm_dir / beta_fname

            # Load beta image
            img = nib.load(beta_path.as_posix())
            data = img.get_fdata(dtype=np.float32)

            # Initialize sum on first iteration
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine, header = img.affine, img.header

            sum_data += data

        # Average across runs
        assert sum_data is not None
        avg = sum_data / float(len(idxs))
        averaged[cond] = nib.Nifti1Image(avg, affine=affine, header=header)

    logger.info(f"[Subject {subject_id}] Loaded {len(averaged)} conditions")
    return averaged


def extract_roi_voxel_matrices(
    subject_id: str,
    atlas_data: np.ndarray,
    roi_labels: np.ndarray,
    glm_dir: Path
) -> Dict[int, np.ndarray]:
    """
    Extract (conditions × voxels) matrix per ROI for one subject.

    Steps:
      1) Load averaged betas per condition (via load_spm_beta_images)
      2) For each ROI, gather beta values at ROI voxels for every condition

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '03')
    atlas_data : np.ndarray
        3D array containing integer ROI labels for each voxel
    roi_labels : np.ndarray
        The unique ROI labels to extract (non-zero)
    glm_dir : Path
        Base directory containing GLM results

    Returns
    -------
    roi_data : dict
        roi_label (int) -> array of shape (n_conditions, n_voxels_in_roi)

    Raises
    ------
    ValueError
        If an ROI has 0 voxels in the atlas

    Example
    -------
    >>> atlas_img = nib.load(ATLAS_GLASSER_22_BILATERAL)
    >>> atlas_data = atlas_img.get_fdata().astype(int)
    >>> roi_labels = np.unique(atlas_data)[1:]  # Exclude 0
    >>> from common import CONFIG
    >>> roi_mats = extract_roi_voxel_matrices('sub-03', atlas_data, roi_labels, CONFIG['SPM_GLM_UNSMOOTHED'])
    >>> print(roi_mats[1].shape)  # (40, n_voxels)
    """
    logger.info(f"[Subject {subject_id}] Extracting ROI voxel matrices...")

    # Load averaged beta images
    averaged_betas = load_spm_beta_images(subject_id, Path(glm_dir))
    conditions = sorted(averaged_betas.keys())
    n_conditions = len(conditions)

    logger.debug(f"[Subject {subject_id}] Found {n_conditions} conditions")

    # Initialize storage
    roi_data: Dict[int, np.ndarray] = {}

    for roi_label in roi_labels:
        # Create mask for this ROI
        mask = atlas_data == roi_label
        n_vox = int(mask.sum())

        if n_vox == 0:
            raise ValueError(f"ROI {roi_label} has 0 voxels in atlas.")

        # Initialize matrix for this ROI
        mat = np.zeros((n_conditions, n_vox), dtype=np.float32)

        # Fill matrix with beta values
        for ci, cname in enumerate(conditions):
            beta_vals = averaged_betas[cname].get_fdata()
            mat[ci, :] = beta_vals[mask]

        roi_data[int(roi_label)] = mat

    logger.info(f"[Subject {subject_id}] ROI extraction complete")
    return roi_data


def _get_spm_dir(spm_obj, fallback: Path) -> Path:
    """Extract SPM working directory robustly across SPM versions."""
    swd = getattr(spm_obj, "swd", None)
    return Path(swd) if swd else fallback


def _get_beta_filename(beta_struct) -> str:
    """
    Get beta filename from SPM Vbeta structure.

    Handles different SPM versions which may have 'fname' or 'filename' field.

    Parameters
    ----------
    beta_struct : object
        SPM Vbeta structure entry

    Returns
    -------
    filename : str
        Beta image filename

    Raises
    ------
    AttributeError
        If neither 'fname' nor 'filename' field is found
    """
    # Try 'fname' first (most common)
    fname = getattr(beta_struct, "fname", None)
    if fname is not None:
        return fname

    # Try 'filename' (alternative)
    fname = getattr(beta_struct, "filename", None)
    if fname is not None:
        return fname

    # Neither field found
    available = [f for f in dir(beta_struct) if not f.startswith("_")]
    raise AttributeError(
        f"Beta structure has no 'fname' or 'filename' field. "
        f"Available fields: {available}"
    )


def validate_atlas_roi_coverage(
    atlas_data: np.ndarray,
    roi_labels: np.ndarray,
    min_voxels: int = 10
) -> Dict[int, int]:
    """
    Validate that ROIs have sufficient voxel coverage in the atlas.

    Parameters
    ----------
    atlas_data : np.ndarray
        3D atlas array with ROI labels
    roi_labels : np.ndarray
        ROI labels to check
    min_voxels : int, default=10
        Minimum number of voxels required per ROI

    Returns
    -------
    voxel_counts : dict
        roi_label -> number of voxels

    Warns
    -----
    If any ROI has fewer than min_voxels

    Example
    -------
    >>> counts = validate_atlas_roi_coverage(atlas_data, roi_labels, min_voxels=10)
    """
    voxel_counts = {}

    for roi in roi_labels:
        n_vox = int((atlas_data == roi).sum())
        voxel_counts[int(roi)] = n_vox

        if n_vox < min_voxels:
            logger.warning(
                f"ROI {roi} has only {n_vox} voxels (< {min_voxels} minimum)"
            )

    return voxel_counts


def _normalize_subject_id(subject_id: str) -> str:
    """Normalize subject identifier to BIDS style 'sub-XX'.

    Accepts '03', '3', or 'sub-03' and returns 'sub-03'. For non-numeric ids,
    prefixes with 'sub-'.
    """
    s = str(subject_id)
    if s.startswith('sub-'):
        return s
    # Extract digits if present and pad to 2
    digits = ''.join(ch for ch in s if ch.isdigit())
    if digits:
        return f"sub-{int(digits):02d}"
    return f"sub-{s}"


__all__ = [
    'load_spm_beta_images',
    'extract_roi_voxel_matrices',
    'validate_atlas_roi_coverage',
]
