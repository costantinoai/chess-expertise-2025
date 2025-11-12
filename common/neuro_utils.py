"""
Neuroimaging utilities for loading and manipulating brain data.

This module provides common functions for working with NIfTI files, ROI masks,
GLM beta images, and other neuroimaging data structures.

Dependencies
------------
- nibabel: For NIfTI file I/O
- numpy: For array operations
- scipy: For resampling and interpolation
"""

import numpy as np
import nibabel as nib
from nibabel.freesurfer import io as fsio
from pathlib import Path
from typing import Union, Tuple, Dict, Optional
from .bids_utils import load_roi_metadata
import logging
import pandas as pd
from .constants import CONFIG

logger = logging.getLogger(__name__)


def load_nifti(file_path):
    """
    Load a NIfTI file.

    Parameters
    ----------
    file_path : str or Path
        Path to NIfTI file (.nii or .nii.gz)

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        Loaded NIfTI image object

    Raises
    ------
    FileNotFoundError
        If file does not exist

    Example
    -------
    >>> img = load_nifti('/path/to/beta_0001.nii.gz')
    >>> data = img.get_fdata()
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    return nib.load(str(file_path))


def load_roi_mask(roi_path, threshold=0.5):
    """
    Load an ROI mask and return binary mask array.

    Parameters
    ----------
    roi_path : str or Path
        Path to ROI NIfTI file
    threshold : float, default=0.5
        Threshold for binarizing mask (values > threshold become 1)

    Returns
    -------
    tuple of (np.ndarray, nibabel.nifti1.Nifti1Image)
        - Binary mask array (3D boolean)
        - Original NIfTI image object

    Example
    -------
    >>> mask_data, mask_img = load_roi_mask('rois/motor_cortex.nii.gz')
    >>> n_voxels = np.sum(mask_data)
    """
    img = load_nifti(roi_path)
    data = img.get_fdata()

    # Binarize mask
    mask = data > threshold

    return mask, img


def get_roi_names_and_colors(roi_dir: Path):
    """
    Return ROI names and colors from an ROI directory.

    Parameters
    ----------
    roi_dir : Path
        Path to directory containing ROI metadata TSV (roi_labels.tsv or region_info.tsv)

    Returns
    -------
    (names, colors) : (list[str], list[str])
        ROI display names and their colors (hex). If color column is missing,
        a default color is repeated.
    """
    meta = load_roi_metadata(Path(roi_dir))
    names = meta['roi_name'].tolist() if 'roi_name' in meta.columns else meta.iloc[:, 1].astype(str).tolist()
    if 'color' in meta.columns:
        colors = meta['color'].tolist()
    else:
        colors = ['#4C78A8'] * len(names)
    return names, colors


def extract_roi_data(beta_imgs, roi_mask):
    """
    Extract voxel data from ROI mask across multiple beta images.

    Parameters
    ----------
    beta_imgs : list of nibabel.nifti1.Nifti1Image
        List of beta images (one per condition/trial)
    roi_mask : np.ndarray
        Binary ROI mask (3D boolean array)

    Returns
    -------
    np.ndarray
        Data matrix of shape (n_conditions, n_voxels_in_roi)

    Notes
    -----
    - All beta images must have same dimensions as ROI mask
    - Voxels outside ROI are excluded
    - Useful for MVPA/RSA analyses

    Example
    -------
    >>> beta_imgs = [load_nifti(f'beta_{i:04d}.nii.gz') for i in range(1, 41)]
    >>> mask_data, _ = load_roi_mask('rois/motor.nii.gz')
    >>> roi_data = extract_roi_data(beta_imgs, mask_data)
    >>> # roi_data shape: (40 conditions, n_voxels)
    """
    n_conditions = len(beta_imgs)
    n_voxels = np.sum(roi_mask)

    # Initialize data matrix
    data_matrix = np.zeros((n_conditions, n_voxels))

    # Extract data from each beta image
    for i, img in enumerate(beta_imgs):
        beta_data = img.get_fdata()
        # Extract voxels within ROI
        data_matrix[i, :] = beta_data[roi_mask]

    return data_matrix


def mask_brain_data(data, mask):
    """
    Apply brain mask to data array.

    Parameters
    ----------
    data : np.ndarray
        Brain data (3D or 4D array)
    mask : np.ndarray
        Binary mask (3D boolean array)

    Returns
    -------
    np.ndarray
        Masked data (1D or 2D array)
        - If data is 3D: returns 1D array of voxels
        - If data is 4D: returns 2D array (time/conditions × voxels)

    Example
    -------
    >>> # 3D case
    >>> brain_data = img.get_fdata()  # shape: (x, y, z)
    >>> mask = load_roi_mask('roi.nii.gz')[0]
    >>> masked = mask_brain_data(brain_data, mask)  # shape: (n_voxels,)
    >>>
    >>> # 4D case
    >>> timeseries = img_4d.get_fdata()  # shape: (x, y, z, time)
    >>> masked = mask_brain_data(timeseries, mask)  # shape: (time, n_voxels)
    """
    if data.ndim == 3:
        # Single volume
        return data[mask]
    elif data.ndim == 4:
        # Multiple volumes (e.g., time series)
        n_vols = data.shape[3]
        n_voxels = np.sum(mask)
        masked_data = np.zeros((n_vols, n_voxels))
        for i in range(n_vols):
            masked_data[i, :] = data[:, :, :, i][mask]
        return masked_data
    else:
        raise ValueError(f"Data must be 3D or 4D, got shape {data.shape}")


def save_brain_map(data, output_path, reference_img):
    """
    Save brain data as NIfTI file using reference image for header.

    Parameters
    ----------
    data : np.ndarray
        Brain data to save (must match reference image dimensions)
    output_path : str or Path
        Output file path
    reference_img : nibabel.nifti1.Nifti1Image
        Reference image for header information (affine, etc.)

    Returns
    -------
    None

    Example
    -------
    >>> ref_img = load_nifti('mask.nii.gz')
    >>> result_data = np.random.randn(*ref_img.shape)
    >>> save_brain_map(result_data, 'output/result.nii.gz', ref_img)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create new NIfTI image with reference header
    new_img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)

    # Save to disk
    nib.save(new_img, str(output_path))


def reconstruct_brain_map(masked_data, mask, fill_value=0):
    """
    Reconstruct full brain volume from masked data.

    Parameters
    ----------
    masked_data : np.ndarray
        Masked data (1D array of voxel values)
    mask : np.ndarray
        Binary mask (3D boolean array)
    fill_value : float, default=0
        Value to use for voxels outside mask

    Returns
    -------
    np.ndarray
        Full 3D brain volume

    Example
    -------
    >>> mask_data, mask_img = load_roi_mask('roi.nii.gz')
    >>> roi_values = np.random.randn(np.sum(mask_data))
    >>> full_brain = reconstruct_brain_map(roi_values, mask_data)
    >>> save_brain_map(full_brain, 'output.nii.gz', mask_img)
    """
    # Initialize full volume with fill_value
    full_volume = np.full(mask.shape, fill_value, dtype=masked_data.dtype)

    # Fill in masked voxels
    full_volume[mask] = masked_data

    return full_volume


def get_roi_coordinates(mask):
    """
    Get voxel coordinates of ROI mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (3D boolean array)

    Returns
    -------
    np.ndarray
        Coordinates of voxels in ROI, shape (n_voxels, 3)

    Example
    -------
    >>> mask_data, _ = load_roi_mask('roi.nii.gz')
    >>> coords = get_roi_coordinates(mask_data)
    >>> print(f"ROI contains {len(coords)} voxels")
    """
    return np.array(np.where(mask)).T


def compute_roi_size(mask, affine=None):
    """
    Compute ROI size in voxels and mm³.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (3D boolean array)
    affine : np.ndarray, optional
        Affine transformation matrix. If provided, also returns volume in mm³.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'n_voxels': Number of voxels in ROI
        - 'volume_mm3': Volume in mm³ (if affine provided)

    Example
    -------
    >>> mask_data, mask_img = load_roi_mask('roi.nii.gz')
    >>> roi_size = compute_roi_size(mask_data, mask_img.affine)
    >>> print(f"ROI: {roi_size['n_voxels']} voxels, {roi_size['volume_mm3']:.1f} mm³")
    """
    n_voxels = np.sum(mask)

    result = {'n_voxels': int(n_voxels)}

    if affine is not None:
        # Compute voxel volume robustly from affine (handles rotations/shears)
        voxel_volume = abs(np.linalg.det(affine[:3, :3]))
        result['volume_mm3'] = float(n_voxels * voxel_volume)

    return result


def load_atlas(atlas_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load brain atlas NIfTI file and extract unique ROI labels.

    Parameters
    ----------
    atlas_path : str or Path
        Path to atlas NIfTI file

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - atlas_data: 3D int array with ROI labels
        - roi_labels: 1D array of unique non-zero ROI labels (sorted)

    Raises
    ------
    FileNotFoundError
        If atlas file does not exist
    ValueError
        If atlas contains no valid ROI labels

    Example
    -------
    >>> from common import CONFIG
    >>> atlas_data, roi_labels = load_atlas(CONFIG['ROI_GLASSER_22_ATLAS'])
    >>> print(f"Loaded {len(roi_labels)} ROIs")
    """
    atlas_path = Path(atlas_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    logger.info(f"Loading atlas: {atlas_path}")

    # Load atlas
    atlas_img = load_nifti(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Extract unique non-zero labels
    roi_labels = np.unique(atlas_data)
    roi_labels = roi_labels[roi_labels != 0]  # Remove background

    if len(roi_labels) == 0:
        raise ValueError(f"Atlas {atlas_path} contains no non-zero ROI labels")

    logger.info(f"Loaded atlas with {len(roi_labels)} ROI labels: {roi_labels}")

    return atlas_data, roi_labels


def get_gray_matter_mask(ref_img, prob_thresh: float = 0.5):
    """
    Build a gray-matter (GM) mask in the space of a reference image.

    Uses ICBM152 2009 GM probability map (via nilearn) resampled to `ref_img`.

    Parameters
    ----------
    ref_img : nibabel.Nifti1Image
        Reference image providing target shape, affine, and header.
    prob_thresh : float, default=0.5
        Probability threshold for binarization (retain voxels > prob_thresh).

    Returns
    -------
    nibabel.Nifti1Image
        Binary GM mask in reference space.
    """
    from nilearn import datasets, image

    mni = datasets.fetch_icbm152_2009()
    gm_img = image.load_img(mni["gm"])
    gm_res = image.resample_to_img(gm_img, ref_img, interpolation="nearest")
    mask_data = (gm_res.get_fdata() > float(prob_thresh)).astype(np.uint8)
    return nib.Nifti1Image(mask_data, ref_img.affine, ref_img.header)


def fisher_z_transform(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Apply Fisher r→z transform voxelwise to a correlation map.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input correlation image with values in [-1, 1].

    Returns
    -------
    nibabel.Nifti1Image
        Image with arctanh applied voxelwise.
    """
    data = img.get_fdata()
    z = np.arctanh(data)
    return nib.Nifti1Image(z, img.affine, img.header)


def clean_voxels(
    data: np.ndarray,
    brain_mask_flat: np.ndarray | None = None,
    var_thresh: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove unusable voxels from stacked maps (rows=maps, cols=voxels).

    - Drops columns with any non-finite value
    - Drops columns with variance below `var_thresh`
    - Optionally ANDs with a provided flat brain mask

    Parameters
    ----------
    data : np.ndarray, shape (n_maps, n_voxels)
    brain_mask_flat : np.ndarray of bool, optional
    var_thresh : float, default=1e-5

    Returns
    -------
    (np.ndarray, np.ndarray)
        Cleaned data and boolean keep-mask of voxels.
    """
    if data.ndim != 2:
        raise ValueError("clean_voxels expects a 2D array (maps × voxels)")

    n_vox = data.shape[1]
    finite_mask = np.all(np.isfinite(data), axis=0)
    variance = np.var(data, axis=0)
    keep_mask = finite_mask & (variance >= float(var_thresh))

    if brain_mask_flat is not None:
        if brain_mask_flat.shape[0] != n_vox:
            raise ValueError("brain_mask_flat must have shape (n_voxels,)")
        keep_mask &= brain_mask_flat.astype(bool)

    return data[:, keep_mask], keep_mask


def map_glasser_roi_to_harvard_oxford(
    roi_label: int,
    glasser_atlas_img: nib.Nifti1Image,
    threshold: float = 0.25,
) -> str:
    """
    Map a Glasser atlas ROI to its Harvard-Oxford anatomical label via center of mass.

    This function computes the center of mass of a Glasser ROI in MNI space, then
    looks up the corresponding Harvard-Oxford cortical atlas label at that coordinate.
    This provides familiar anatomical terminology to complement Glasser's fine-grained
    parcellation.

    Procedure:
    1. Extract all voxels belonging to the specified Glasser ROI
    2. Compute center of mass in voxel space
    3. Transform center of mass to MNI coordinates
    4. Look up Harvard-Oxford label at that MNI coordinate

    Parameters
    ----------
    roi_label : int
        Glasser ROI label/index to map (from atlas intensity values)
    glasser_atlas_img : nibabel.Nifti1Image
        Glasser atlas image in MNI space
    threshold : float, default=0.25
        Probability threshold for Harvard-Oxford atlas (0-1). Standard values are
        0.25 (default, more inclusive) or 0.50 (more conservative).

    Returns
    -------
    str
        Harvard-Oxford anatomical label at the ROI's center of mass, or:
        - "No ROI voxels" if ROI label not found in atlas
        - "Invalid CoM" if center of mass computation fails
        - "Unlabeled" if center of mass falls outside Harvard-Oxford parcellation
        - "Unknown label X" if label index is out of range

    Notes
    -----
    - Uses nilearn.datasets.fetch_atlas_harvard_oxford to retrieve the atlas
    - Harvard-Oxford labels are based on probabilistic maps thresholded at the
      specified probability (default 25%)
    - The center of mass provides a single representative coordinate for each ROI,
      which is then mapped to the nearest Harvard-Oxford region

    Examples
    --------
    >>> glasser_img = nib.load('Glasser_MNI_atlas.nii.gz')
    >>> ho_label = map_glasser_roi_to_harvard_oxford(roi_label=42, glasser_atlas_img=glasser_img)
    >>> print(f"Glasser ROI 42 maps to: {ho_label}")
    Glasser ROI 42 maps to: Inferior Parietal Lobule

    References
    ----------
    - Glasser et al. (2016). A multi-modal parcellation of human cerebral cortex.
      Nature, 536(7615), 171-178.
    - Harvard-Oxford atlases: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
    """
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import coord_transform
    from scipy.ndimage import center_of_mass

    try:
        # Step 1: Extract Glasser ROI voxels
        glasser_data = glasser_atlas_img.get_fdata()
        roi_mask = (glasser_data == roi_label)

        if not np.any(roi_mask):
            return "No ROI voxels"

        # Step 2: Compute center of mass in voxel space
        com_voxel = center_of_mass(roi_mask)

        if np.any(np.isnan(com_voxel)):
            return "Invalid CoM"

        # Step 3: Convert center of mass to MNI space
        com_mni = coord_transform(*com_voxel, glasser_atlas_img.affine)

        # Step 4: Fetch Harvard-Oxford atlas
        threshold_pct = int(threshold * 100)
        atlas_ho = fetch_atlas_harvard_oxford(f'cort-maxprob-thr{threshold_pct}-2mm')
        # atlas_ho.maps is already a Nifti1Image object, not a file path
        atlas_img_ho = atlas_ho.maps
        atlas_data_ho = atlas_img_ho.get_fdata()
        atlas_labels_ho = atlas_ho.labels

        # Step 5: Convert MNI coordinate to voxel space of Harvard-Oxford atlas
        inv_affine_ho = np.linalg.inv(atlas_img_ho.affine)
        com_voxel_ho = coord_transform(*com_mni, inv_affine_ho)

        # Round and clip to valid voxel indices
        x, y, z = [int(round(v)) for v in com_voxel_ho]
        x = np.clip(x, 0, atlas_data_ho.shape[0] - 1)
        y = np.clip(y, 0, atlas_data_ho.shape[1] - 1)
        z = np.clip(z, 0, atlas_data_ho.shape[2] - 1)

        # Step 6: Look up Harvard-Oxford label at this voxel
        label_idx = int(atlas_data_ho[x, y, z])

        if label_idx == 0:
            return "Unlabeled"

        if label_idx >= len(atlas_labels_ho):
            return f"Unknown label {label_idx}"

        return atlas_labels_ho[label_idx]

    except Exception as e:
        logger.warning(f"Failed to map Glasser ROI {roi_label} to Harvard-Oxford: {e}")
        return f"Error: {str(e)}"


__all__ = [
    'load_nifti',
    'load_roi_mask',
    'extract_roi_data',
    'mask_brain_data',
    'save_brain_map',
    'reconstruct_brain_map',
    'get_roi_coordinates',
    'compute_roi_size',
    'load_atlas',
    'get_gray_matter_mask',
    'fisher_z_transform',
    'clean_voxels',
    'compute_roi_means',
    'fill_atlas_with_values',
    'compute_surface_roi_means',
    'roi_values_to_surface_texture',
    'map_glasser_roi_to_harvard_oxford',
    'load_glasser180_annotations',
    'load_glasser180_region_info',
    'map_180_to_22',
    'expand_roi22_to_roi180_values',
    'project_volume_to_surfaces',
]

def compute_roi_means(
    data_3d: np.ndarray,
    atlas_data: np.ndarray,
    roi_labels: np.ndarray,
    *,
    allow_nan: bool = True,
) -> np.ndarray:
    """
    Compute per-ROI mean values from a 3D image using an integer-labeled atlas.

    Parameters
    ----------
    data_3d : np.ndarray
        3D array of voxel values (e.g., contrast map or RSA r-map).
    atlas_data : np.ndarray
        3D integer array with same shape as data_3d; each non-zero value is an ROI id.
    roi_labels : np.ndarray
        1D array of ROI ids to extract (e.g., from load_atlas()).
    allow_nan : bool, default=True
        If True, compute NaN-safe means (np.nanmean). If False, use np.mean.

    Returns
    -------
    np.ndarray
        1D array of length len(roi_labels) with mean value per ROI. If an ROI
        has no voxels in the atlas, returns np.nan for that entry.

    Notes
    -----
    - Shapes of data_3d and atlas_data must match exactly.
    - ROI ids in roi_labels must be present in atlas_data; missing ids yield NaN.
    """
    if data_3d.shape != atlas_data.shape:
        raise ValueError(
            f"data_3d shape {data_3d.shape} does not match atlas shape {atlas_data.shape}"
        )

    reducer = np.nanmean if allow_nan else np.mean

    # Flatten once for efficient masking
    data_flat = data_3d.reshape(-1)
    atlas_flat = atlas_data.reshape(-1)

    out = np.full(len(roi_labels), np.nan, dtype=float)
    for i, roi_id in enumerate(roi_labels):
        mask = (atlas_flat == int(roi_id))
        if not np.any(mask):
            out[i] = np.nan
            continue
        vals = data_flat[mask]
        # Handle the edge-case of all-NaN voxels
        if allow_nan:
            if np.all(np.isnan(vals)):
                out[i] = np.nan
            else:
                out[i] = float(reducer(vals))
        else:
            out[i] = float(reducer(vals))
    return out


def fill_atlas_with_values(
    atlas_img: nib.Nifti1Image,
    roi_labels: np.ndarray,
    roi_values: np.ndarray,
    *,
    include_mask: np.ndarray | None = None,
    outside_value: float | None = 0.0,
) -> nib.Nifti1Image:
    """
    Create a NIfTI volume by assigning per-ROI values across an atlas image.

    Parameters
    ----------
    atlas_img : nib.Nifti1Image
        Integer-labeled atlas image; non-zero voxels correspond to ROI ids.
    roi_labels : np.ndarray
        1D array of ROI ids (ints) corresponding to roi_values.
    roi_values : np.ndarray
        1D array of values (floats), same length/order as roi_labels.
    include_mask : np.ndarray of bool, optional
        If provided, length must match roi_labels; only ROIs with True will be
        assigned their value; others are set to outside_value or NaN.
    outside_value : float or None, default=0.0
        Value to assign to voxels not in included ROIs. If None, uses NaN.

    Returns
    -------
    nib.Nifti1Image
        New volume with per-ROI values assigned.
    """
    atlas_data = atlas_img.get_fdata().astype(int)
    vol = np.full(atlas_data.shape, np.nan if outside_value is None else outside_value, dtype=float)

    roi_labels = np.asarray(roi_labels).astype(int)
    roi_values = np.asarray(roi_values).astype(float)
    if roi_labels.shape[0] != roi_values.shape[0]:
        raise ValueError("roi_labels and roi_values must have same length")

    if include_mask is not None:
        include_mask = np.asarray(include_mask).astype(bool)
        if include_mask.shape[0] != roi_labels.shape[0]:
            raise ValueError("include_mask must have same length as roi_labels")
    else:
        include_mask = np.ones_like(roi_labels, dtype=bool)

    for roi_id, val, keep in zip(roi_labels, roi_values, include_mask):
        if not keep:
            continue
        if roi_id == 0:
            continue
        mask = (atlas_data == int(roi_id))
        if np.any(mask):
            vol[mask] = float(val)

    return nib.Nifti1Image(vol, atlas_img.affine, atlas_img.header)


def compute_surface_roi_means_single_hemi(
    texture: np.ndarray,
    labels: np.ndarray,
    roi_labels: np.ndarray,
    *,
    allow_nan: bool = True,
) -> np.ndarray:
    """
    Compute per-ROI mean from surface texture for a single hemisphere.

    Parameters
    ----------
    texture : np.ndarray
        Per-vertex values for one hemisphere (1D array).
    labels : np.ndarray
        Per-vertex integer labels from FreeSurfer .annot file (read via fsio.read_annot).
    roi_labels : np.ndarray
        ROI label ids (integers) to aggregate over; should match annotation ids.
    allow_nan : bool, default=True
        Use NaN-safe mean (np.nanmean) if True.

    Returns
    -------
    np.ndarray
        Mean value per ROI (length len(roi_labels)).
    """
    reducer = np.nanmean if allow_nan else np.mean
    out = np.full(len(roi_labels), np.nan, dtype=float)
    for i, rid in enumerate(np.asarray(roi_labels).astype(int)):
        mask = (labels == rid)
        if not np.any(mask):
            out[i] = np.nan
            continue
        vals = texture[mask]
        if allow_nan and np.all(np.isnan(vals)):
            out[i] = np.nan
        else:
            out[i] = float(reducer(vals))
    return out


def compute_surface_roi_means(
    tex_left: np.ndarray,
    labels_left: np.ndarray,
    tex_right: np.ndarray,
    labels_right: np.ndarray,
    roi_labels: np.ndarray,
    *,
    allow_nan: bool = True,
) -> np.ndarray:
    """
    Compute per-ROI mean from surface textures using fsaverage Glasser labels.

    Parameters
    ----------
    tex_left, tex_right : np.ndarray
        Per-vertex values for left and right hemispheres (1D arrays).
    labels_left, labels_right : np.ndarray
        Per-vertex integer labels from FreeSurfer .annot files (read via fsio.read_annot).
    roi_labels : np.ndarray
        ROI label ids (integers) to aggregate over; should match annotation ids.
    allow_nan : bool, default=True
        Use NaN-safe mean (np.nanmean) if True.

    Returns
    -------
    np.ndarray
        Mean value per ROI (length len(roi_labels)).
    """
    reducer = np.nanmean if allow_nan else np.mean
    out = np.full(len(roi_labels), np.nan, dtype=float)
    for i, rid in enumerate(np.asarray(roi_labels).astype(int)):
        mask_l = (labels_left == rid)
        mask_r = (labels_right == rid)
        vals = []
        if np.any(mask_l):
            vals.append(tex_left[mask_l])
        if np.any(mask_r):
            vals.append(tex_right[mask_r])
        if len(vals) == 0:
            out[i] = np.nan
            continue
        allv = np.concatenate(vals)
        if allow_nan and np.all(np.isnan(allv)):
            out[i] = np.nan
        else:
            out[i] = float(reducer(allv))
    return out


def roi_values_to_surface_texture(
    labels: np.ndarray,
    roi_labels: np.ndarray,
    roi_values: np.ndarray,
    *,
    include_mask: np.ndarray | None = None,
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Build a per-vertex texture by assigning per-ROI values using an .annot labels array.

    Parameters
    ----------
    labels : np.ndarray
        Per-vertex integer labels from FreeSurfer .annot file.
    roi_labels : np.ndarray
        ROI ids corresponding to roi_values; must be integers matching labels.
    roi_values : np.ndarray
        Values to assign per ROI.
    include_mask : np.ndarray of bool, optional
        If provided, mask of length len(roi_labels) indicating which ROIs to draw.
    default_value : float, default=0.0
        Value for vertices not belonging to included ROIs.

    Returns
    -------
    np.ndarray
        Per-vertex texture array.
    """
    roi_labels = np.asarray(roi_labels).astype(int)
    roi_values = np.asarray(roi_values).astype(float)
    if roi_labels.shape[0] != roi_values.shape[0]:
        raise ValueError("roi_labels and roi_values length mismatch")
    if include_mask is not None:
        include_mask = np.asarray(include_mask).astype(bool)
        if include_mask.shape[0] != roi_labels.shape[0]:
            raise ValueError("include_mask length mismatch")
    else:
        include_mask = np.ones_like(roi_labels, dtype=bool)

    tex = np.full(labels.shape[0], float(default_value), dtype=float)
    for rid, val, keep in zip(roi_labels, roi_values, include_mask):
        if not keep:
            continue
        mask = (labels == int(rid))
        if np.any(mask):
            tex[mask] = float(val)
    return tex


def load_glasser180_annotations(hemi: str) -> np.ndarray:
    """
    Load fsaverage Glasser-180 surface annotation labels for a hemisphere.

    Parameters
    ----------
    hemi : {'left','right'}
        Hemisphere to load.

    Returns
    -------
    np.ndarray
        Per-vertex integer labels as read by nibabel.freesurfer.io.read_annot.

    Raises
    ------
    KeyError
        If required CONFIG path is missing.
    FileNotFoundError
        If the annotation file does not exist.
    ValueError
        If `hemi` is not 'left' or 'right'.
    """
    if hemi not in {"left", "right"}:
        raise ValueError("hemi must be 'left' or 'right'")
    key = 'ROI_GLASSER_180_ANNOT_L' if hemi == 'left' else 'ROI_GLASSER_180_ANNOT_R'
    annot_path = CONFIG[key]
    if not Path(annot_path).exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")
    labels, _, _ = fsio.read_annot(annot_path)
    return labels


def load_glasser180_region_info() -> pd.DataFrame:
    """
    Load and normalize Glasser-180 region metadata.

    Expects a TSV at CONFIG['ROI_GLASSER_180']/region_info.tsv including a
    column that identifies ROI ids and a 'region22_id' column mapping each
    180 parcel to a corresponding 22-region label.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least columns: 'roi_id' (int), 'region22_id' (int).

    Raises
    ------
    FileNotFoundError
        If region_info.tsv is missing.
    RuntimeError
        If no usable ROI id column is found or 'region22_id' is missing.
    """
    info_path = Path(CONFIG['ROI_GLASSER_180']) / 'region_info.tsv'
    if not info_path.exists():
        raise FileNotFoundError(f"Glasser-180 region info not found: {info_path}")
    df = pd.read_csv(info_path, sep='\t')

    # Normalize ROI id column to 'roi_id'
    id_candidates = ['roi_id', 'ROI_idx', 'index', 'roi_idx']
    id_cols = [c for c in id_candidates if c in df.columns]
    if not id_cols:
        raise RuntimeError(
            f"No ROI id column found in {info_path}. Expected one of {id_candidates}"
        )
    id_col = id_cols[0]
    df = df.copy()
    df['roi_id'] = df[id_col].astype(int)

    if 'region22_id' not in df.columns:
        raise RuntimeError("'region22_id' column is required in Glasser-180 region info")
    df['region22_id'] = df['region22_id'].astype(int)
    return df


def map_180_to_22(roi180_df: pd.DataFrame, hemisphere: str = 'left') -> Dict[int, int]:
    """
    Build mapping from Glasser-180 parcel ids to Glasser-22 region ids.

    Parameters
    ----------
    roi180_df : pd.DataFrame
        Output of load_glasser180_region_info(). Must contain 'roi_id' and 'region22_id'.
    hemisphere : {'left','right','both'}, default 'left'
        Which hemisphere's parcels to include.

    Returns
    -------
    dict[int,int]
        Mapping of 180-parcel id -> 22-region id.

    Raises
    ------
    ValueError
        If hemisphere argument is invalid.
    """
    hemi = hemisphere.lower()
    if hemi not in {"left", "right", "both"}:
        raise ValueError("hemisphere must be 'left', 'right', or 'both'")

    df = roi180_df
    if hemi == 'left':
        sel = df['roi_id'] <= 180
    elif hemi == 'right':
        sel = df['roi_id'] > 180
    else:  # both
        sel = np.ones(len(df), dtype=bool)

    df_hemi = df.loc[sel, ['roi_id', 'region22_id']].copy()
    return dict(zip(df_hemi['roi_id'].astype(int), df_hemi['region22_id'].astype(int)))


def expand_roi22_to_roi180_values(
    roi22_ids: np.ndarray,
    roi22_values: np.ndarray,
    roi180_df: pd.DataFrame,
    hemisphere: str = 'left',
    include_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand per-22-region values into per-180-parcel values for a hemisphere.

    Parameters
    ----------
    roi22_ids : np.ndarray
        1D array of 22-region ids.
    roi22_values : np.ndarray
        1D array of values per 22-region id (same length as roi22_ids).
    roi180_df : pd.DataFrame
        Region info with 'roi_id' and 'region22_id' columns.
    hemisphere : {'left','right','both'}, default 'left'
        Hemisphere whose 180-parcel ids to generate.
    include_mask : np.ndarray, optional
        Boolean mask for roi22_ids/values indicating which 22-regions to include.
        Excluded regions will yield NaN in the expanded 180 values.

    Returns
    -------
    (roi_ids_180, roi_vals_180) : Tuple[np.ndarray, np.ndarray]
        Integer array of 180-parcel ids and float array of corresponding values.

    Raises
    ------
    ValueError
        On shape mismatch or invalid hemisphere.
    """
    roi22_ids = np.asarray(roi22_ids).astype(int)
    roi22_values = np.asarray(roi22_values).astype(float)
    if roi22_ids.shape[0] != roi22_values.shape[0]:
        raise ValueError("roi22_ids and roi22_values length mismatch")
    if include_mask is not None:
        include_mask = np.asarray(include_mask).astype(bool)
        if include_mask.shape[0] != roi22_ids.shape[0]:
            raise ValueError("include_mask length mismatch for 22-region arrays")
    else:
        include_mask = np.ones_like(roi22_ids, dtype=bool)

    mapping = map_180_to_22(roi180_df, hemisphere=hemisphere)
    # Build a dict from 22-region id -> value (NaN if excluded)
    region22_to_val = {
        int(r22): (float(val) if keep and np.isfinite(val) else np.nan)
        for r22, val, keep in zip(roi22_ids, roi22_values, include_mask)
    }

    roi_ids_180: list[int] = []
    roi_vals_180: list[float] = []
    for rid180, rid22 in mapping.items():
        roi_ids_180.append(int(rid180))
        roi_vals_180.append(region22_to_val.get(int(rid22), np.nan))
    return np.asarray(roi_ids_180, dtype=int), np.asarray(roi_vals_180, dtype=float)


def create_glasser22_contours(
    region_names: list[str],
    hemisphere: str = 'both',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create per-vertex contour labels for specified Glasser-22 regions.

    This function generates contour arrays suitable for use with surface plotting
    functions that support ROI boundary overlays (e.g., plot_flat_pair with
    contours_rois parameter).

    Parameters
    ----------
    region_names : list of str
        Region names or aliases to highlight. Supported names/aliases:
        - 'V1', 'primary_visual', 'Primary_Visual' (region22_id=1)
        - 'PCC', 'posterior_cingulate', 'Posterior_Cingulate' (region22_id=18)
        - 'dLPFC', 'dlpfc', 'dorsolateral_prefrontal', 'Dorsolateral_Prefrontal' (region22_id=22)
        Names are case-insensitive and support multiple aliases.
    hemisphere : {'left', 'right', 'both'}, default='both'
        Which hemispheres to process.

    Returns
    -------
    contours_left : np.ndarray
        Per-vertex integer labels for left hemisphere (same shape as fsaverage7).
        Value is region22_id if vertex belongs to a target region, 0 otherwise.
    contours_right : np.ndarray
        Per-vertex integer labels for right hemisphere.

    Raises
    ------
    ValueError
        If hemisphere is invalid or region_names contains unrecognized names.
    FileNotFoundError
        If required annotation files are missing.

    Examples
    --------
    >>> from common.neuro_utils import create_glasser22_contours
    >>> from common.plotting import plot_flat_pair
    >>>
    >>> # Create contours for multiple regions
    >>> contours_l, contours_r = create_glasser22_contours(
    ...     ['dLPFC', 'posterior_cingulate', 'V1']
    ... )
    >>> # Use with plot_flat_pair
    >>> fig = plot_flat_pair(
    ...     textures=(tex_l, tex_r),
    ...     contours_rois=(contours_l, contours_r),
    ...     vmin=vmin, vmax=vmax
    ... )

    See Also
    --------
    load_glasser180_annotations : Load per-vertex Glasser-180 labels
    map_180_to_22 : Map Glasser-180 parcels to 22 regions
    """
    # Define region name aliases mapping to region22_id
    REGION_ALIASES = {
        # Primary Visual (region22_id=1)
        'v1': 1,
        'primary_visual': 1,
        'primary visual': 1,
        # Ventral Stream Visual (region22_id=4)
        'vvs': 4,
        'ventral_stream_visual': 4,
        'ventral stream visual': 4,
        # Temporo-Parieto-Occipital Junction (region22_id=15)
        'tpoj': 15,
        'temporo parieto occipital junction': 15,
        'temporo-parieto-occipital junction': 15,
        # Superior Parietal (region22_id=16)
        'sp': 16,
        'superior_parietal': 16,
        'superior parietal': 16,
        # Posterior Cingulate (region22_id=18)
        'pcc': 18,
        'posterior_cingulate': 18,
        'posterior cingulate': 18,
        # Dorsolateral Prefrontal (region22_id=22)
        'dlpfc': 22,
        'dorsolateral_prefrontal': 22,
        'dorsolateral prefrontal': 22,
    }

    # Normalize region names to region22_ids
    target_region22_ids = set()
    for name in region_names:
        name_normalized = name.lower().replace('-', ' ').replace('_', ' ')
        if name_normalized not in REGION_ALIASES:
            raise ValueError(
                f"Unknown region name: '{name}'. Supported names: "
                f"{list(set(REGION_ALIASES.keys()))}"
            )
        target_region22_ids.add(REGION_ALIASES[name_normalized])

    # Load Glasser-180 to Glasser-22 mapping
    roi180_df = load_glasser180_region_info()

    # Process hemispheres
    process_left = hemisphere in {'left', 'both'}
    process_right = hemisphere in {'right', 'both'}

    if not (process_left or process_right):
        raise ValueError("hemisphere must be 'left', 'right', or 'both'")

    # Initialize output arrays (will be resized to match annotation length)
    contours_left = np.array([])
    contours_right = np.array([])

    if process_left:
        # Load left hemisphere annotations
        labels_left = load_glasser180_annotations('left')
        contours_left = np.zeros(len(labels_left), dtype=int)

        # Build 180->22 mapping for left hemisphere
        mapping_left = map_180_to_22(roi180_df, hemisphere='left')

        # Assign region22_id to vertices belonging to target regions
        for vertex_idx, label180 in enumerate(labels_left):
            if label180 in mapping_left:
                region22_id = mapping_left[label180]
                if region22_id in target_region22_ids:
                    contours_left[vertex_idx] = region22_id

    if process_right:
        # Load right hemisphere annotations
        labels_right = load_glasser180_annotations('right')
        contours_right = np.zeros(len(labels_right), dtype=int)

        # Build 180->22 mapping for right hemisphere
        # Note: annotation files use 1-180 for both hemispheres, but region_info.tsv
        # uses 181-360 for right hemisphere. Offset by 180 to match.
        mapping_right = map_180_to_22(roi180_df, hemisphere='right')

        # Assign region22_id to vertices belonging to target regions
        for vertex_idx, label180 in enumerate(labels_right):
            # Offset label by 180 for right hemisphere to match region_info.tsv numbering
            label180_offset = label180 + 180 if label180 > 0 else 0
            if label180_offset in mapping_right:
                region22_id = mapping_right[label180_offset]
                if region22_id in target_region22_ids:
                    contours_right[vertex_idx] = region22_id

    return contours_left, contours_right


def project_volume_to_surfaces(
    volume_img,
    surfaces: Tuple[str, ...] = ('pial_left', 'pial_right'),
) -> Tuple[np.ndarray, ...]:
    """
    Project NIfTI volume to fsaverage surface textures.

    Centralizes the volume→surface projection logic for consistent handling
    across visualization workflows. Callers can then use the textures for
    range computation and plotting without re-projecting.

    Parameters
    ----------
    volume_img : NIfTI-like
        Volumetric image to project (e.g., statistical map, activation map).
    surfaces : tuple of str, default=('pial_left', 'pial_right')
        Surface names from fsaverage to project onto.
        Valid names: 'pial_left', 'pial_right', 'infl_left', 'infl_right', etc.

    Returns
    -------
    textures : tuple of np.ndarray
        Per-vertex textures for each requested surface, in the same order as input.

    Raises
    ------
    AttributeError
        If a requested surface name is not available in fsaverage.

    Examples
    --------
    >>> # Project once, use for both range computation and plotting
    >>> from nilearn import image
    >>> from common.neuro_utils import project_volume_to_surfaces
    >>> from common.plotting import compute_ylim_range, plot_flat_pair
    >>>
    >>> nifti_vol = image.load_img('zmap.nii.gz')
    >>> tex_l, tex_r = project_volume_to_surfaces(nifti_vol)
    >>>
    >>> # Compute symmetric range from projected textures
    >>> vmin, vmax = compute_ylim_range(tex_l, tex_r, symmetric=True, padding_pct=0.0)
    >>>
    >>> # Plot using pre-computed textures (no re-projection!)
    >>> fig = plot_flat_pair(
    ...     textures=(tex_l, tex_r),
    ...     title='Statistical Map',
    ...     vmin=vmin,
    ...     vmax=vmax,
    ...     show_colorbar=True
    ... )

    See Also
    --------
    common.plotting.compute_ylim_range : Compute symmetric or zero-anchored ranges
    common.plotting.plot_flat_pair : Plot flat surface maps from textures
    common.plotting.plot_pial_views_triplet : Plot pial views from textures
    """
    from nilearn import surface, datasets

    fsavg = datasets.fetch_surf_fsaverage('fsaverage7')

    textures = []
    for surf_name in surfaces:
        # Get surface mesh from fsaverage
        try:
            surf_mesh = getattr(fsavg, surf_name)
        except AttributeError as e:
            raise AttributeError(
                f"Surface '{surf_name}' not found in fsaverage. "
                f"Available surfaces: pial_left, pial_right, infl_left, infl_right, etc."
            ) from e

        # Project volume to surface
        tex = surface.vol_to_surf(volume_img, surf_mesh)
        textures.append(tex)

    return tuple(textures)
