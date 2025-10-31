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
from typing import List, Union, Tuple
from .bids_utils import find_beta_images, load_roi_metadata
import logging

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
    >>> img = load_nifti('/path/to/beta_0001.nii')
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
    >>> mask_data, mask_img = load_roi_mask('rois/motor_cortex.nii')
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
    >>> beta_imgs = [load_nifti(f'beta_{i:04d}.nii') for i in range(1, 41)]
    >>> mask_data, _ = load_roi_mask('rois/motor.nii')
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


def load_glm_betas(subject_id, bids_glm_path, n_betas=None):
    """
    Load beta images for a subject from GLM directory.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-01')
    bids_glm_path : str or Path
        Path to BIDS GLM derivatives folder
    n_betas : int, optional
        Number of beta images to load. If None, loads all beta_*.nii files.

    Returns
    -------
    list of nibabel.nifti1.Nifti1Image
        List of beta images

    Raises
    ------
    FileNotFoundError
        If subject directory or beta images not found

    Example
    -------
    >>> from common import CONFIG
    >>> betas = load_glm_betas('sub-01', CONFIG['BIDS_GLM_UNSMOOTHED'], n_betas=40)
    >>> len(betas)
    40
    """
    # Discover beta files via BIDS utilities for consistency
    beta_files = find_beta_images(subject_id, bids_deriv_path=bids_glm_path, pattern='beta_*.nii')

    # Limit to n_betas if specified
    if n_betas is not None:
        beta_files = beta_files[:n_betas]

    # Load beta images
    beta_imgs = [load_nifti(f) for f in beta_files]

    return beta_imgs


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
    >>> mask = load_roi_mask('roi.nii')[0]
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
    >>> ref_img = load_nifti('mask.nii')
    >>> result_data = np.random.randn(*ref_img.shape)
    >>> save_brain_map(result_data, 'output/result.nii', ref_img)
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
    >>> mask_data, mask_img = load_roi_mask('roi.nii')
    >>> roi_values = np.random.randn(np.sum(mask_data))
    >>> full_brain = reconstruct_brain_map(roi_values, mask_data)
    >>> save_brain_map(full_brain, 'output.nii', mask_img)
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
    >>> mask_data, _ = load_roi_mask('roi.nii')
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
    >>> mask_data, mask_img = load_roi_mask('roi.nii')
    >>> roi_size = compute_roi_size(mask_data, mask_img.affine)
    >>> print(f"ROI: {roi_size['n_voxels']} voxels, {roi_size['volume_mm3']:.1f} mm³")
    """
    n_voxels = np.sum(mask)

    result = {'n_voxels': int(n_voxels)}

    if affine is not None:
        # Compute voxel volume robustly from affine (handles rotations/shears)
        import numpy as _np
        voxel_volume = abs(_np.linalg.det(affine[:3, :3]))
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


__all__ = [
    'load_nifti',
    'load_roi_mask',
    'extract_roi_data',
    'load_glm_betas',
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
