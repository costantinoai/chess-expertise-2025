"""
Prepare CAB-NP Subcortical Atlas for ROI-based RSA
===================================================

Downloads the Cole-Anticevic Brain-wide Network Partition (CAB-NP) atlas,
extracts subcortical parcels, groups them into bilateral anatomical ROIs,
and resamples to match the existing functional data space.

METHODS
-------

Atlas Source
------------
The CAB-NP atlas (Ji et al., 2019, NeuroImage) provides a whole-brain
parcellation with 360 cortical parcels (identical to Glasser HCP-MMP1) and
358 subcortical parcels, all assigned to 12 functional networks. We use the
volumetric MNI-space version from https://github.com/ColeLab/ColeAnticevicNetPartition.

The repository provides hemisphere-separated volumetric NIfTI files for the
subcortex: subcortex_atlas_GSR_parcels_L.nii and subcortex_atlas_GSR_parcels_R.nii
in the SeparateHemispheres/ directory. These contain integer parcel labels for
left and right subcortical structures in MNI152NLin6Asym space (2 mm).

Subcortical Grouping
--------------------
Parcel labels in the CAB-NP atlas encode the anatomical structure in their
label names (e.g., "Hippocampus", "Caudate", "Thalamus"). We group parcels
by matching label names to anatomical keywords and merge left and right
hemisphere parcels for each structure into a single bilateral mask. This
produces one integer label per bilateral ROI.

Resampling
----------
The resulting atlas is resampled from its native MNI space to the target
space (MNI152NLin2009cAsym, 2 mm isotropic) using nearest-neighbor
interpolation to preserve integer labels. The existing Glasser 22-region
bilateral atlas serves as the reference image for exact spatial alignment.

Outputs
-------
- tpl-MNI152NLin2009cAsym_res-02_atlas-CABNP_desc-subcortical_bilateral_resampled.nii.gz
- region_info.tsv: ROI metadata following glasser22/region_info.tsv format
"""

from pathlib import Path


import numpy as np
import nibabel as nib
import pandas as pd
import tempfile
import subprocess

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="subcortical_atlas_prep",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

# Output atlas directory (alongside glasser22)
CABNP_DIR = CONFIG['ROI_ROOT'] / "cab-np"
CABNP_DIR.mkdir(parents=True, exist_ok=True)

# Reference image for resampling (Glasser 22-region bilateral atlas)
REFERENCE_ATLAS = CONFIG['ROI_GLASSER_22_ATLAS']
if not REFERENCE_ATLAS.exists():
    raise FileNotFoundError(f"Reference atlas not found: {REFERENCE_ATLAS}")

ATLAS_NIFTI_NAME = "tpl-MNI152NLin2009cAsym_res-02_atlas-CABNP_desc-subcortical_bilateral_resampled.nii.gz"
ATLAS_OUTPUT_PATH = CABNP_DIR / ATLAS_NIFTI_NAME
REGION_INFO_PATH = CABNP_DIR / "region_info.tsv"

logger.info(f"Output atlas dir: {CABNP_DIR}")
logger.info(f"Reference atlas: {REFERENCE_ATLAS}")

# =============================================================================
# 2. DOWNLOAD CAB-NP ATLAS
# =============================================================================

logger.info("Downloading CAB-NP atlas from GitHub...")

CABNP_REPO_URL = "https://github.com/ColeLab/ColeAnticevicNetPartition.git"

with tempfile.TemporaryDirectory() as tmpdir:
    clone_dir = Path(tmpdir) / "ColeAnticevicNetPartition"

    logger.info("Cloning ColeAnticevicNetPartition repository (sparse)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", CABNP_REPO_URL, str(clone_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    logger.info(f"Cloned to {clone_dir}")

    # =============================================================================
    # 3. LOAD HEMISPHERE-SEPARATED SUBCORTICAL VOLUMETRIC ATLASES
    # =============================================================================

    # The SeparateHemispheres/ directory provides volumetric NIfTI files
    # with individual parcel labels for left and right subcortex
    sep_hemi_dir = clone_dir / "SeparateHemispheres"
    left_parcels_path = sep_hemi_dir / "subcortex_atlas_GSR_parcels_L.nii"
    right_parcels_path = sep_hemi_dir / "subcortex_atlas_GSR_parcels_R.nii"

    if not left_parcels_path.exists() or not right_parcels_path.exists():
        raise FileNotFoundError(
            f"Expected subcortical parcel NIfTI files not found in {sep_hemi_dir}. "
            f"Available: {[f.name for f in sep_hemi_dir.glob('*')]}"
        )

    left_img = nib.load(str(left_parcels_path))
    right_img = nib.load(str(right_parcels_path))
    left_data = np.asarray(left_img.dataobj).astype(np.int32)
    right_data = np.asarray(right_img.dataobj).astype(np.int32)

    logger.info(f"Left subcortex parcels shape: {left_data.shape}, "
                f"unique labels: {np.unique(left_data[left_data > 0]).tolist()[:10]}...")
    logger.info(f"Right subcortex parcels shape: {right_data.shape}, "
                f"unique labels: {np.unique(right_data[right_data > 0]).tolist()[:10]}...")

    # =============================================================================
    # 4. LOAD LABEL KEY TO MAP PARCEL INDICES TO STRUCTURE NAMES
    # =============================================================================

    # The label key file maps each parcel index to a name like
    # "Visual1-04_L-Hippocampus" which encodes network and structure
    label_key_path = clone_dir / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
    if not label_key_path.exists():
        raise FileNotFoundError(f"Label key file not found: {label_key_path}")

    # Parse label key: each line has "INDEX LABELNAME RGBA"
    label_key = {}
    with open(label_key_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    label_key[idx] = name
                except ValueError:
                    continue

    logger.info(f"Loaded {len(label_key)} labels from key file")

    # Identify subcortical labels: those containing structure keywords
    # Subcortical label names contain structure suffixes like "-Hippocampus",
    # "-Caudate", "-Putamen", "-Thalamus", etc.
    subcortical_keywords = [
        'Hippocampus', 'Amygdala', 'Caudate', 'Putamen', 'Pallidum',
        'Thalamus', 'Accumbens', 'Cerebellum', 'BrainStem', 'Brain_Stem',
        'Brainstem', 'Diencephalon',
    ]

    # Map labels to structures
    label_to_structure = {}
    for idx, name in label_key.items():
        for kw in subcortical_keywords:
            if kw.lower() in name.lower():
                label_to_structure[idx] = kw
                break

    # Log some examples
    subcort_labels = {k: v for k, v in label_to_structure.items()}
    logger.info(f"Identified {len(subcort_labels)} subcortical parcel labels")
    for idx in sorted(subcort_labels.keys())[:10]:
        logger.info(f"  Label {idx}: {label_key[idx]} -> {subcort_labels[idx]}")

    # Also identify from the volumetric data directly: get unique labels
    left_labels_unique = set(np.unique(left_data[left_data > 0]).tolist())
    right_labels_unique = set(np.unique(right_data[right_data > 0]).tolist())
    logger.info(f"Left parcel unique labels: {len(left_labels_unique)}")
    logger.info(f"Right parcel unique labels: {len(right_labels_unique)}")

    # For each unique label in left/right volumes, look up structure from label key
    # If the label key doesn't cover these indices (volumetric files may use
    # different numbering), we need to inspect the actual volume label values
    all_vol_labels = left_labels_unique | right_labels_unique
    logger.info(f"Total unique volumetric labels: {len(all_vol_labels)}")
    logger.info(f"Sample volumetric labels: {sorted(all_vol_labels)[:20]}")

    # The volumetric parcel files may use their own numbering scheme.
    # Let's also load the CIFTI to get the definitive label-to-name mapping
    # for the subcortical portion
    cifti_path = clone_dir / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii"
    cifti_img = nib.load(str(cifti_path))
    cifti_data = cifti_img.get_fdata()  # (1, n_grayordinates)
    label_axis = cifti_img.header.get_axis(0)
    brain_model_axis = cifti_img.header.get_axis(1)

    # Get the label table from the CIFTI (definitive mapping)
    cifti_label_table = label_axis.label[0]  # dict: int -> (name, rgba)

    # Extract subcortical brain models and their voxel assignments
    # Build a mapping: (i,j,k) voxel -> parcel label from the CIFTI
    vol_shape = brain_model_axis.volume_shape
    vol_affine = brain_model_axis.affine

    # Create a 3D volume from the CIFTI subcortical data
    subcort_vol = np.zeros(vol_shape, dtype=np.int32)
    for bm in brain_model_axis.iter_structures():
        structure_name, data_slice, brain_model = bm
        struct_str = str(structure_name)
        if 'CORTEX' in struct_str:
            continue  # Skip cortical structures

        voxels = brain_model.voxel  # (n_voxels, 3) IJK indices
        # data_slice is a slice object
        start = data_slice.start if data_slice.start is not None else 0
        stop = data_slice.stop if data_slice.stop is not None else cifti_data.shape[1]
        parcel_ids = cifti_data[0, start:stop]

        for v_idx, v in enumerate(voxels):
            subcort_vol[v[0], v[1], v[2]] = int(parcel_ids[v_idx])

    n_subcort_voxels = np.sum(subcort_vol > 0)
    subcort_unique = np.unique(subcort_vol[subcort_vol > 0])
    logger.info(f"CIFTI subcortical volume: {n_subcort_voxels} voxels, "
                f"{len(subcort_unique)} unique parcels")

    # Now map each subcortical parcel label to its anatomical structure
    # using the CIFTI label table
    parcel_to_structure = {}
    for lbl_idx in subcort_unique:
        lbl_idx_int = int(lbl_idx)
        if lbl_idx_int in cifti_label_table:
            name = cifti_label_table[lbl_idx_int][0]
        elif lbl_idx_int in label_key:
            name = label_key[lbl_idx_int]
        else:
            name = f"Unknown_{lbl_idx_int}"

        # Parse structure from label name
        # Labels have format: "Network-XX_H-Structure"
        # e.g., "Visual1-04_L-Hippocampus", "Default-01_R-Caudate"
        name_upper = name.upper()
        structure = None

        structure_mapping = {
            'HIPPOCAMPUS': 'Hippocampus',
            'AMYGDALA': 'Amygdala',
            'CAUDATE': 'Caudate',
            'PUTAMEN': 'Putamen',
            'PALLIDUM': 'Pallidum',
            'THALAMUS': 'Thalamus',
            'ACCUMBENS': 'NAcc',
            'CEREBELLUM': 'Cerebellum',
            'BRAINSTEM': 'Brainstem',
            'BRAIN_STEM': 'Brainstem',
            'DIENCEPHALON': 'Thalamus',
        }

        for kw, struct in structure_mapping.items():
            if kw in name_upper:
                structure = struct
                break

        if structure is None:
            logger.warning(f"Cannot map parcel {lbl_idx_int} ({name}) to a structure")
            continue

        parcel_to_structure[lbl_idx_int] = structure

    logger.info(f"Mapped {len(parcel_to_structure)} parcels to structures")

    # Count parcels per structure
    structure_counts = {}
    for struct in parcel_to_structure.values():
        structure_counts[struct] = structure_counts.get(struct, 0) + 1
    for struct, count in sorted(structure_counts.items()):
        logger.info(f"  {struct}: {count} parcels")

    # =============================================================================
    # 5. BUILD BILATERAL ROI ATLAS
    # =============================================================================

    # Ordered list of bilateral ROIs
    roi_order = ['Hippocampus', 'Amygdala', 'Caudate', 'Putamen', 'Pallidum',
                 'Thalamus', 'NAcc', 'Cerebellum', 'Brainstem']

    # Keep only ROIs that have parcels
    active_rois = [r for r in roi_order if r in structure_counts]
    logger.info(f"Active bilateral ROIs ({len(active_rois)}): {active_rois}")

    # Create atlas volume: replace parcel labels with bilateral ROI labels
    atlas_vol = np.zeros(vol_shape, dtype=np.int16)

    for roi_idx, roi_name in enumerate(active_rois, start=1):
        # Find all parcel labels belonging to this structure
        member_parcels = [p for p, s in parcel_to_structure.items() if s == roi_name]
        mask = np.isin(subcort_vol, member_parcels)
        atlas_vol[mask] = roi_idx
        n_vox = int(np.sum(mask))
        logger.info(f"ROI {roi_idx} ({roi_name}): {n_vox} voxels "
                    f"({len(member_parcels)} parcels merged)")

    # Verify atlas
    unique_labels = np.unique(atlas_vol)
    logger.info(f"Atlas unique labels: {unique_labels.tolist()}")
    logger.info(f"Non-zero voxels: {np.sum(atlas_vol > 0)}")

    # Save as NIfTI in the CIFTI volume space
    atlas_native = nib.Nifti1Image(atlas_vol, vol_affine)
    native_path = Path(tmpdir) / "atlas_native.nii.gz"
    nib.save(atlas_native, str(native_path))
    logger.info(f"Saved native-space atlas: {native_path}")

    # =============================================================================
    # 6. RESAMPLE TO FUNCTIONAL DATA SPACE
    # =============================================================================

    logger.info("Resampling atlas to functional data space...")

    from nilearn.image import resample_to_img

    # Load reference image (Glasser 22 atlas in MNI152NLin2009cAsym 2mm)
    ref_img = nib.load(str(REFERENCE_ATLAS))
    logger.info(f"Reference image shape: {ref_img.shape}")

    # Resample using nearest-neighbor to preserve integer labels
    resampled_img = resample_to_img(
        source_img=atlas_native,
        target_img=ref_img,
        interpolation='nearest',
    )

    # Verify resampled atlas
    resampled_data = resampled_img.get_fdata().astype(np.int16)
    unique_resampled = np.unique(resampled_data)
    logger.info(f"Resampled atlas shape: {resampled_data.shape}")
    logger.info(f"Resampled unique labels: {unique_resampled.tolist()}")
    for roi_idx, roi_name in enumerate(active_rois, start=1):
        n_vox = int(np.sum(resampled_data == roi_idx))
        logger.info(f"  ROI {roi_idx} ({roi_name}): {n_vox} voxels after resampling")

    # Check and resolve overlap with cortical atlas
    ref_data = ref_img.get_fdata()
    cortical_mask = ref_data > 0
    subcortical_mask = resampled_data > 0
    overlap = int(np.sum(cortical_mask & subcortical_mask))
    logger.info(f"Overlap with cortical atlas: {overlap} voxels")
    if overlap > 0:
        logger.warning(f"Zeroing {overlap} overlapping voxels to avoid conflict")
        resampled_data[cortical_mask & subcortical_mask] = 0

    # Save final resampled atlas
    final_img = nib.Nifti1Image(resampled_data, resampled_img.affine, resampled_img.header)
    nib.save(final_img, str(ATLAS_OUTPUT_PATH))
    logger.info(f"Saved resampled atlas: {ATLAS_OUTPUT_PATH}")

# =============================================================================
# 7. CREATE region_info.tsv
# =============================================================================

# Define ROI metadata following the glasser22/region_info.tsv format
roi_groups = {
    'Hippocampus': 'MTL',
    'Amygdala': 'MTL',
    'Caudate': 'Basal Ganglia',
    'Putamen': 'Basal Ganglia',
    'Pallidum': 'Basal Ganglia',
    'NAcc': 'Basal Ganglia',
    'Thalamus': 'Diencephalon',
    'Cerebellum': 'Cerebellum',
    'Brainstem': 'Brainstem',
}

# Colors per group (distinct, colorblind-friendly)
group_colors = {
    'MTL': '#e41a1c',
    'Basal Ganglia': '#377eb8',
    'Diencephalon': '#4daf4a',
    'Cerebellum': '#984ea3',
    'Brainstem': '#ff7f00',
}

group_colors_cb = {
    'MTL': '#D55E00',
    'Basal Ganglia': '#0072B2',
    'Diencephalon': '#009E73',
    'Cerebellum': '#CC79A7',
    'Brainstem': '#E69F00',
}

rows = []
for roi_idx, roi_name in enumerate(active_rois, start=1):
    group = roi_groups.get(roi_name, 'Other')
    rows.append({
        'ROI_idx': roi_idx,
        'roi_name': roi_name,
        'pretty_name': roi_name.replace('_', ' '),
        'group': group,
        'color': group_colors.get(group, '#999999'),
        'color_cb': group_colors_cb.get(group, '#999999'),
        'order': roi_idx,
    })

region_df = pd.DataFrame(rows)
region_df.to_csv(REGION_INFO_PATH, sep='\t', index=False)
logger.info(f"Saved region_info.tsv: {REGION_INFO_PATH}")
logger.info(f"  {len(active_rois)} bilateral ROIs: {active_rois}")

# =============================================================================
# 8. VERIFY OUTPUTS
# =============================================================================

verify_img = nib.load(str(ATLAS_OUTPUT_PATH))
verify_data = verify_img.get_fdata()
verify_labels = np.unique(verify_data[verify_data > 0]).astype(int)
expected_labels = np.arange(1, len(active_rois) + 1)

if not np.array_equal(verify_labels, expected_labels):
    raise RuntimeError(
        f"Atlas verification failed: expected labels {expected_labels}, "
        f"got {verify_labels}"
    )

verify_tsv = pd.read_csv(REGION_INFO_PATH, sep='\t')
if len(verify_tsv) != len(active_rois):
    raise RuntimeError(
        f"region_info.tsv verification failed: expected {len(active_rois)} rows, "
        f"got {len(verify_tsv)}"
    )

logger.info("Atlas verification passed")
logger.info(f"Final atlas: {ATLAS_OUTPUT_PATH}")
logger.info(f"Final metadata: {REGION_INFO_PATH}")

log_script_end(logger)
