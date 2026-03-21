"""
Visualize Glasser and CAB-NP Subcortical Atlases on MNI Anatomy
================================================================

Produces axial slice montages showing both the Glasser 22-region cortical
atlas and the CAB-NP 9-region subcortical atlas overlaid on the MNI152
template anatomy. This allows visual verification of atlas coverage and
spatial alignment.

Figures Produced
----------------
- atlas_cortical_axial.pdf: Glasser 22-region cortical atlas on MNI anatomy
- atlas_subcortical_axial.pdf: CAB-NP 9-region subcortical atlas on MNI anatomy
- atlas_combined_axial.pdf: Both atlases combined on MNI anatomy
- panels/atlas_overview_panel.pdf: Combined panel

Inputs
------
- CONFIG['ROI_GLASSER_22_ATLAS']: Glasser cortical atlas NIfTI
- CONFIG['ROI_CABNP_SUBCORTICAL_ATLAS']: CAB-NP subcortical atlas NIfTI
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import CONFIG

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import pandas as pd
from nilearn import plotting, datasets

from common import setup_script, log_script_end
from common.bids_utils import load_roi_metadata
from common.plotting import apply_nature_rc, save_panel_pdf


# =============================================================================
# Setup
# =============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='subcortical_rois',
    output_subdirs=['figures'],
    log_name='plot_atlas_on_mni.log',
)
FIGURES_DIR = dirs['figures']

apply_nature_rc()


# =============================================================================
# Load atlases and metadata
# =============================================================================

logger.info("Loading atlases...")

# Glasser cortical atlas
glasser_img = nib.load(str(CONFIG['ROI_GLASSER_22_ATLAS']))
glasser_data = glasser_img.get_fdata()
glasser_meta = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
logger.info(f"Glasser: {len(glasser_meta)} ROIs, {int(np.sum(glasser_data > 0))} voxels")

# CAB-NP subcortical atlas
cabnp_img = nib.load(str(CONFIG['ROI_CABNP_SUBCORTICAL_ATLAS']))
cabnp_data = cabnp_img.get_fdata()
cabnp_meta = load_roi_metadata(CONFIG['ROI_CABNP'])
logger.info(f"CAB-NP: {len(cabnp_meta)} ROIs, {int(np.sum(cabnp_data > 0))} voxels")

# MNI template for background
mni_template = datasets.load_mni152_template(resolution=2)

# Axial slices to display (MNI z-coordinates)
# Cover subcortical structures: z = -20 to +20
# Plus some cortical coverage: z = +30 to +60
z_slices_subcort = [-20, -14, -8, -2, 4, 10, 16, 22]
z_slices_cortical = [10, 20, 30, 40, 50, 60]
z_slices_combined = [-14, -4, 6, 16, 26, 40, 50, 60]


# =============================================================================
# Helper: create a discrete colormap from ROI metadata
# =============================================================================

def make_roi_cmap(meta_df):
    """Create a ListedColormap from ROI metadata colors."""
    colors_hex = meta_df['color'].tolist()
    from matplotlib.colors import to_rgba
    colors_rgba = [to_rgba(c) for c in colors_hex]
    return ListedColormap(colors_rgba)


def create_atlas_overlay(atlas_img, atlas_meta, bg_img, z_slices, title, output_path):
    """Create axial slice montage of atlas on MNI anatomy."""
    n_slices = len(z_slices)
    n_cols = 4
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = np.atleast_2d(axes)

    atlas_data = atlas_img.get_fdata()
    bg_data = bg_img.get_fdata()

    # Get atlas affine to convert MNI coords to voxel indices
    affine_inv = np.linalg.inv(atlas_img.affine)

    for idx, z_mni in enumerate(z_slices):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Convert MNI z to voxel k
        voxel_coord = affine_inv @ np.array([0, 0, z_mni, 1])
        k = int(round(voxel_coord[2]))

        if k < 0 or k >= atlas_data.shape[2]:
            ax.axis('off')
            continue

        # Plot MNI background
        bg_slice = np.rot90(bg_data[:, :, k])
        ax.imshow(bg_slice, cmap='gray', origin='upper', aspect='equal')

        # Overlay atlas
        atlas_slice = np.rot90(atlas_data[:, :, k])
        masked_atlas = np.ma.masked_where(atlas_slice == 0, atlas_slice)

        cmap = make_roi_cmap(atlas_meta)
        n_rois = len(atlas_meta)
        ax.imshow(masked_atlas, cmap=cmap, vmin=0.5, vmax=n_rois + 0.5,
                  alpha=0.6, origin='upper', aspect='equal')

        ax.set_title(f'z = {z_mni}', fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n_slices, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    # Add legend
    legend_elements = [
        Patch(facecolor=row['color'], label=row.get('pretty_name', row['roi_name']))
        for _, row in atlas_meta.iterrows()
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=min(5, len(legend_elements)), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=10, weight='bold', y=1.02)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Create atlas visualizations
# =============================================================================

logger.info("Creating atlas overlay figures...")

# 1. Subcortical atlas
create_atlas_overlay(
    atlas_img=cabnp_img,
    atlas_meta=cabnp_meta,
    bg_img=mni_template,
    z_slices=z_slices_subcort,
    title='CAB-NP Subcortical Atlas (9 bilateral ROIs)',
    output_path=FIGURES_DIR / 'atlas_subcortical_axial.pdf',
)

# 2. Cortical atlas
create_atlas_overlay(
    atlas_img=glasser_img,
    atlas_meta=glasser_meta,
    bg_img=mni_template,
    z_slices=z_slices_cortical,
    title='Glasser Cortical Atlas (22 bilateral ROIs)',
    output_path=FIGURES_DIR / 'atlas_cortical_axial.pdf',
)

# 3. Combined atlas (merge both into one volume for display)
combined_data = glasser_data.copy()
# Subcortical labels start after max cortical label
max_cortical = int(glasser_data.max())
subcort_shifted = cabnp_data.copy()
subcort_shifted[subcort_shifted > 0] += max_cortical
combined_data[cabnp_data > 0] = subcort_shifted[cabnp_data > 0]

combined_img = nib.Nifti1Image(combined_data.astype(np.int16), glasser_img.affine)

# Build combined metadata
combined_meta = pd.concat([glasser_meta, cabnp_meta.assign(
    roi_id=cabnp_meta['roi_id'] + max_cortical
)], ignore_index=True)

create_atlas_overlay(
    atlas_img=combined_img,
    atlas_meta=combined_meta,
    bg_img=mni_template,
    z_slices=z_slices_combined,
    title='Combined Cortical (Glasser) + Subcortical (CAB-NP) Atlas',
    output_path=FIGURES_DIR / 'atlas_combined_axial.pdf',
)

# =============================================================================
# Also use nilearn's plot_roi for a cleaner glass-brain view
# =============================================================================

logger.info("Creating glass brain views...")

fig_glass, axes_glass = plt.subplots(1, 2, figsize=(14, 5))

# Subcortical glass brain
plotting.plot_roi(
    cabnp_img, bg_img=mni_template,
    display_mode='ortho', cut_coords=(0, -10, 5),
    title='CAB-NP Subcortical',
    axes=axes_glass[0],
    alpha=0.5,
)

# Cortical glass brain
plotting.plot_roi(
    glasser_img, bg_img=mni_template,
    display_mode='ortho', cut_coords=(0, -30, 40),
    title='Glasser Cortical',
    axes=axes_glass[1],
    alpha=0.5,
)

glass_path = FIGURES_DIR / 'atlas_glass_brain.pdf'
fig_glass.savefig(glass_path, dpi=300, bbox_inches='tight')
plt.close(fig_glass)
logger.info(f"Saved: {glass_path}")

logger.info("Atlas visualization complete")
log_script_end(logger)
