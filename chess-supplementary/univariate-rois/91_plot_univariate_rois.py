#!/usr/bin/env python3
"""
Generate Univariate ROI Supplementary Figure (Pylustrator)
==========================================================

Creates publication-ready flat surface plots showing ROI-level univariate
t-statistics (Experts vs Novices) for each contrast. Uses pylustrator for
interactive layout arrangement. Displays bilateral Glasser-180 ROIs on
flattened cortical surfaces with shared symmetric color scale.

Figures Produced
----------------

Univariate ROIs Panel
- File: figures/panels/univariate_rois_panel.svg (and .pdf)
- Individual axes saved to figures/: univariate_rois_1_Univariate_con_0001.svg, etc.
- Content: Flat cortical surface pair (left/right hemispheres) for each contrast
- Colors: Symmetric diverging colormap (blue = novices > experts, red = experts > novices)
- ROIs: Glasser-180 bilateral parcellation

Panel Layout:
- One panel per univariate contrast (e.g., all-gt-rest, check-gt-nocheck)
- Each panel shows left and right hemisphere flat surfaces side-by-side
- T-statistics from Welch's t-test (expert vs novice) mapped to ROI colors
- Shared vmin/vmax across all contrasts for consistent color interpretation

Inputs
------
- univ_group_stats.pkl (from univariate ROI analysis)
  Contains: Per-ROI Welch t-test results for each contrast
- ROI annotations: Glasser-180 bilateral surface labels (L/R hemispheres)
- ROI metadata: roi_info with roi_id, pretty_name, hemisphere, etc.

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives (apply_nature_rc, plot_flat_pair, save_axes_svgs)
- common.neuro_utils (roi_values_to_surface_texture for mapping ROI values to vertices)
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-supplementary/univariate-rois/91_plot_univariate_rois.py
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np

# Add parent (repo root) to sys.path for 'common' and 'modules'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import matplotlib.pyplot as plt
from nibabel.freesurfer import io as fsio
from common import setup_script, log_script_end
from common.bids_utils import load_roi_metadata
from common.neuro_utils import roi_values_to_surface_texture, create_glasser22_contours
from common.plotting import (
    apply_nature_rc,
    plot_flat_pair,
    embed_figure_on_ax,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
)
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from modules import UNIV_CONTRASTS


# =============================================================================
# Configuration and Results Loading
# =============================================================================
# Find latest univariate ROI results directory and initialize logging

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='univariate_rois',
    output_subdirs=['figures'],
    log_name='pylustrator_univariate_rois.log',
)
figures_dir = dirs['figures']

apply_nature_rc()  # Apply Nature journal style to all figures

# Load univariate group statistics (per-ROI Welch t-test results)
# Structure: index['contrasts'][contrast_code]['welch_expert_vs_novice']
# Contains: ROI_Label, t_stat, p_val, etc.
with open(results_dir / 'univ_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

# Load surface annotations (vertex-to-ROI mapping for left/right hemispheres)
# labels_l/r: array of ROI labels for each surface vertex
labels_l, _, _ = fsio.read_annot(CONFIG['ROI_GLASSER_180_ANNOT_L'])
labels_r, _, _ = fsio.read_annot(CONFIG['ROI_GLASSER_180_ANNOT_R'])

# Load ROI metadata (pretty names, hemisphere, etc.)
# Filter to bilateral ROIs only (roi_id <= 180)
roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_180'])
roi_info_bilateral = roi_info[roi_info['roi_id'] <= 180].copy()

logger.info(f"Loaded surface annotations: L={len(np.unique(labels_l))} labels, R={len(np.unique(labels_r))} labels")
logger.info(f"ROI metadata: {len(roi_info_bilateral)} bilateral ROIs")

# Get list of all univariate contrasts to plot
contrasts = list(UNIV_CONTRASTS.keys())


# =============================================================================
# Compute Shared Color Scale
# =============================================================================
# Compute symmetric vmin/vmax across all contrasts to ensure consistent
# color interpretation. Uses max absolute t-statistic across all contrasts.

all_abs = []
for con_code in contrasts:
    if 'contrasts' in index and con_code in index['contrasts']:
        # Extract t-statistics for this contrast
        tvals = index['contrasts'][con_code]['welch_expert_vs_novice']['t_stat'].to_numpy()
        # Keep only finite values (exclude NaN/Inf)
        all_abs.append(np.abs(tvals[np.isfinite(tvals)]))

# Compute global max absolute value
vmax_shared = float(np.max(all_abs)) if len(all_abs) else 1.0
vmin_shared = -vmax_shared  # Symmetric range for diverging colormap
logger.info(f"Univariate shared symmetric color scale (t): vmin={vmin_shared:.3f}, vmax={vmax_shared:.3f}")

# Create contours for key regions of interest
contours_l, contours_r = create_glasser22_contours(['dLPFC', 'PCC', 'V1', 'TPOJ', 'SP', 'VVS'])
roi_contours = {
    'contours_left': contours_l,
    'contours_right': contours_r,
    'labels': {1: 'V1', 4: 'VVS', 15: 'TPOJ', 16: 'SP', 18: 'PCC', 22: 'dLPFC'},
    'color': 'black',
    'width': 2.0
}

# =============================================================================
# Figure: Univariate ROI Surfaces (Independent Axes for Pylustrator)
# =============================================================================
# Creates one axis per contrast, each showing bilateral flat surfaces with
# ROI-level t-statistics. Pylustrator will arrange these axes in final layout.
# For each contrast, creates two panels:
# 1. All ROIs (showing all t-statistics)
# 2. Significant ROIs only (FDR-corrected, same color scale)

fig = plt.figure(1)

for idx, con_code in enumerate(contrasts, start=1):
    # Skip contrasts with missing data
    if 'contrasts' not in index or con_code not in index['contrasts']:
        logger.warning(f"No stats for {con_code}; skipping")
        continue

    # Extract Welch t-test results for this contrast
    blocks = index['contrasts'][con_code]
    welch = blocks['welch_expert_vs_novice']

    # Extract ROI IDs, t-statistics, and FDR significance
    bilateral_roi_ids = welch['ROI_Label'].to_numpy()  # ROI labels (1-180)
    tvals = welch['t_stat'].to_numpy()                 # Welch t-statistics
    sig_fdr = welch['significant_fdr'].to_numpy()      # FDR-corrected significance
    finite_mask = np.isfinite(tvals)                   # Exclude NaN/Inf values

    # Map bilateral ROI values to surface textures (vertex-level)
    # Each bilateral ROI ID is used for both L and R hemisphere surfaces
    roi_ids = bilateral_roi_ids

    # Create texture arrays for ALL ROIs (one value per surface vertex)
    # roi_values_to_surface_texture maps ROI-level values to all vertices in that ROI
    tex_l_all = roi_values_to_surface_texture(
        labels_l,              # Left hemisphere vertex-to-ROI mapping
        roi_ids,               # ROI IDs to map
        tvals,                 # T-statistics to assign to each ROI
        include_mask=finite_mask,  # Only include finite values
        default_value=0.0      # Value for excluded/missing ROIs
    )
    tex_r_all = roi_values_to_surface_texture(
        labels_r,              # Right hemisphere vertex-to-ROI mapping
        roi_ids,
        tvals,
        include_mask=finite_mask,
        default_value=0.0
    )

    # Create texture arrays for SIGNIFICANT ROIs ONLY (FDR-corrected)
    # Mask combines finite values AND FDR significance
    sig_mask = finite_mask & sig_fdr
    tex_l_sig = roi_values_to_surface_texture(
        labels_l,
        roi_ids,
        tvals,
        include_mask=sig_mask,  # Only include FDR-significant ROIs
        default_value=0.0
    )
    tex_r_sig = roi_values_to_surface_texture(
        labels_r,
        roi_ids,
        tvals,
        include_mask=sig_mask,
        default_value=0.0
    )

    # -----------------------------------------------------------------------------
    # Panel 1: All ROIs - Flat surface pair (L/R hemispheres) for this contrast
    # -----------------------------------------------------------------------------
    # Create matplotlib axis with descriptive label for pylustrator
    ax_all = plt.axes()
    ax_all.set_label(f'{idx}_Univariate_{con_code}_all')

    # Create flat surface plot (in-memory, no disk writes)
    surface_fig_all = plot_flat_pair(
        (tex_l_all, tex_r_all),
        title='',
        threshold=None,
        output_file=None,
        show_hemi_labels=False,
        show_colorbar=False,
        vmin=vmin_shared,
        vmax=vmax_shared,
        show_directions=True,
        roi_contours=roi_contours,
    )

    # Embed surface figure in matplotlib axis with title
    embed_figure_on_ax(
        ax_all,
        surface_fig_all,
        title=f'{UNIV_CONTRASTS[con_code]} (All ROIs)'  # Contrast name
    )

    # -----------------------------------------------------------------------------
    # Panel 2: Significant ROIs Only - Flat surface pair (L/R hemispheres)
    # -----------------------------------------------------------------------------
    # Create matplotlib axis with descriptive label for pylustrator
    ax_sig = plt.axes()
    ax_sig.set_label(f'{idx}_Univariate_{con_code}_sig')

    # Create flat surface plot (in-memory, no disk writes)
    surface_fig_sig = plot_flat_pair(
        (tex_l_sig, tex_r_sig),
        title='',
        threshold=None,
        output_file=None,
        show_hemi_labels=False,
        show_colorbar=False,
        vmin=vmin_shared,
        vmax=vmax_shared,
        show_directions=True,
        roi_contours=roi_contours,
    )

    # Embed surface figure in matplotlib axis with title
    embed_figure_on_ax(
        ax_sig,
        surface_fig_sig,
        title=f'{UNIV_CONTRASTS[con_code]} (FDR-Significant)'  # Contrast name
    )

    # Log number of significant ROIs for this contrast
    n_sig = sig_mask.sum()
    logger.info(f"{con_code}: {n_sig}/{len(sig_mask)} FDR-significant ROIs")

# Provide axis dictionary for pylustrator convenience
# This allows pylustrator to reference axes by label
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}


# =============================================================================
# Pylustrator Auto-Generated Layout Code
# =============================================================================
# The code between "#% start:" and "#% end:" markers is automatically generated
# by pylustrator when you save the layout interactively. This code positions
# and styles each axis according to your manual adjustments in the GUI.
# DO NOT manually edit this section - it will be overwritten on next save.

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(cm_to_inches(18.29), cm_to_inches(10.00), forward=True)
plt.figure(1).ax_dict["1_Univariate_con_0001_all"].set(position=[0.02, 0.7447, 0.45, 0.2154])
plt.figure(1).ax_dict["1_Univariate_con_0001_all"].set_position([0.026475, 0.497151, 0.436756, 0.424243])
plt.figure(1).ax_dict["1_Univariate_con_0001_sig"].set(position=[0.52, 0.7447, 0.45, 0.2154])
plt.figure(1).ax_dict["1_Univariate_con_0001_sig"].set_position([0.526177, 0.497151, 0.436756, 0.424243])
plt.figure(1).ax_dict["2_Univariate_con_0002_all"].set(position=[0.02, 0.5118, 0.45, 0.2154])
plt.figure(1).ax_dict["2_Univariate_con_0002_all"].set_position([0.026475, 0.038374, 0.436756, 0.424243])
plt.figure(1).ax_dict["2_Univariate_con_0002_sig"].set(position=[0.52, 0.5118, 0.45, 0.2154])
plt.figure(1).ax_dict["2_Univariate_con_0002_sig"].set_position([0.526177, 0.038374, 0.436756, 0.424243])
#% end: automatic generated code from pylustrator

# Display figure in pylustrator GUI for interactive layout adjustment
if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()


# =============================================================================
# Save Figures (Individual Axes and Full Panel)
# =============================================================================
# After arranging axes in pylustrator, save both:
# 1. Individual axes as separate SVG/PDF files (for modular figure assembly)
# 2. Full panel as complete SVG/PDF file (for standalone use)

# Save individual axes (one file per axis, named by prefix + axis label)
save_axes_svgs(fig, figures_dir, 'univariate_rois')

# Save full panel (complete multi-axis figure)
save_panel_pdf(fig, figures_dir / 'panels' / 'univariate_rois_panel.pdf')

logger.info("✓ Panel: univariate ROI maps complete")

log_script_end(logger)
