"""
Generate Neurosynth Univariate Figure Panels (Pylustrator)
===========================================================

Creates publication-ready multi-panel figures for Neurosynth univariate analysis.
Uses pylustrator for interactive layout arrangement. The script builds
independent axes using standardized plotting primitives and then saves both
individual axes (SVG/PDF) and assembled panels (SVG/PDF) into the current
neurosynth univariate results directory.

Figures Produced
----------------

Panel: Neurosynth Univariate Multi-Panel Figure
- File: figures/panels/neurosynth_univariate_panel.svg (and .pdf)
- Axes saved to figures/: neurosynth_univariate_*.svg and neurosynth_univariate_*.pdf
- Content:
  - A1: Term correlations (All > Baseline) - grouped bars showing POS vs NEG correlations
  - A2: Correlation differences (All > Baseline) - bars showing POS - NEG differences
  - B1: Term correlations (Checkmate > Non-checkmate) - grouped bars showing POS vs NEG correlations
  - B2: Correlation differences (Checkmate > Non-checkmate) - bars showing POS - NEG differences
  - C: Flat surface maps (All > Baseline) - left and right hemisphere projections
  - D: Flat surface maps (Checkmate > Non-checkmate) - left and right hemisphere projections

Inputs
------
- CSV files with term correlation results:
  - *_pos_term_corr.csv: Positive term correlations (Expert > Novice regions)
  - *_neg_term_corr.csv: Negative term correlations (Novice > Expert regions)
  - *_diff_term_corr.csv: Correlation differences (POS - NEG)
- zmap_*.nii.gz: Statistical z-maps for surface plotting

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives and style (apply_nature_rc, plot_grouped_bars_on_ax, etc.)
- modules.plot_utils for Neurosynth-specific plotting helpers
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-neurosynth/91_plot_neurosynth_univariate.py
"""

import sys
import os
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import matplotlib.pyplot as plt
from nilearn import image

from common.plotting import (
    apply_nature_rc,
    save_axes_svgs,
    save_panel_pdf,
    compute_ylim_range,
    plot_flat_pair,
    embed_figure_on_ax,
)
from common.neuro_utils import project_volume_to_surfaces
from modules.plot_utils import (
    plot_correlations_on_ax,
    plot_differences_on_ax,
    load_term_corr_triple,
)
from common.io_utils import find_latest_results_directory
from common.logging_utils import setup_analysis_in_dir, log_script_end


# NOTE: Helper functions moved to modules.plot_utils for reusability


# =============================================================================
# Configuration and results
# =============================================================================

apply_nature_rc()

# Find latest neurosynth univariate results directory
# Creates 'figures' subdirectory if needed for saving outputs
RESULTS_DIR = find_latest_results_directory(
    script_dir / 'results',
    pattern='*_neurosynth_univariate',  # Match neurosynth univariate analysis results
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True,
)
FIGURES_DIR = RESULTS_DIR / 'figures'


# =============================================================================
# Setup logging
# =============================================================================

extra = {"RESULTS_DIR": str(RESULTS_DIR), "FIGURES_DIR": str(FIGURES_DIR)}
_, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='pylustrator_neurosynth_univariate.log',
)


# =============================================================================
# Load Neurosynth term correlation data
# =============================================================================

# Define file stems for both contrasts
# - all-gt-rest: All chess boards > Baseline (fixation)
# - check-gt-nocheck: Checkmate boards > Non-checkmate boards
stem_all = 'spmT_exp-gt-nonexp_all-gt-rest'       # All > Baseline contrast
stem_check = 'spmT_exp-gt-nonexp_check-gt-nocheck'  # Checkmate > Non-checkmate contrast

# Load term correlation results for All > Baseline contrast
# Each triple contains: (pos_df, neg_df, diff_df)
# - pos_df: Positive correlations (Expert > Novice regions)
# - neg_df: Negative correlations (Novice > Expert regions)
# - diff_df: Difference (POS - NEG)
# DataFrames have columns: term, r (correlation), p (p-value), ci_low, ci_high
df_pos_all, df_neg_all, df_diff_all = load_term_corr_triple(RESULTS_DIR, stem_all)

# Load term correlation results for Checkmate > Non-checkmate contrast
df_pos_chk, df_neg_chk, df_diff_chk = load_term_corr_triple(RESULTS_DIR, stem_check)


# =============================================================================
# Figure: Neurosynth Univariate Multi-Panel Figure
# =============================================================================
# This figure shows Neurosynth univariate analysis results:
# - Panels A: All > Baseline contrast (term correlations and differences)
# - Panels B: Checkmate > Non-checkmate contrast (term correlations and differences)
# - Panels C/D: Surface projections of statistical maps for both contrasts

fig = plt.figure(1)

# -----------------------------------------------------------------------------
# Panel A1: Term Correlations (All > Baseline)
# -----------------------------------------------------------------------------
# Grouped barplot showing correlations between top Neurosynth terms and brain activity
# POS = Expert > Novice regions (positive z-values)
# NEG = Novice > Expert regions (negative z-values)
# Shows which cognitive terms are associated with expertise-related brain activity
ax_A1 = plt.axes(); ax_A1.set_label('A1_Corr_All_gt_Rest')
plot_correlations_on_ax(
    ax_A1,
    df_pos_all,                         # Positive term correlations (Expert > Novice)
    df_neg_all,                         # Negative term correlations (Novice > Expert)
    title='Term correlations',
    subtitle='All > Baseline'
)

# -----------------------------------------------------------------------------
# Panel B1: Term Correlations (Checkmate > Non-checkmate)
# -----------------------------------------------------------------------------
# Same as Panel A1, but for checkmate vs non-checkmate contrast
ax_B1 = plt.axes(); ax_B1.set_label('B1_Corr_Check_gt_NoCheck')
plot_correlations_on_ax(
    ax_B1,
    df_pos_chk,                         # Positive term correlations (Checkmate > Non-checkmate)
    df_neg_chk,                         # Negative term correlations (No-Check > Check)
    title='Term correlations',
    subtitle='Checkmate > Non-checkmate'
)

# -----------------------------------------------------------------------------
# Panel A2: Correlation Differences (All > Baseline)
# -----------------------------------------------------------------------------
# Barplot showing difference between POS and NEG correlations (POS - NEG)
# Positive values = term more associated with Expert > Novice regions
# Negative values = term more associated with Novice > Expert regions
ax_A2 = plt.axes(); ax_A2.set_label('A2_Diff_All_gt_Rest')
plot_differences_on_ax(
    ax_A2,
    df_diff_all,                        # Correlation differences (POS - NEG)
    title='ΔCorrelation',
    subtitle='All > Baseline'
)

# -----------------------------------------------------------------------------
# Panel B2: Correlation Differences (Checkmate > Non-checkmate)
# -----------------------------------------------------------------------------
# Same as Panel A2, but for checkmate vs non-checkmate contrast
ax_B2 = plt.axes(); ax_B2.set_label('B2_Diff_Check_gt_NoCheck')
plot_differences_on_ax(
    ax_B2,
    df_diff_chk,                        # Correlation differences (POS - NEG)
    title='ΔCorrelation',
    subtitle='Checkmate > Non-checkmate'
)

# Load volumes for both contrasts
z_img_all = image.load_img(str(RESULTS_DIR / f"zmap_{stem_all}.nii.gz"))
z_img_check = image.load_img(str(RESULTS_DIR / f"zmap_{stem_check}.nii.gz"))

# Project each volume to surfaces ONCE (DRY: no re-projection!)
tex_all_l, tex_all_r = project_volume_to_surfaces(z_img_all)
tex_check_l, tex_check_r = project_volume_to_surfaces(z_img_check)

# Compute symmetric color range across ALL surface textures
# Use same vmin/vmax across both contrasts for consistent color interpretation
vmin_univ, vmax_univ = compute_ylim_range(
    tex_all_l, tex_all_r, tex_check_l, tex_check_r,
    symmetric=True,
    padding_pct=0.0
)

# -----------------------------------------------------------------------------
# Panel C: Flat Surface Maps (All > Baseline)
# -----------------------------------------------------------------------------
# Left and right hemisphere flat surface projections of z-map
# Shows spatial distribution of Expert > Novice activity across cortex
# Warm colors = Expert > Novice, Cool colors = Novice > Expert
ax_C = plt.axes(); ax_C.set_label('C_Flat_All_gt_Rest')

# Plot using pre-computed textures (no re-projection!)
surface_fig_all = plot_flat_pair(
    textures=(tex_all_l, tex_all_r),  # Pre-computed surface textures
    threshold=0,                       # Show all values (no thresholding)
    output_file=None,                  # Don't save (will embed in main figure)
    show_hemi_labels=False,            # No hemisphere labels (added manually)
    show_colorbar=False,               # Colorbar shown separately
    vmin=vmin_univ,                    # Symmetric minimum
    vmax=vmax_univ,                    # Symmetric maximum
    show_directions=True,              # Show anterior/posterior labels
)
embed_figure_on_ax(ax_C, surface_fig_all, title='Surface Projection', subtitle="All > Baseline")

# -----------------------------------------------------------------------------
# Panel D: Flat Surface Maps (Checkmate > Non-checkmate)
# -----------------------------------------------------------------------------
# Same as Panel C, but for checkmate vs non-checkmate contrast
ax_D = plt.axes(); ax_D.set_label('D_Flat_Check_gt_NoCheck')

# Plot using pre-computed textures (no re-projection!)
surface_fig_check = plot_flat_pair(
    textures=(tex_check_l, tex_check_r),  # Pre-computed surface textures
    threshold=0,                           # Show all values (no thresholding)
    output_file=None,                      # Don't save (will embed in main figure)
    show_hemi_labels=False,                # No hemisphere labels (added manually)
    show_colorbar=False,                   # Colorbar shown separately
    vmin=vmin_univ,                        # Symmetric minimum (same as Panel C)
    vmax=vmax_univ,                        # Symmetric maximum (same as Panel C)
    show_directions=True,                  # Show anterior/posterior labels
)
embed_figure_on_ax(ax_D, surface_fig_check, title='Surface Projection', subtitle='Checkmate > Non-Checkmate')

# -----------------------------------------------------------------------------
# Panel E: Glass Brain Terms Visualization
# -----------------------------------------------------------------------------
# Display pre-made glass brain image showing term locations from manuscript data
ax_E = plt.axes(); ax_E.set_label('E_Terms_Glass')

# Load image from ROI directory (centralized path from CONFIG)
import matplotlib.image as mpimg
terms_glass_path = CONFIG['NEUROSYNTH_ROOT'] / 'terms_glass.png'

img = mpimg.imread(str(terms_glass_path))
ax_E.imshow(img)
ax_E.set_axis_off()
logger.info(f"Loaded glass brain image from {terms_glass_path}")


# =============================================================================
# Pylustrator Setup and Interactive Layout
# =============================================================================
# Create ax_dict for the figure to enable easy axis reference in pylustrator
# This allows pylustrator to reference axes by label (e.g., fig.ax_dict["A1_Corr_All_gt_Rest"])
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
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(18.230000/2.54, 14.170000/2.54, forward=True)
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].legend(loc=(0.4927, 0.7789), frameon=False)
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].set(position=[0.05757, 0.4808, 0.2188, 0.2177], ylim=(-0.15, 0.19))
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].set_position([0.062204, 0.385487, 0.216904, 0.258545])
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].texts[0].set(position=(0.5, 1.071))
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].texts[1].set(position=(0.5, 1.017))
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].text(-0.1489, 1.0710, 'b', transform=plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].texts[2].new
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].legend(loc=(0.4261, 0.7811), frameon=False)
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].set(position=[0.3343, 0.4808, 0.1422, 0.2177], yticks=[-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4], yticklabels=['−0.2', '−0.1', '0.0', '0.1', '0.2', '0.3', '0.4'], ylim=(-0.15, 0.35))
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].set_position([0.336506, 0.385487, 0.140968, 0.258545])
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].texts[0].set(position=(0.5, 1.071))
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].texts[1].set(position=(0.5, 1.017))
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].legend(loc=(0.5172, 0.7789), frameon=False)
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].set(position=[0.5479, 0.4808, 0.2188, 0.2177], yticks=[-0.18, -0.12, -0.06, 0., 0.06, 0.12, 0.18, 0.24], yticklabels=['−0.18', '−0.12', '−0.06', '0.00', '0.06', '0.12', '0.18', '0.24'], ylim=(-0.15, 0.19))
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].set_position([0.548255, 0.385487, 0.216904, 0.258545])
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].texts[0].set(position=(0.5, 1.071))
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].texts[1].set(position=(0.5, 1.017))
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].legend(loc=(0.4026, 0.7811), frameon=False)
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].set(position=[0.8381, 0.4808, 0.1422, 0.2177], yticks=[-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4], yticklabels=['−0.2', '−0.1', '0.0', '0.1', '0.2', '0.3', '0.4'], ylim=(-0.15, 0.35))
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].set_position([0.835940, 0.385487, 0.140968, 0.258545])
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].texts[0].set(position=(0.5, 1.071))
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].texts[1].set(position=(0.5, 1.017))
plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].set(position=[0.06391, 0.1678, 0.3865, 0.2206])
plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].set_position([0.068908, 0.014028, 0.382317, 0.261338])
plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].texts[1].set(position=(0.5, 0.9793))
plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].text(0.0217, 0.9793, 'c', transform=plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].texts[2].new
plt.figure(1).ax_dict["D_Flat_Check_gt_NoCheck"].set(position=[0.5543, 0.1682, 0.3858, 0.2202])
plt.figure(1).ax_dict["D_Flat_Check_gt_NoCheck"].set_position([0.555011, 0.014529, 0.381586, 0.260838])
plt.figure(1).ax_dict["D_Flat_Check_gt_NoCheck"].texts[1].set(position=(0.5, 0.9793))
plt.figure(1).ax_dict["E_Terms_Glass"].set(position=[0.0143, 0.7285, 0.9547, 0.2598], xticks=[], xticklabels=[], yticks=[], yticklabels=[])
plt.figure(1).ax_dict["E_Terms_Glass"].set_position([0.020344, 0.680068, 0.944324, 0.307795])
plt.figure(1).ax_dict["E_Terms_Glass"].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.2833, 0.9750, 'All Boards > Baseline', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[0].new
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.5563, 0.9750, 'Checkmate > Non-Checkmate', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[1].new
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.3062, 0.9215, 'Experts > Novices', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, )  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[2].new
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.3062, 0.4753, 'Novices > Experts', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, )  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[3].new
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.6099, 0.9215, 'Experts > Novices', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, )  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[4].new
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.6099, 0.4753, 'Novices > Experts', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, )  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[5].new
plt.figure(1).ax_dict["E_Terms_Glass"].text(0.0096, 0.9750, 'a', transform=plt.figure(1).ax_dict["E_Terms_Glass"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["E_Terms_Glass"].texts[6].new
#% end: automatic generated code from pylustrator

# Display figures in pylustrator GUI for interactive layout adjustment
plt.show()


# =============================================================================
# Save Figures (Individual Axes and Full Panels)
# =============================================================================
# After arranging axes in pylustrator, save both:
# 1. Individual axes as separate SVG/PDF files (for modular figure assembly)
# 2. Full panels as complete SVG/PDF files (for standalone use)

# Save individual axes (one file per axis, named by axis label)
fig = plt.gcf()
save_axes_svgs(fig, FIGURES_DIR, 'neurosynth_univariate')  # e.g., neurosynth_univariate_A1_Corr_All_gt_Rest.svg

# Save full panels (complete multi-axis figure)
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'neurosynth_univariate_panel.pdf')

logger.info("✓ Panel: neurosynth univariate correlations complete")

log_script_end(logger)
