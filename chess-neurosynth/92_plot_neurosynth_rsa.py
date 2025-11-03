"""
Generate Neurosynth RSA Figure Panels (Pylustrator)
====================================================

Creates publication-ready multi-panel figures for Neurosynth RSA searchlight analysis.
Uses pylustrator for interactive layout arrangement. The script builds
independent axes using standardized plotting primitives and then saves both
individual axes (SVG/PDF) and assembled panels (SVG/PDF) into the current
neurosynth RSA results directory.

Figures Produced
----------------

Panel: Neurosynth RSA Searchlight Multi-Panel Figure
- File: figures/panels/neurosynth_rsa_panel.svg (and .pdf)
- Axes saved to figures/: neurosynth_rsa_*.svg and neurosynth_rsa_*.pdf
- Content (for each of 3 patterns: Checkmate, Strategy, Visual Similarity):
  - A: Term correlations - grouped bars showing POS vs NEG Neurosynth term correlations
  - B: Correlation differences - bars showing POS - NEG differences
  - C: Flat surface maps - left and right hemisphere projections of RSA searchlight maps

Inputs
------
- CSV files with term correlation results for each pattern:
  - searchlight_checkmate_pos_term_corr.csv, *_neg_term_corr.csv, *_diff_term_corr.csv
  - searchlight_strategy_pos_term_corr.csv, *_neg_term_corr.csv, *_diff_term_corr.csv
  - searchlight_visual_similarity_pos_term_corr.csv, *_neg_term_corr.csv, *_diff_term_corr.csv
- zmap_searchlight_*.nii.gz: Statistical z-maps for surface plotting (Expert > Novice RSA)

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives and style (apply_nature_rc, plot_grouped_bars_on_ax, etc.)
- modules.plot_utils for Neurosynth-specific plotting helpers
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-neurosynth/92_plot_neurosynth_rsa.py
"""

import os
import sys
from pathlib import Path
script_dir = Path(__file__).parent

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import matplotlib.pyplot as plt
from nilearn import image
from common.bids_utils import load_stimulus_metadata
from common.rsa_utils import create_model_rdm

from common.plotting import (
    apply_nature_rc,
    save_axes_svgs,
    save_panel_pdf,
    compute_ylim_range,
    plot_flat_pair,
    embed_figure_on_ax,
    compute_stimulus_palette,
    plot_rdm_on_ax,
    PLOT_YLIMITS,
    cm_to_inches,
)
from common.neuro_utils import project_volume_to_surfaces
from common import setup_script, log_script_end
from modules.plot_utils import (
    plot_correlations_on_ax,
    plot_differences_on_ax,
    load_term_corr_triple,
)


# Define RSA patterns to analyze
# Each pattern represents a different model RDM used in RSA searchlight analysis:
# - Checkmate: Binary checkmate vs non-checkmate dissimilarity
# - Strategy: Chess strategy-based dissimilarity
# - Visual Similarity: Low-level visual feature dissimilarity
# Tuple format: (file_stem, pretty_label_for_plots)
PATTERNS = [
    ('searchlight_checkmate', 'Checkmate | RSA searchlight'),
    ('searchlight_strategy', 'Strategy | RSA searchlight'),
    ('searchlight_visual_similarity', 'Visual Similarity | RSA searchlight'),
]


# NOTE: Helper functions moved to modules.plot_utils for reusability


# =============================================================================
# Configuration and results
# =============================================================================

apply_nature_rc()

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='neurosynth_rsa',
    output_subdirs=['figures'],
    log_name='pylustrator_neurosynth_rsa.log',
)
RESULTS_DIR = results_dir
FIGURES_DIR = dirs['figures']


# =============================================================================
# Setup logging
# =============================================================================

extra = {"RESULTS_DIR": str(RESULTS_DIR), "FIGURES_DIR": str(FIGURES_DIR)}



# =============================================================================
# Figure: Neurosynth RSA Searchlight Multi-Panel Figure
# =============================================================================
# This figure shows Neurosynth RSA searchlight analysis results for 3 patterns:
# - Pattern 1: Checkmate (binary checkmate vs non-checkmate dissimilarity)
# - Pattern 2: Strategy (chess strategy-based dissimilarity)
# - Pattern 3: Visual Similarity (low-level visual feature dissimilarity)
# For each pattern, shows term correlations, differences, and surface maps

fig = plt.figure(1)

# Load all volumes and project to surfaces ONCE (DRY: no re-projection!)
z_imgs = {}
surface_textures = {}
for stem, _pretty in PATTERNS:
    z_img = image.load_img(str(RESULTS_DIR / f"zmap_{stem}.nii.gz"))
    z_imgs[stem] = z_img
    # Project each volume to surfaces once
    tex_l, tex_r = project_volume_to_surfaces(z_img)
    surface_textures[stem] = (tex_l, tex_r)

# Compute global symmetric color range from ALL surface textures
# Use same vmin/vmax across all patterns for consistent color interpretation
all_textures = [tex for tex_pair in surface_textures.values() for tex in tex_pair]
vmin_rsa, vmax_rsa = compute_ylim_range(
    *all_textures,
    symmetric=True,
    padding_pct=0.0
)

# Create panels for each RSA pattern (Checkmate, Strategy, Visual Similarity)
for idx, (stem, pretty) in enumerate(PATTERNS, start=1):
    # Load term correlation data for this pattern
    # Each triple contains: (pos_df, neg_df, diff_df)
    # - pos_df: Positive correlations (Expert > Novice RSA regions)
    # - neg_df: Negative correlations (Novice > Expert RSA regions)
    # - diff_df: Difference (POS - NEG)
    df_pos, df_neg, df_diff = load_term_corr_triple(RESULTS_DIR, stem)


    # -------------------------------------------------------------------------
    # Panel {idx}A: Term Correlations for this pattern
    # -------------------------------------------------------------------------
    # Grouped barplot showing correlations between top Neurosynth terms and RSA map
    # POS = Expert > Novice RSA regions (positive z-values)
    # NEG = Novice > Expert RSA regions (negative z-values)
    # Shows which cognitive terms are associated with expertise-related RSA patterns
    ax_corr = plt.axes(); ax_corr.set_label(f'{idx}A_RSA_Corr_{stem}')

    # Extract dimension name from pretty label (e.g., "Checkmate" from "Checkmate | RSA searchlight")
    dimension = pretty.split('|')[0].strip()

    plot_correlations_on_ax(
        ax_corr,
        df_pos,                         # Positive term correlations (Expert > Novice)
        df_neg,                         # Negative term correlations (Novice > Expert)
        title="",
        subtitle=dimension,
        ylim=PLOT_YLIMITS['rsa_neurosynth_corr']  # Centralized neurosynth RSA correlation limits
    )

    # -------------------------------------------------------------------------
    # Panel {idx}B: Correlation Differences for this pattern
    # -------------------------------------------------------------------------
    # Barplot showing difference between POS and NEG correlations (POS - NEG)
    # Positive values = term more associated with Expert > Novice RSA regions
    # Negative values = term more associated with Novice > Expert RSA regions
    ax_diff = plt.axes(); ax_diff.set_label(f'{idx}B_RSA_Diff_{stem}')
    plot_differences_on_ax(
        ax_diff,
        df_diff,                        # Correlation differences (POS - NEG)
        title="",
        subtitle=dimension,
        ylim=PLOT_YLIMITS['rsa_neurosynth_diff']  # Centralized neurosynth RSA difference limits
    )

    # -------------------------------------------------------------------------
    # Panel {idx}C: Flat Surface Maps for this pattern
    # -------------------------------------------------------------------------
    # Left and right hemisphere flat surface projections of RSA searchlight z-map
    # Shows where in the brain this pattern RDM correlates with neural RDMs
    # Warm colors = Expert > Novice RSA, Cool colors = Novice > Expert RSA
    ax_flat = plt.axes(); ax_flat.set_label(f'{idx}C_RSA_Flat_{stem}')

    # Get pre-computed textures for this pattern (no re-projection!)
    tex_l, tex_r = surface_textures[stem]

    # Create flat surface figure using pre-computed textures
    surface_fig_rsa = plot_flat_pair(
        textures=(tex_l, tex_r),  # Pre-computed surface textures
        title='',                  # No title (added by embed_figure_on_ax)
        threshold=0,               # Show all values (no thresholding)
        output_file=None,          # Don't save (will embed in main figure)
        show_hemi_labels=False,    # No hemisphere labels (added manually)
        show_colorbar=False,       # Colorbar shown separately
        vmin=vmin_rsa,             # Symmetric minimum (same across all patterns)
        vmax=vmax_rsa,             # Symmetric maximum (same across all patterns)
        show_directions=True,      # Show anterior/posterior labels
    )
    embed_figure_on_ax(ax_flat, surface_fig_rsa, title="", subtitle=dimension)

# =============================================================================
# Figure: Model RDMs for Three Dimensions
# =============================================================================
# This section creates RDM visualizations for the three model dimensions
# (visual similarity, strategy, checkmate)
# Load stimulus metadata and create model RDMs for visualization
# stimuli_df: DataFrame with stimulus info (checkmate status, strategy type, visual similarity)
MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']
stimuli_df = load_stimulus_metadata()
stim_colors, stim_alphas = compute_stimulus_palette(stimuli_df)

# Create model RDMs for the three target dimensions
# All dimensions are treated as categorical (Hamming distance: 0=same category, 1=different)
model_rdms = {}
model_rdms['visual_similarity'] = create_model_rdm(stimuli_df['visual'].values, is_categorical=True)
model_rdms['strategy'] = create_model_rdm(stimuli_df['strategy'].values, is_categorical=True)
model_rdms['checkmate'] = create_model_rdm(
    (stimuli_df['check'] == 'checkmate').astype(int).values,
    is_categorical=True
)

# Compute global symmetric color scale for RDM panels
# Use same vmin/vmax across all model RDMs for consistent color interpretation
rdm_vmin = 0
rdm_vmax = max([rdm.max() for rdm in model_rdms.values()])

# Create one RDM panel for each target dimension
for idx, tgt in enumerate(MAIN_TARGETS):
    # -------------------------------------------------------------------------
    # Panel RDM_{idx+1}: Model RDM for {target}
    # -------------------------------------------------------------------------
    # Shows the theoretical model RDM for this dimension
    # Matrix is 40x40, with values representing dissimilarity between stimuli
    # All dimensions use categorical encoding: binary (0=same category, 1=different category)
    ax = plt.axes()
    ax.set_label(f'RDM_{idx+1}_{tgt}')

    plot_rdm_on_ax(
        ax=ax,
        rdm=model_rdms[tgt],         # Model RDM (40x40)
        colors=stim_colors,          # Strategy-based colors for matrix borders
        alphas=stim_alphas,          # Alpha values for each stimulus
        show_colorbar=False,         # Don't show individual colorbars
    )
    ax.set_xticks([])
    ax.set_yticks([])  # Hide tick labels (too many stimuli)

# =============================================================================
# Pylustrator Setup and Interactive Layout
# =============================================================================
# Create ax_dict for the figure to enable easy axis reference in pylustrator
# This allows pylustrator to reference axes by label (e.g., fig.ax_dict["1A_RSA_Corr_searchlight_checkmate"])
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
plt.figure(1).set_size_inches(cm_to_inches(16.67), cm_to_inches(15.02), forward=True)
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].legend(loc=(0.4579, 0.808), frameon=False)
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].set(position=[0.4549, 0.1873, 0.2174, 0.1699])
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].set_position([0.496846, 0.104319, 0.234715, 0.187779])
plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].set(position=[0.7527, 0.1873, 0.1487, 0.1699])
plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].set_position([0.818364, 0.104319, 0.160543, 0.187779])
plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].set(position=[0.009016, 0.1395, 0.3802, 0.2176])
plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].set_position([0.017607, 0.052023, 0.406167, 0.239429])
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].legend(loc=(0.4483, 0.808), frameon=False)
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].set(position=[0.4568, 0.4725, 0.2136, 0.1699])
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].set_position([0.498897, 0.419532, 0.230612, 0.187779])
plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].set(position=[0.7527, 0.4725, 0.1487, 0.1699])
plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].set_position([0.818364, 0.419532, 0.160543, 0.187779])
plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].set(position=[0.009016, 0.428, 0.3745, 0.2144])
plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].set_position([0.017574, 0.370910, 0.400078, 0.235839])
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].legend(loc=(0.4543, 0.808), frameon=False)
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].set(position=[0.4549, 0.7577, 0.2174, 0.1699])
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].set_position([0.496846, 0.734745, 0.234715, 0.187779])
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].text(0.0960, 1.2573, 'Neurosynth-RSA Correlations', transform=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].texts[0].new
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].text(-0.0801, 1.2520, 'b', transform=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visual_similarity"].texts[1].new
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].set(position=[0.7527, 0.7577, 0.1487, 0.1699])
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].set_position([0.818364, 0.734745, 0.160543, 0.187779])
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].text(0.0370, 1.2573, 'Correlation Difference', transform=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].texts[0].new
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].text(-0.1373, 1.2520, 'c', transform=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visual_similarity"].texts[1].new
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].set(position=[0.009016, 0.7132, 0.3745, 0.2144])
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].set_position([0.017574, 0.686123, 0.400078, 0.235839])
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].text(0.3458, 1.2039, 'Surface Projection', transform=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].texts[0].new
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].text(0.0982, 1.2081, 'a', transform=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visual_similarity"].texts[1].new
plt.figure(1).ax_dict["RDM_1_visual_similarity"].set(position=[0.3634, 0.7363, 0.04401, 0.04724])
plt.figure(1).ax_dict["RDM_1_visual_similarity"].set_position([0.398304, 0.711159, 0.046999, 0.052151])
plt.figure(1).ax_dict["RDM_2_strategy"].set(position=[0.3634, 0.4475, 0.04401, 0.04724])
plt.figure(1).ax_dict["RDM_2_strategy"].set_position([0.398304, 0.391973, 0.046999, 0.052151])
plt.figure(1).ax_dict["RDM_3_checkmate"].set(position=[0.3634, 0.1587, 0.04401, 0.04724])
plt.figure(1).ax_dict["RDM_3_checkmate"].set_position([0.398304, 0.072787, 0.046999, 0.052151])
#% end: automatic generated code from pylustrator

# Display figures in pylustrator GUI for interactive layout adjustment
if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()


# =============================================================================
# Save Figures (Individual Axes and Full Panels)
# =============================================================================
# After arranging axes in pylustrator, save both:
# 1. Individual axes as separate SVG/PDF files (for modular figure assembly)
# 2. Full panels as complete SVG/PDF files (for standalone use)

# Save individual axes (one file per axis, named by axis label)
fig = plt.gcf()
save_axes_svgs(fig, FIGURES_DIR, 'neurosynth_rsa')  # e.g., neurosynth_rsa_1A_RSA_Corr_searchlight_checkmate.svg

# Save full panels (complete multi-axis figure)
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'neurosynth_rsa_panel.pdf')

logger.info("✓ Panel: neurosynth RSA searchlight complete")

log_script_end(logger)
