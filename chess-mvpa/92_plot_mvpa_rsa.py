"""
Generate MVPA RSA Figure Panels (Pylustrator)
==============================================

Creates publication-ready multi-panel figures for MVPA ROI-based RSA analysis.
Uses pylustrator for interactive layout arrangement. The script builds
independent axes using standardized plotting primitives and then saves both
individual axes (SVG/PDF) and assembled panels (SVG/PDF) into the current
MVPA RSA results directory.

Figures Produced
----------------

Panel: MVPA RSA Multi-Panel Figure
- File: figures/panels/mvpa_rsa_panel.svg (and .pdf)
- Axes saved to figures/: mvpa_rsa_*.svg and mvpa_rsa_*.pdf
- Content:
  - RSA_1: Visual Similarity RSA - grouped bars showing Expert vs Novice correlations per ROI
  - RSA_2: Strategy RSA - grouped bars showing Expert vs Novice correlations per ROI
  - RSA_3: Checkmate RSA - grouped bars showing Expert vs Novice correlations per ROI
  - RSA_Pial_1/2/3: Left hemisphere pial surface maps showing Δr (Expert - Novice) for FDR-significant ROIs
  - RSA_Pial_Colorbar: Horizontal colorbar for pial surface maps
  - RSA_Pial_Legend: ROI group legend

Inputs
------
- mvpa_group_stats.pkl: Group-level MVPA statistics containing RSA correlations per ROI
  - Dict structure: ['rsa_corr'][target_name]['welch_expert_vs_novice']
  - Contains ROI_Label, mean_diff (Δr), p_val_fdr, group means and CIs
- ROI metadata from CONFIG['ROI_GLASSER_22'] and CONFIG['ROI_GLASSER_180']

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives and style (apply_nature_rc, plot_grouped_bars_on_ax, etc.)
- modules.mvpa_plot_utils for MVPA-specific data extraction helpers
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-mvpa/92_plot_mvpa_rsa.py
"""

import os
import sys
import pickle
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

import numpy as np
import matplotlib.pyplot as plt
from common import setup_script, log_script_end
from common.bids_utils import load_roi_metadata
from common.plotting import (
    apply_nature_rc,
    plot_grouped_boxplots_on_ax,
    compute_whisker_ylim,
    compute_ylim_range,
    PLOT_PARAMS,
    PLOT_YLIMITS,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
    embed_figure_on_ax,
    plot_pial_views_triplet,
    create_standalone_colorbar,
    CMAP_BRAIN,
    create_roi_group_legend,
)
from modules.mvpa_plot_utils import extract_mvpa_bar_data
from common.neuro_utils import (
    load_glasser180_annotations,
    load_glasser180_region_info,
    expand_roi22_to_roi180_values,
    roi_values_to_surface_texture,
)


# =============================================================================
# Configuration and results
# =============================================================================

RESULTS_DIR_NAME = None  # Use latest results directory
RESULTS_BASE = script_dir / "results"

# Define RSA target models to analyze
# These represent different model RDMs used in RSA analysis:
# - visual_similarity: Low-level visual feature dissimilarity
# - strategy: Chess strategy-based dissimilarity
# - checkmate: Binary checkmate vs non-checkmate dissimilarity
MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']

# Pretty titles for each RSA target (for plot labels)
RSA_TITLES = {
    'visual_similarity': 'Visual Similarity RSA',
    'strategy': 'Strategy RSA',
    'checkmate': 'Checkmate RSA',
}


# =============================================================================
# Load MVPA RSA results
# =============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='mvpa_group',
    output_subdirs=['figures'],
    log_name='pylustrator_mvpa_rsa.log',
)
RESULTS_DIR = results_dir
FIGURES_DIR = dirs['figures']


# =============================================================================
# Setup logging
# =============================================================================


# Load group-level MVPA statistics
# Dict structure: group_stats['rsa_corr'][target_name]['welch_expert_vs_novice']
# Contains: ROI_Label, mean_diff (Δr), p_val_fdr, group means and CIs
logger.info("Loading MVPA RSA group statistics...")
with open(RESULTS_DIR / "mvpa_group_stats.pkl", "rb") as f:
    group_stats = pickle.load(f)

# Load ROI metadata for Glasser 22-region parcellation
# Contains: roi_id, pretty_name, color, group/family information
roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])

apply_nature_rc()

# =============================================================================
# Figure: MVPA RSA Multi-Panel Figure - RSA Correlation Barplots/Boxplots
# =============================================================================

# Extract RSA correlation data for all targets (for p-values, label colors, etc.)
rsa_data = extract_mvpa_bar_data(
    group_stats, roi_info, MAIN_TARGETS,
    method='rsa_corr', subtract_chance=False
)

# Load per-subject RSA data for boxplots
from common.io_utils import find_subject_tsvs
from common.bids_utils import get_participants_with_expertise
from modules.mvpa_io import build_group_dataframe

rsa_subject_dir = Path(CONFIG['BIDS_MVPA_RSA'])
roi_col_names = roi_info['roi_name'].tolist()
participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
tsv_files = find_subject_tsvs(rsa_subject_dir)
df_all = build_group_dataframe(tsv_files, participants, roi_col_names)

subject_data = {t: {'experts': [], 'novices': []} for t in MAIN_TARGETS}
for target in MAIN_TARGETS:
    tgt_df = df_all[df_all['target'] == target]
    for _, row in tgt_df.iterrows():
        group = 'experts' if row['expert'] else 'novices'
        vals = row[roi_col_names].values.astype(float)
        subject_data[target][group].append(vals)
for t in MAIN_TARGETS:
    for g in ('experts', 'novices'):
        subject_data[t][g] = np.array(subject_data[t][g])
logger.info(f"Loaded per-subject RSA data: {subject_data['checkmate']['experts'].shape[0]} experts, "
            f"{subject_data['checkmate']['novices'].shape[0]} novices")

# Compute shared whisker-based ylim across all targets
shared_boxplot_ylim = compute_whisker_ylim(
    *[subject_data[t][g] for t in MAIN_TARGETS for g in ('experts', 'novices')]
)

fig1 = plt.figure(1)

for idx, tgt in enumerate(MAIN_TARGETS):
    if tgt not in rsa_data:
        continue

    data = rsa_data[tgt]
    roi_names = data['roi_names']
    roi_colors = data['roi_colors']
    label_colors = data['label_colors']
    x = np.arange(len(roi_names))

    ax = plt.axes()
    ax.set_label(f'RSA_{idx+1}_{tgt}')

    plot_grouped_boxplots_on_ax(
        ax=ax,
        x_positions=x,
        group1_data=subject_data[tgt]['experts'],
        group2_data=subject_data[tgt]['novices'],
        group1_color=roi_colors,
        group2_color=roi_colors,
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=data['pvals'],
        ylim=shared_boxplot_ylim,
        y_label=PLOT_PARAMS['ylabel_correlation_r'],
        subtitle=RSA_TITLES[tgt],
        xtick_labels=roi_names,
        x_label_colors=label_colors,
        x_tick_rotation=30,
        x_tick_align='right',
        show_legend=(idx == 0),
        legend_loc='upper right',
        visible_spines=['left','bottom'],
        params=PLOT_PARAMS
    )

# Create ax_dict for pylustrator convenience
fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}

# =============================================================================
# Figure: MVPA RSA Multi-Panel Figure - Pial Surface Maps
# =============================================================================
# This section creates left hemisphere pial surface maps showing Δr (Expert - Novice)
# for FDR-significant ROIs only. Maps are created for each of the 3 RSA targets.
# Uses Glasser-180 parcellation for fine-grained surface visualization.

# Get FDR threshold from configuration (strict: no fallback)
alpha_fdr = float(CONFIG['ALPHA_FDR'])

# Load surface annotations and Glasser-180 metadata (strict)
labels_l = load_glasser180_annotations('left')
roi180_df = load_glasser180_region_info()

left_textures_by_target = {}
delta_values_by_target = {}
for tgt in MAIN_TARGETS:
    if 'rsa_corr' not in group_stats or tgt not in group_stats['rsa_corr']:
        raise RuntimeError(f"Missing RSA group stats for target '{tgt}'")
    welch = group_stats['rsa_corr'][tgt]['welch_expert_vs_novice']
    if not {'ROI_Label', 'mean_diff', 'p_val_fdr'}.issubset(set(welch.columns)):
        raise RuntimeError(f"Welch dataframe missing required columns for target '{tgt}'")

    roi22_ids = welch['ROI_Label'].to_numpy()
    delta_r = welch['mean_diff'].to_numpy()
    p_fdr = welch['p_val_fdr'].to_numpy()
    include_mask = (p_fdr < alpha_fdr) & np.isfinite(delta_r)

    # Map 22-region deltas (sig-only) to 180 parcels (left hemi)
    roi_ids_180, roi_vals_180 = expand_roi22_to_roi180_values(
        roi22_ids=roi22_ids,
        roi22_values=delta_r,
        roi180_df=roi180_df,
        hemisphere='left',
        include_mask=include_mask,
    )

    # Build per-vertex texture for left hemisphere
    tex_left = roi_values_to_surface_texture(
        labels=labels_l,
        roi_labels=roi_ids_180,
        roi_values=roi_vals_180,
        include_mask=None,
        default_value=np.nan,
    )
    left_textures_by_target[tgt] = tex_left
    delta_values_by_target[tgt] = roi_vals_180

# Shared color scale: anchor at 0 and extend to extrema from significant values
vmin_surf, vmax_surf = compute_ylim_range(
    *[vals for vals in delta_values_by_target.values() if vals is not None],
    symmetric=True,
    zero_anchor=True,
    padding_pct=0.0,
    round_decimals=1,
)
if not np.isfinite(vmin_surf) or not np.isfinite(vmax_surf) or vmax_surf <= vmin_surf:
    raise RuntimeError(f"Invalid pial color range computed: vmin={vmin_surf}, vmax={vmax_surf}")

# Create and embed one left-lateral pial surface per target
for idx, tgt in enumerate(MAIN_TARGETS):
    ax = plt.axes()
    ax.set_label(f'RSA_Pial_{idx+1}_{tgt}')
    fig_pial = plot_pial_views_triplet(
        data={'left': left_textures_by_target[tgt]},
        hemi='left',
        views=('lateral', 'medial', 'ventral'),
        threshold=None,
        vmin=vmin_surf,
        vmax=vmax_surf,
    )
    embed_figure_on_ax(ax, fig_pial)

# Add a standalone horizontal colorbar for Δr across all pial panels
ax_cbar = plt.axes()
ax_cbar.set_label('RSA_Pial_Colorbar')
cbar_fig = create_standalone_colorbar(
    cmap=CMAP_BRAIN,
    vmin=vmin_surf,
    vmax=vmax_surf,
    orientation='horizontal',
    label='Δr (Experts − Novices)'
)
embed_figure_on_ax(ax_cbar, cbar_fig, title='')

# Add a 1-row legend (ROI groups) for pylustrator layout
ax_legend = plt.axes()
ax_legend.set_label('RSA_Pial_Legend')
legend_fig = create_roi_group_legend(
    roi_metadata_path=CONFIG['ROI_GLASSER_22'] / 'region_info.tsv',
    output_path=None,
    single_row=True,
    colorblind=False,
)
embed_figure_on_ax(ax_legend, legend_fig, title='')

# =============================================================================
# Pylustrator Setup and Interactive Layout
# =============================================================================
# Update ax_dict for the figure to enable easy axis reference in pylustrator
# This allows pylustrator to reference axes by label (e.g., fig1.ax_dict["RSA_1_visual_similarity"])
fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}


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
plt.figure(1).set_size_inches(cm_to_inches(11.43), cm_to_inches(16.00), forward=True)
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set(position=[0.07572, 0.7997, 0.4217, 0.1421])
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set_position([0.119538, 0.790607, 0.665730, 0.148551])
plt.figure(1).ax_dict["RSA_2_strategy"].set(position=[0.07572, 0.5055, 0.4217, 0.1421])
plt.figure(1).ax_dict["RSA_2_strategy"].set_position([0.119538, 0.483051, 0.665730, 0.148551])
plt.figure(1).ax_dict["RSA_3_checkmate"].set(position=[0.07572, 0.2113, 0.4217, 0.1421])
plt.figure(1).ax_dict["RSA_3_checkmate"].set_position([0.119538, 0.175495, 0.665730, 0.148551])
plt.figure(1).ax_dict["RSA_Pial_1_visual_similarity"].set(position=[0.5066, 0.7149, 0.07588, 0.2366])
plt.figure(1).ax_dict["RSA_Pial_1_visual_similarity"].set_position([0.799915, 0.702312, 0.119358, 0.246521])
plt.figure(1).ax_dict["RSA_Pial_2_strategy"].set(position=[0.5065, 0.4207, 0.07588, 0.2366])
plt.figure(1).ax_dict["RSA_Pial_2_strategy"].set_position([0.799757, 0.394756, 0.119358, 0.246521])
plt.figure(1).ax_dict["RSA_Pial_3_checkmate"].set(position=[0.5066, 0.1267, 0.07581, 0.2364])
plt.figure(1).ax_dict["RSA_Pial_3_checkmate"].set_position([0.799915, 0.087471, 0.119248, 0.246293])
plt.figure(1).ax_dict["RSA_Pial_Colorbar"].set(position=[0.4769, 0.1086, 0.1385, 0.03607])
plt.figure(1).ax_dict["RSA_Pial_Colorbar"].set_position([0.753298, 0.068206, 0.217931, 0.037581])
plt.figure(1).ax_dict["RSA_Pial_Legend"].set(position=[-0.02237, 0.02747, 0.6129, 0.09074])
plt.figure(1).ax_dict["RSA_Pial_Legend"].set_position([-0.033568, -0.016491, 0.964081, 0.094480])
plt.figure(1).text(0.2317, 0.9836, 'Brain-Model RSA', transform=plt.figure(1).transFigure, fontsize=7., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.365780, 0.982855])
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
save_axes_svgs(fig1, FIGURES_DIR, 'mvpa_rsa')  # e.g., mvpa_rsa_RSA_1_visual_similarity.svg

# Save full panels (complete multi-axis figure)
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'mvpa_rsa_panel.pdf')

logger.info("✓ Panel: MVPA RSA results complete")

log_script_end(logger)
