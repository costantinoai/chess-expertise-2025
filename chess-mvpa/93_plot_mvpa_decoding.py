"""
Generate MVPA Decoding + RSA + RDM Figure Panels (Pylustrator)
===============================================================

Creates publication-ready multi-panel figures combining MVPA ROI-based SVM decoding,
RSA correlations, and model RDM visualizations for the three main dimensions
(visual similarity, strategy, checkmate). Uses pylustrator for interactive layout
arrangement. The script builds independent axes using standardized plotting primitives
and then saves both individual axes (SVG/PDF) and assembled panels (SVG/PDF) into
the current MVPA decoding results directory.

Figures Produced
----------------

Panel: MVPA Decoding + RSA + RDM Multi-Panel Figure
- File: figures/panels/mvpa_svm_panel.svg (and .pdf)
- Axes saved to figures/: mvpa_svm_*.svg and mvpa_svm_*.pdf
- Content:
  - SVM_1-3: SVM Decoding - grouped bars showing Expert vs Novice accuracy per ROI
    (Visual Similarity, Strategy, Checkmate)
  - RSA_1-3: RSA Correlations - grouped bars showing Expert vs Novice correlations per ROI
    (Visual Similarity, Strategy, Checkmate)
  - RDM_1-3: Model RDMs - theoretical model RDMs for each dimension
    (All dimensions: categorical/binary encoding)

Inputs
------
- *_mvpa_group/mvpa_group_stats.pkl: Unified MVPA statistics
  - SVM decoding: ['svm'][target_name]['welch_expert_vs_novice']
  - RSA correlations: ['rsa_corr'][target_name]['welch_expert_vs_novice']
- Each contains: ROI_Label, mean_diff, p_val_fdr, group means and CIs
- ROI metadata from CONFIG['ROI_GLASSER_22']
- Stimulus metadata from CONFIG['STIMULI_FILE'] for model RDM construction

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives and style (apply_nature_rc, plot_grouped_bars_on_ax, plot_rdm_on_ax, etc.)
- common.rsa_utils for model RDM creation
- modules.mvpa_plot_utils for MVPA-specific data extraction helpers
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-mvpa/93_plot_mvpa_decoding.py
"""

import pickle
from pathlib import Path
script_dir = Path(__file__).parent

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import numpy as np
import matplotlib.pyplot as plt
from common import setup_script, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata, load_stimulus_metadata
from common.rsa_utils import create_model_rdm
from common.plotting import (
    apply_nature_rc,
    plot_grouped_boxplots_on_ax,
    compute_whisker_ylim,
    plot_rdm_on_ax,
    compute_stimulus_palette,
    PLOT_PARAMS,
    PLOT_YLIMITS,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
)
from analyses.mvpa.plot_utils import extract_mvpa_bar_data


# =============================================================================
# Configuration and results
# =============================================================================

RESULTS_DIR_NAME = None  # Use latest results directory
RESULTS_BASE = script_dir / "results"

# Define SVM decoding target classes to analyze
# These represent different binary classification tasks:
# - visual_similarity: High vs low visual similarity stimuli
# - strategy: Different chess strategy types
# - checkmate: Checkmate vs non-checkmate stimuli
MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']

# Pretty titles for each target (for plot labels)
SVM_TITLES = {
    'visual_similarity': 'Visual Similarity',
    'strategy': 'Strategy',
    'checkmate': 'Checkmate',
}

RSA_TITLES = {
    'visual_similarity': 'Visual Similarity',
    'strategy': 'Strategy',
    'checkmate': 'Checkmate',
}

# =============================================================================
# Load MVPA results from both decoding and RSA directories
# =============================================================================

# Find latest MVPA group decoding results directory
# Creates 'figures' subdirectory if needed for saving outputs
dec_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='mvpa_group',
    output_subdirs=['figures'],
    log_name='pylustrator_mvpa_decoding.log',
)
DECODING_RESULTS_DIR = dec_dir

# Save outputs to the unified MVPA group directory
RESULTS_DIR = DECODING_RESULTS_DIR
FIGURES_DIR = dirs['figures']


# =============================================================================
# Setup logging
# =============================================================================


# Load group-level MVPA statistics from unified directory
# group_stats['svm'][target_name]['welch_expert_vs_novice'] (decoding)
# group_stats['rsa_corr'][target_name]['welch_expert_vs_novice'] (RSA)
logger.info("Loading MVPA RSA + SVM group statistics from unified directory...")
with open(RESULTS_DIR / "mvpa_group_stats.pkl", "rb") as f:
    group_stats = pickle.load(f)

# Load ROI metadata for Glasser 22-region parcellation
# Contains: roi_id, pretty_name, color, group/family information
roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])

# Load stimulus metadata and create model RDMs for visualization
# stimuli_df: DataFrame with stimulus info (checkmate status, strategy type, visual similarity)
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

apply_nature_rc()

# =============================================================================
# Figure: MVPA Decoding + RSA + RDM Multi-Panel Figure
# =============================================================================

# Extract group-level data (for p-values, label colors, etc.)
svm_data = extract_mvpa_bar_data(
    group_stats, roi_info, MAIN_TARGETS, method='svm', subtract_chance=True
)
rsa_data = extract_mvpa_bar_data(
    group_stats, roi_info, MAIN_TARGETS, method='rsa_corr', subtract_chance=False
)

# Load per-subject data for boxplots
import pandas as pd
from common.io_utils import find_subject_tsvs
from common.bids_utils import get_participants_with_expertise
from analyses.mvpa.io import build_group_dataframe

roi_col_names = roi_info['roi_name'].tolist()
participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)

# Per-subject RSA
rsa_files = find_subject_tsvs(Path(CONFIG['BIDS_MVPA_RSA']))
rsa_df = build_group_dataframe(rsa_files, participants, roi_col_names)
rsa_subject = {t: {'experts': [], 'novices': []} for t in MAIN_TARGETS}
for t in MAIN_TARGETS:
    for _, row in rsa_df[rsa_df['target'] == t].iterrows():
        g = 'experts' if row['expert'] else 'novices'
        rsa_subject[t][g].append(row[roi_col_names].values.astype(float))
for t in MAIN_TARGETS:
    for g in ('experts', 'novices'):
        rsa_subject[t][g] = np.array(rsa_subject[t][g])

# Per-subject SVM decoding (subtract chance)
svm_files = find_subject_tsvs(Path(CONFIG['BIDS_MVPA_DECODING']))
svm_df = build_group_dataframe(svm_files, participants, roi_col_names)
svm_subject = {t: {'experts': [], 'novices': []} for t in MAIN_TARGETS}
for t in MAIN_TARGETS:
    chance = group_stats['svm'][t].get('chance', 0.5) if isinstance(group_stats['svm'][t], dict) else 0.5
    for _, row in svm_df[svm_df['target'] == t].iterrows():
        g = 'experts' if row['expert'] else 'novices'
        svm_subject[t][g].append(row[roi_col_names].values.astype(float) - chance)
for t in MAIN_TARGETS:
    for g in ('experts', 'novices'):
        svm_subject[t][g] = np.array(svm_subject[t][g])

# Shared whisker-based ylim per method
svm_boxplot_ylim = compute_whisker_ylim(
    *[svm_subject[t][g] for t in MAIN_TARGETS for g in ('experts', 'novices')]
)
rsa_boxplot_ylim = compute_whisker_ylim(
    *[rsa_subject[t][g] for t in MAIN_TARGETS for g in ('experts', 'novices')]
)
logger.info(f"Loaded per-subject data: SVM {svm_subject['checkmate']['experts'].shape}, RSA {rsa_subject['checkmate']['experts'].shape}")

fig1 = plt.figure(1)

# --- SVM Decoding panels ---
for idx, tgt in enumerate(MAIN_TARGETS):
    if tgt not in svm_data:
        continue
    data = svm_data[tgt]
    roi_names = data['roi_names']
    roi_colors = data['roi_colors']
    label_colors = data['label_colors']
    x = np.arange(len(roi_names))

    ax = plt.axes()
    ax.set_label(f'SVM_{idx+1}_{tgt}')

    plot_grouped_boxplots_on_ax(
        ax=ax, x_positions=x,
        group1_data=svm_subject[tgt]['experts'],
        group2_data=svm_subject[tgt]['novices'],
        group1_color=roi_colors, group2_color=roi_colors,
        group1_label='Experts', group2_label='Novices',
        comparison_pvals=data['pvals'],
        ylim=svm_boxplot_ylim,
        y_label='Accuracy - chance',
        xtick_labels=roi_names, x_label_colors=label_colors,
        x_tick_rotation=30, x_tick_align='right',
        show_legend=(idx == 0), legend_loc='upper right',
        visible_spines=['left','bottom'], params=PLOT_PARAMS
    )

# --- RSA Correlation panels ---
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
        ax=ax, x_positions=x,
        group1_data=rsa_subject[tgt]['experts'],
        group2_data=rsa_subject[tgt]['novices'],
        group1_color=roi_colors, group2_color=roi_colors,
        group1_label='Experts', group2_label='Novices',
        comparison_pvals=data['pvals'],
        ylim=rsa_boxplot_ylim,
        y_label=PLOT_PARAMS['ylabel_correlation_r'],
        xtick_labels=roi_names, x_label_colors=label_colors,
        x_tick_rotation=30, x_tick_align='right',
        show_legend=(idx == 0), legend_loc='upper right',
        visible_spines=['left','bottom'], params=PLOT_PARAMS
    )

# =============================================================================
# Figure: Model RDMs for Three Dimensions
# =============================================================================
# This section creates RDM visualizations for the three model dimensions
# (visual similarity, strategy, checkmate)

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
# This allows pylustrator to reference axes by label (e.g., fig1.ax_dict["SVM_1_visual_similarity"])
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
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(18.290000/2.54, 13.500000/2.54, forward=True)
plt.figure(1).ax_dict["RDM_1_visual_similarity"].set(position=[0.9216, 0.8861, 0.04483, 0.05118])
plt.figure(1).ax_dict["RDM_1_visual_similarity"].set_position([0.921061, 0.904012, 0.044527, 0.060310])
plt.figure(1).ax_dict["RDM_2_strategy"].set(position=[0.9216, 0.6217, 0.04483, 0.05118])
plt.figure(1).ax_dict["RDM_2_strategy"].set_position([0.921061, 0.592256, 0.044527, 0.060310])
plt.figure(1).ax_dict["RDM_3_checkmate"].set(position=[0.9216, 0.3572, 0.04483, 0.05118])
plt.figure(1).ax_dict["RDM_3_checkmate"].set_position([0.921061, 0.280499, 0.044527, 0.060310])
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set(position=[0.5488, 0.7902, 0.4176, 0.1321])
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set_position([0.548513, 0.790941, 0.417204, 0.155756])
plt.figure(1).ax_dict["RSA_1_visual_similarity"].get_legend().set(visible=False)
plt.figure(1).ax_dict["RSA_1_visual_similarity"].text(0.5000, 1.1414, 'Brain-Model RSA', transform=plt.figure(1).ax_dict["RSA_1_visual_similarity"].transAxes, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["RSA_1_visual_similarity"].texts[0].new
plt.figure(1).ax_dict["RSA_1_visual_similarity"].text(-0.0224, 1.1414, 'c', transform=plt.figure(1).ax_dict["RSA_1_visual_similarity"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["RSA_1_visual_similarity"].texts[1].new
plt.figure(1).ax_dict["RSA_1_visual_similarity"].text(0.5000, 0.9358, 'Visual Similarity', transform=plt.figure(1).ax_dict["RSA_1_visual_similarity"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["RSA_1_visual_similarity"].texts[2].new
plt.figure(1).ax_dict["RSA_1_visual_similarity"].title.set(visible=False)
plt.figure(1).ax_dict["RSA_2_strategy"].set(position=[0.5488, 0.5217, 0.4176, 0.1321])
plt.figure(1).ax_dict["RSA_2_strategy"].set_position([0.548513, 0.474390, 0.417204, 0.155756])
plt.figure(1).ax_dict["RSA_2_strategy"].text(0.5000, 0.9074, 'Strategy', transform=plt.figure(1).ax_dict["RSA_2_strategy"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["RSA_2_strategy"].texts[10].new
plt.figure(1).ax_dict["RSA_2_strategy"].title.set(visible=False)
plt.figure(1).ax_dict["RSA_3_checkmate"].set(position=[0.5488, 0.2531, 0.4176, 0.1321])
plt.figure(1).ax_dict["RSA_3_checkmate"].set_position([0.548513, 0.157711, 0.417204, 0.155756])
plt.figure(1).ax_dict["RSA_3_checkmate"].text(0.5000, 0.9038, 'Checkmate', transform=plt.figure(1).ax_dict["RSA_3_checkmate"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["RSA_3_checkmate"].texts[8].new
plt.figure(1).ax_dict["RSA_3_checkmate"].title.set(visible=False)
plt.figure(1).ax_dict["SVM_1_visual_similarity"].set(position=[0.06112, 0.7902, 0.4176, 0.1321])
plt.figure(1).ax_dict["SVM_1_visual_similarity"].set_position([0.061330, 0.790911, 0.417204, 0.155756])
plt.figure(1).ax_dict["SVM_1_visual_similarity"].text(0.4773, 1.1416, 'Brain Decoding', transform=plt.figure(1).ax_dict["SVM_1_visual_similarity"].transAxes, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["SVM_1_visual_similarity"].texts[5].new
plt.figure(1).ax_dict["SVM_1_visual_similarity"].text(-0.0386, 1.1416, 'a', transform=plt.figure(1).ax_dict["SVM_1_visual_similarity"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["SVM_1_visual_similarity"].texts[6].new
plt.figure(1).ax_dict["SVM_1_visual_similarity"].text(0.4773, 0.9360, 'Visual Similarity', transform=plt.figure(1).ax_dict["SVM_1_visual_similarity"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["SVM_1_visual_similarity"].texts[7].new
plt.figure(1).ax_dict["SVM_1_visual_similarity"].title.set(visible=False)
plt.figure(1).ax_dict["SVM_2_strategy"].set(position=[0.06112, 0.5216, 0.4176, 0.1321])
plt.figure(1).ax_dict["SVM_2_strategy"].set_position([0.061330, 0.474296, 0.417204, 0.155756])
plt.figure(1).ax_dict["SVM_2_strategy"].text(0.4773, 0.9080, 'Strategy', transform=plt.figure(1).ax_dict["SVM_2_strategy"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["SVM_2_strategy"].texts[12].new
plt.figure(1).ax_dict["SVM_2_strategy"].title.set(visible=False)
plt.figure(1).ax_dict["SVM_3_checkmate"].set(position=[0.06112, 0.253, 0.4176, 0.1321])
plt.figure(1).ax_dict["SVM_3_checkmate"].set_position([0.061330, 0.157681, 0.417204, 0.155756])
plt.figure(1).ax_dict["SVM_3_checkmate"].text(0.4773, 0.9040, 'Checkmate', transform=plt.figure(1).ax_dict["SVM_3_checkmate"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["SVM_3_checkmate"].texts[13].new
plt.figure(1).ax_dict["SVM_3_checkmate"].title.set(visible=False)
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
save_axes_svgs(fig1, FIGURES_DIR, 'mvpa_svm')  # e.g., mvpa_svm_SVM_1_visual_similarity.svg

# Save full panels (complete multi-axis figure)
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'mvpa_svm_panel.pdf')

logger.info("✓ Panel: MVPA SVM decoding + RSA + RDM panels complete")

log_script_end(logger)
