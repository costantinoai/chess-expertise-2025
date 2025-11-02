"""
Generate Behavioral RSA Figure Panels (Pylustrator)
====================================================

Creates publication-ready multi-panel figures for behavioral RSA analysis.
Uses pylustrator for interactive layout arrangement. The script builds
independent axes using standardized plotting primitives and then saves both
individual axes (SVG/PDF) and assembled panels (SVG/PDF) into the current
behavioral results directory.

Figures Produced
----------------

Panel: Behavioral RSA Multi-Panel Figure
- File: figures/panels/behavioral_rsa_panel.svg (and .pdf)
- Axes saved to figures/: behavioral_*.svg and behavioral_*.pdf
- Content:
  - A1: Directional preference matrix (Experts) - shows behavioral preference dissimilarity
  - A2: Directional preference matrix (Novices) - shows behavioral preference dissimilarity
  - A3: Behavioral RDM (Experts) - representational dissimilarity matrix
  - A4: Behavioral RDM (Novices) - representational dissimilarity matrix
  - B1: MDS embedding (Experts) - 2D projection of behavioral RDM
  - B2: MDS embedding (Novices) - 2D projection of behavioral RDM
  - C1: Stimulus selection frequency (Experts) - choice histogram
  - C2: Stimulus selection frequency (Novices) - choice histogram
  - D: Behavioral-model RSA correlations - barplot comparing model fits
  - E: Symmetric colorbar for RDM/DSM panels

Inputs
------
- expert_behavioral_rdm.npy: Expert group behavioral RDM (40x40)
- novice_behavioral_rdm.npy: Novice group behavioral RDM (40x40)
- expert_directional_dsm.npy: Expert directional preference matrix (40x40)
- novice_directional_dsm.npy: Novice directional preference matrix (40x40)
- expert_mds_coords.npy: Expert MDS 2D coordinates (40x2)
- novice_mds_coords.npy: Novice MDS 2D coordinates (40x2)
- pairwise_data.pkl: Pairwise comparison data (expert_pairwise, novice_pairwise DataFrames)
- correlation_results.pkl: RSA model correlation results with FDR correction

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives and style (apply_nature_rc, plot_rdm_on_ax, etc.)
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-behavioral/91_plot_behavioral_panels.py
"""

import sys
import os
import pickle
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

import numpy as np
import matplotlib.pyplot as plt

from common import load_stimulus_metadata, MODEL_ORDER, MODEL_LABELS_PRETTY
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.plotting import (
    apply_nature_rc,
    plot_rdm_on_ax,
    plot_2d_embedding_on_ax,
    plot_counts_on_ax,
    plot_grouped_bars_on_ax,
    set_axis_title,
    compute_stimulus_palette,
    COLORS_EXPERT_NOVICE,
    PLOT_PARAMS,
    save_axes_svgs,
    save_panel_pdf,
    CMAP_BRAIN,
    create_standalone_colorbar,
    embed_figure_on_ax,
)


# =============================================================================
# Configuration and results
# =============================================================================

RESULTS_DIR_NAME = None  # Use latest results directory
RESULTS_BASE = script_dir / "results"

# Find latest behavioral RSA results directory
# Creates 'figures' subdirectory if needed for saving outputs
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern="*_behavioral_rsa",  # Match behavioral analysis results
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=["figures"],
    require_exists=True,
    verbose=True,
)

FIGURES_DIR = RESULTS_DIR / "figures"


# =============================================================================
# Setup logging
# =============================================================================

extra = {"RESULTS_DIR": str(RESULTS_DIR), "FIGURES_DIR": str(FIGURES_DIR)}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="pylustrator_behavioral_layout.log",
)


# =============================================================================
# Load behavioral data
# =============================================================================

logger.info("Loading behavioral results...")

# Load representational dissimilarity matrices (RDMs) and directional preference matrices (DSMs)
# - RDMs: 40x40 matrices showing behavioral dissimilarity between all stimulus pairs
# - DSMs: 40x40 matrices showing directional preference (which stimulus preferred over which)
expert_rdm = np.load(RESULTS_DIR / "expert_behavioral_rdm.npy")      # Expert behavioral RDM
novice_rdm = np.load(RESULTS_DIR / "novice_behavioral_rdm.npy")      # Novice behavioral RDM
expert_dsm = np.load(RESULTS_DIR / "expert_directional_dsm.npy")     # Expert directional preferences
novice_dsm = np.load(RESULTS_DIR / "novice_directional_dsm.npy")     # Novice directional preferences

# Load MDS embeddings (2D projections of behavioral RDMs)
# Shape: (40, 2) - each stimulus has (x, y) coordinates in 2D space
expert_mds_coords = np.load(RESULTS_DIR / "expert_mds_coords.npy")   # Expert MDS coordinates
novice_mds_coords = np.load(RESULTS_DIR / "novice_mds_coords.npy")   # Novice MDS coordinates

# Load pairwise comparison data (choice behavior)
# Dict with 'expert_pairwise' and 'novice_pairwise' DataFrames
# Each DataFrame has columns: better (chosen stimulus), worse (rejected stimulus), count (frequency)
with open(RESULTS_DIR / "pairwise_data.pkl", "rb") as f:
    pairwise_data = pickle.load(f)
expert_pairwise = pairwise_data["expert_pairwise"]  # Expert choice data
novice_pairwise = pairwise_data["novice_pairwise"]  # Novice choice data

# Load RSA model correlation results
# Dict with 'expert' and 'novice' lists of (model_name, r, p, ci_low, ci_high)
# Also includes 'expert_p_fdr' and 'novice_p_fdr' dicts for FDR-corrected p-values
with open(RESULTS_DIR / "correlation_results.pkl", "rb") as f:
    correlation_results = pickle.load(f)
expert_corrs = correlation_results["expert"]                        # Expert RSA correlations
novice_corrs = correlation_results["novice"]                        # Novice RSA correlations
exp_p_fdr_map = correlation_results.get("expert_p_fdr")            # Expert FDR-corrected p-values (dict)
nov_p_fdr_map = correlation_results.get("novice_p_fdr")            # Novice FDR-corrected p-values (dict)

# Load stimulus metadata and compute color palette
# stimuli_df: DataFrame with stimulus info (checkmate status, strategy type)
# strat_colors: list of colors for each stimulus (based on strategy)
# strat_alphas: list of alpha values for each stimulus
stimuli_df = load_stimulus_metadata()
strat_colors, strat_alphas = compute_stimulus_palette(stimuli_df)

# Compute global symmetric color scale for RDM/DSM panels
# Use same vmin/vmax across all matrices for consistent color interpretation
# Centered at 0 with symmetric range (diverging colormap)
global_max_abs = max(
    np.abs(expert_rdm).max(),    # Maximum absolute value in expert RDM
    np.abs(novice_rdm).max(),    # Maximum absolute value in novice RDM
    np.abs(expert_dsm).max(),    # Maximum absolute value in expert DSM
    np.abs(novice_dsm).max()     # Maximum absolute value in novice DSM
)
behavioral_vmin = -global_max_abs  # Symmetric minimum
behavioral_vmax = global_max_abs   # Symmetric maximum

logger.info("Behavioral data loaded successfully\n")


# =============================================================================
# Style
# =============================================================================

apply_nature_rc()


# =============================================================================
# Figure: Behavioral RSA Multi-Panel Figure
# =============================================================================
# This figure shows the behavioral RSA analysis results:
# - Panels A: Directional preference matrices and behavioral RDMs (Experts vs Novices)
# - Panels B: MDS embeddings showing 2D projection of behavioral spaces
# - Panels C: Stimulus selection frequency histograms
# - Panel D: RSA correlations between behavioral and model RDMs
# - Panel E: Colorbar for RDM/DSM panels

fig = plt.figure(1)

# -----------------------------------------------------------------------------
# Panel A3: Directional Preference Matrix (Experts)
# -----------------------------------------------------------------------------
# Shows directional preference dissimilarity: which stimuli are preferred over which
# Matrix is 40x40, with positive values indicating preference asymmetry
# Colors represent strategy types; uses symmetric diverging colormap
ax_A3 = plt.axes(); ax_A3.set_label('A3_DirPref_Experts')
plot_rdm_on_ax(
    ax=ax_A3,
    rdm=expert_dsm,                  # Expert directional preference matrix (40x40)
    colors=strat_colors,             # Strategy-based colors for matrix borders
    alphas=strat_alphas,             # Alpha values for each stimulus
    vmin=behavioral_vmin,            # Symmetric minimum (for diverging colormap)
    vmax=behavioral_vmax,            # Symmetric maximum
    show_colorbar=False,             # Colorbar shown separately in Panel E
)
set_axis_title(ax_A3, title="Directional preference", subtitle="Experts")
ax_A3.set_xticks([]); ax_A3.set_yticks([])  # Hide tick labels (too many stimuli)

# -----------------------------------------------------------------------------
# Panel A4: Directional Preference Matrix (Novices)
# -----------------------------------------------------------------------------
# Same as Panel A3, but for novices
ax_A4 = plt.axes(); ax_A4.set_label('A4_DirPref_Novices')
plot_rdm_on_ax(
    ax=ax_A4,
    rdm=novice_dsm,                  # Novice directional preference matrix (40x40)
    colors=strat_colors,             # Strategy-based colors for matrix borders
    alphas=strat_alphas,             # Alpha values for each stimulus
    vmin=behavioral_vmin,            # Symmetric minimum (for diverging colormap)
    vmax=behavioral_vmax,            # Symmetric maximum
    show_colorbar=False,             # Colorbar shown separately in Panel E
)
set_axis_title(ax_A4, title="Directional preference", subtitle="Novices")
ax_A4.set_xticks([]); ax_A4.set_yticks([])  # Hide tick labels (too many stimuli)

# -----------------------------------------------------------------------------
# Panel A1: Behavioral RDM (Experts)
# -----------------------------------------------------------------------------
# Shows behavioral representational dissimilarity matrix
# Matrix is 40x40, with higher values indicating more dissimilar behavioral responses
ax_A1 = plt.axes(); ax_A1.set_label('A1_RDM_Experts')
plot_rdm_on_ax(
    ax=ax_A1,
    rdm=expert_rdm,                  # Expert behavioral RDM (40x40)
    colors=strat_colors,             # Strategy-based colors for matrix borders
    alphas=strat_alphas,             # Alpha values for each stimulus
    vmin=behavioral_vmin,            # Symmetric minimum (for diverging colormap)
    vmax=behavioral_vmax,            # Symmetric maximum
    show_colorbar=False,             # Colorbar shown separately in Panel E
)
set_axis_title(ax_A1, title="Behavioral RDM", subtitle="Experts")
ax_A1.set_xticks([]); ax_A1.set_yticks([])  # Hide tick labels (too many stimuli)

# -----------------------------------------------------------------------------
# Panel A2: Behavioral RDM (Novices)
# -----------------------------------------------------------------------------
# Same as Panel A1, but for novices
ax_A2 = plt.axes(); ax_A2.set_label('A2_RDM_Novices')
plot_rdm_on_ax(
    ax=ax_A2,
    rdm=novice_rdm,                  # Novice behavioral RDM (40x40)
    colors=strat_colors,             # Strategy-based colors for matrix borders
    alphas=strat_alphas,             # Alpha values for each stimulus
    vmin=behavioral_vmin,            # Symmetric minimum (for diverging colormap)
    vmax=behavioral_vmax,            # Symmetric maximum
    show_colorbar=False,             # Colorbar shown separately in Panel E
)
set_axis_title(ax_A2, title="Behavioral RDM", subtitle="Novices")
ax_A2.set_xticks([]); ax_A2.set_yticks([])  # Hide tick labels (too many stimuli)

# -----------------------------------------------------------------------------
# Panel B1: MDS Embedding (Experts)
# -----------------------------------------------------------------------------
# 2D projection of expert behavioral RDM using multidimensional scaling
# Each point represents one stimulus, positioned based on behavioral similarity
# Stimuli with similar behavioral responses appear closer in 2D space
ax_B1 = plt.axes(); ax_B1.set_label('B1_MDS_Experts')
plot_2d_embedding_on_ax(
    ax=ax_B1,
    coords=expert_mds_coords,        # Expert MDS coordinates (40 stimuli × 2 dimensions)
    point_colors=strat_colors,       # Color each point by strategy type
    point_alphas=strat_alphas,       # Alpha values for each point
    params=PLOT_PARAMS,
    hide_tick_marks=True,            # Hide tick marks (arbitrary MDS units)
    x_label='Dimension 1',           # First MDS dimension
    y_label='Dimension 2',           # Second MDS dimension
)
set_axis_title(ax_B1, title="MDS embedding", subtitle="Experts")

# -----------------------------------------------------------------------------
# Panel B2: MDS Embedding (Novices)
# -----------------------------------------------------------------------------
# Same as Panel B1, but for novices
ax_B2 = plt.axes(); ax_B2.set_label('B2_MDS_Novices')
plot_2d_embedding_on_ax(
    ax=ax_B2,
    coords=novice_mds_coords,        # Novice MDS coordinates (40 stimuli × 2 dimensions)
    point_colors=strat_colors,       # Color each point by strategy type
    point_alphas=strat_alphas,       # Alpha values for each point
    params=PLOT_PARAMS,
    hide_tick_marks=True,            # Hide tick marks (arbitrary MDS units)
    x_label='Dimension 1',           # First MDS dimension
    y_label='Dimension 2',           # Second MDS dimension
)
set_axis_title(ax_B2, title="MDS embedding", subtitle="Novices")

# -----------------------------------------------------------------------------
# Panel C1: Stimulus Selection Frequency (Experts)
# -----------------------------------------------------------------------------
# Histogram showing how often each stimulus was selected in pairwise comparisons
# Higher bars = stimulus was chosen more frequently as "better"
ax_C1 = plt.axes(); ax_C1.set_label('C1_Choice_Experts')

# Compute selection frequency for each stimulus
# If 'count' column exists, sum counts; otherwise, count occurrences
if 'count' in expert_pairwise.columns:
    frequency_exp = expert_pairwise.groupby('better')['count'].sum().sort_index()
else:
    frequency_exp = expert_pairwise['better'].value_counts().sort_index()

# Create legend items (checkmate vs non-checkmate stimuli)
legend_items = [
    ('Checkmate', strat_colors[0], 0.7),                        # First strategy color
    ('Non-checkmate', strat_colors[len(strat_colors)//2], 0.7), # Middle strategy color
]

plot_counts_on_ax(
    ax=ax_C1,
    x_values=frequency_exp.index,              # Stimulus IDs (1-40)
    counts=frequency_exp.values,               # Selection counts
    colors=strat_colors[:len(frequency_exp)],  # Strategy-based colors
    alphas=strat_alphas[:len(frequency_exp)],  # Alpha values
    xlabel='Stimulus ID',
    ylabel='Selection count',
    title='Stimulus selection frequency',
    subtitle='Experts',
    legend=legend_items,                       # Show checkmate vs non-checkmate legend
    params=PLOT_PARAMS,
)

# -----------------------------------------------------------------------------
# Panel C2: Stimulus Selection Frequency (Novices)
# -----------------------------------------------------------------------------
# Same as Panel C1, but for novices
ax_C2 = plt.axes(); ax_C2.set_label('C2_Choice_Novices')

# Compute selection frequency for each stimulus
if 'count' in novice_pairwise.columns:
    frequency_nov = novice_pairwise.groupby('better')['count'].sum().sort_index()
else:
    frequency_nov = novice_pairwise['better'].value_counts().sort_index()

plot_counts_on_ax(
    ax=ax_C2,
    x_values=frequency_nov.index,              # Stimulus IDs (1-40)
    counts=frequency_nov.values,               # Selection counts
    colors=strat_colors[:len(frequency_nov)],  # Strategy-based colors
    alphas=strat_alphas[:len(frequency_nov)],  # Alpha values
    xlabel='Stimulus ID',
    ylabel='Selection count',
    title='Stimulus selection frequency',
    subtitle='Novices',
    legend=legend_items,                       # Show checkmate vs non-checkmate legend
    params=PLOT_PARAMS,
)

# -----------------------------------------------------------------------------
# Panel D: RSA Model Correlations
# -----------------------------------------------------------------------------
# Grouped barplot comparing correlations between behavioral and model RDMs
# Shows how well different computational models predict behavioral similarity
# Bars show Spearman correlation (r) with 95% CIs and FDR-corrected significance
ax_D = plt.axes(); ax_D.set_label('D_RSA_Models')

# Extract correlation results for each model
# Each result is a tuple: (model_name, r, p, ci_low, ci_high)
r_exp = [res[1] for res in expert_corrs]       # Expert correlation values
ci_exp = [(res[3], res[4]) for res in expert_corrs]  # Expert 95% CIs

# Use FDR-corrected p-values if available, otherwise use raw p-values
if isinstance(exp_p_fdr_map, dict):
    p_exp = [exp_p_fdr_map.get(res[0], res[2]) for res in expert_corrs]  # Expert FDR p-values
else:
    p_exp = [res[2] for res in expert_corrs]   # Expert raw p-values

r_nov = [res[1] for res in novice_corrs]       # Novice correlation values
ci_nov = [(res[3], res[4]) for res in novice_corrs]  # Novice 95% CIs

# Use FDR-corrected p-values if available
if isinstance(nov_p_fdr_map, dict):
    p_nov = [nov_p_fdr_map.get(res[0], res[2]) for res in novice_corrs]  # Novice FDR p-values
else:
    p_nov = [res[2] for res in novice_corrs]   # Novice raw p-values

# Reorder results to match MODEL_ORDER (consistent model ordering across figures)
column_labels = [res[0] for res in expert_corrs]  # Model names from results
idx_order = [column_labels.index(lbl) for lbl in MODEL_ORDER]  # Indices for reordering

# Apply ordering to expert results
r_exp_ordered = [r_exp[i] for i in idx_order]
ci_exp_ordered = [ci_exp[i] for i in idx_order]
p_exp_ordered = [p_exp[i] for i in idx_order]

# Apply ordering to novice results
r_nov_ordered = [r_nov[i] for i in idx_order]
ci_nov_ordered = [ci_nov[i] for i in idx_order]
p_nov_ordered = [p_nov[i] for i in idx_order]

# Create x-positions for grouped bars
x = np.arange(len(MODEL_LABELS_PRETTY))

plot_grouped_bars_on_ax(
    ax=ax_D,
    x_positions=x,
    group1_values=r_exp_ordered,                      # Expert correlations
    group1_cis=ci_exp_ordered,                        # Expert 95% CIs
    group1_color=COLORS_EXPERT_NOVICE['expert'],      # Expert color (blue)
    group2_values=r_nov_ordered,                      # Novice correlations
    group2_cis=ci_nov_ordered,                        # Novice 95% CIs
    group2_color=COLORS_EXPERT_NOVICE['novice'],      # Novice color (orange)
    group1_label="Experts",                           # Legend label
    group2_label="Novices",                           # Legend label
    group1_pvals=p_exp_ordered,                       # Expert FDR p-values for significance stars
    group2_pvals=p_nov_ordered,                       # Novice FDR p-values for significance stars
    ylim=(-0.2, 1.0),                                 # Y-axis limits (correlation range)
    y_label=PLOT_PARAMS['ylabel_correlation_r'],      # Y-axis label (Spearman r)
    title="Behavioral-model RSA",
    subtitle="FDR corrected",
    xtick_labels=MODEL_LABELS_PRETTY,                 # Pretty model names
    x_tick_rotation=0,                                # Horizontal labels
    x_tick_align='center',
    show_legend=True,                                 # Show Expert/Novice legend
    legend_loc='upper left',
    visible_spines=['left','bottom'],
    params=PLOT_PARAMS,
)

# -----------------------------------------------------------------------------
# Panel E: Symmetric Colorbar for RDM/DSM Panels
# -----------------------------------------------------------------------------
# Standalone colorbar showing the symmetric color scale used in Panels A1-A4
# Centered at 0 with diverging colormap (blue-white-red)
ax_E = plt.axes(); ax_E.set_label('E_RDM_Colorbar')
cbar_fig = create_standalone_colorbar(
    cmap=CMAP_BRAIN,                 # Diverging colormap (blue-white-red)
    vmin=behavioral_vmin,            # Symmetric minimum
    vmax=behavioral_vmax,            # Symmetric maximum
    orientation='vertical',          # Vertical colorbar
    label='Preference',              # Colorbar label
    params=PLOT_PARAMS,
    tick_position='left'             # Ticks on left side
)
embed_figure_on_ax(ax_E, cbar_fig, title='')  # Embed colorbar figure on axis


# =============================================================================
# Pylustrator Setup and Interactive Layout
# =============================================================================
# Create ax_dict for the figure to enable easy axis reference in pylustrator
# This allows pylustrator to reference axes by label (e.g., fig.ax_dict["A1_RDM_Experts"])
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
plt.figure(1).set_size_inches(16.820000/2.54, 15.610000/2.54, forward=True)
plt.figure(1).ax_dict["A1_RDM_Experts"].set(position=[0.3841, 0.6819, 0.2196, 0.2361])
plt.figure(1).ax_dict["A1_RDM_Experts"].texts[0].set(position=(0.5, 1.064))
plt.figure(1).ax_dict["A1_RDM_Experts"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["A2_RDM_Novices"].set(position=[0.3841, 0.3773, 0.2196, 0.2361])
plt.figure(1).ax_dict["A2_RDM_Novices"].texts[0].set(position=(0.5, 1.067))
plt.figure(1).ax_dict["A2_RDM_Novices"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["A3_DirPref_Experts"].set(position=[0.1125, 0.6819, 0.2196, 0.2361])
plt.figure(1).ax_dict["A3_DirPref_Experts"].texts[0].set(position=(0.5, 1.065))
plt.figure(1).ax_dict["A3_DirPref_Experts"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["A4_DirPref_Novices"].set(position=[0.1125, 0.3773, 0.2196, 0.2361])
plt.figure(1).ax_dict["A4_DirPref_Novices"].texts[0].set(position=(0.5, 1.067))
plt.figure(1).ax_dict["A4_DirPref_Novices"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["B1_MDS_Experts"].set(position=[0.6689, 0.6854, 0.2815, 0.2327])
plt.figure(1).ax_dict["B1_MDS_Experts"].texts[0].set(position=(0.5, 1.069))
plt.figure(1).ax_dict["B1_MDS_Experts"].texts[1].set(position=(0.5, 1.018))
plt.figure(1).ax_dict["B2_MDS_Novices"].set(position=[0.6689, 0.3807, 0.2815, 0.2327])
plt.figure(1).ax_dict["B2_MDS_Novices"].texts[0].set(position=(0.5, 1.071))
plt.figure(1).ax_dict["B2_MDS_Novices"].texts[1].set(position=(0.5, 1.018))
plt.figure(1).ax_dict["C1_Choice_Experts"].set(position=[0.04475, 0.2642, 0.204, 0.1883], xticks=[0., 19., 39.], xticklabels=['1', '20', ' 40'], xlim=(-1., 41.), yticks=[0., 100., 200., 300., 400.], yticklabels=['0', '100', '200', '300', '400'], ylim=(0., 450.))
plt.figure(1).ax_dict["C1_Choice_Experts"].set_position([0.065104, 0.073662, 0.296728, 0.237033])
plt.figure(1).ax_dict["C1_Choice_Experts"].get_legend().set(visible=True)
plt.figure(1).ax_dict["C1_Choice_Experts"].texts[0].set(position=(0.5, 1.046))
plt.figure(1).ax_dict["C1_Choice_Experts"].texts[1].set(position=(0.5, 0.9879))
plt.figure(1).ax_dict["C2_Choice_Novices"].set(position=[0.2739, 0.2642, 0.204, 0.186], xticks=[0., 19., 39.], xticklabels=['1', '20', ' 40'], xlim=(-1., 41.), ylabel='', yticks=[0., 100., 200., 300., 400.], yticklabels=['0', '100', '200', '300', '400'], ylim=(0., 450.))
plt.figure(1).ax_dict["C2_Choice_Novices"].set_position([0.398501, 0.073662, 0.296728, 0.234166])
plt.figure(1).ax_dict["C2_Choice_Novices"].get_legend().set(visible=False)
plt.figure(1).ax_dict["C2_Choice_Novices"].texts[0].set(position=(0.5, 1.059))
plt.figure(1).ax_dict["C2_Choice_Novices"].texts[1].set(position=(0.5, 1.))
plt.figure(1).ax_dict["C2_Choice_Novices"].get_yaxis().get_label().set(text='')
plt.figure(1).ax_dict["D_RSA_Models"].set(position=[0.5179, 0.2653, 0.1387, 0.1871], xticks=[0., 1., 2.], ylim=(-0.25, 0.8))
plt.figure(1).ax_dict["D_RSA_Models"].set_position([0.753374, 0.075104, 0.201763, 0.235536])
plt.figure(1).ax_dict["D_RSA_Models"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["D_RSA_Models"].yaxis.labelpad = -0.262108
plt.figure(1).ax_dict["D_RSA_Models"].texts[3].set(position=(0.5, 1.046))
plt.figure(1).ax_dict["D_RSA_Models"].texts[4].set(position=(0.5, 0.9879))
plt.figure(1).ax_dict["E_RDM_Colorbar"].set(position=[-0.00199, 0.4756, 0.08187, 0.3358])
plt.figure(1).text(0.0959, 0.9421, 'a', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.6620, 0.9421, 'b', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.0340, 0.3347, 'c', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.7288, 0.3347, 'd', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[3].new
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
save_axes_svgs(fig, FIGURES_DIR, 'behavioral')  # e.g., behavioral_A1_RDM_Experts.svg

# Save full panels (complete multi-axis figure)
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'behavioral_rsa_panel.pdf')

logger.info("✓ Panel: behavioral RSA panels complete")

log_script_end(logger)
