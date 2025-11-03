"""
Generate MVPA Finer-Grained Analysis Supplementary Figure (Pylustrator)
=======================================================================

Creates publication-ready multi-panel figure showing fine-grained MVPA analyses
on checkmate boards only. For each fine-grained dimension (strategy, moves to
mate, total pieces, legal moves, tactical motif, side to move), generates three
panels: RSA barplot, SVM decoding barplot, and model RDM visualization.

Uses pylustrator for interactive layout arrangement and reuses standardized
plotting functions from common.plotting and chess-mvpa modules.

Figures Produced
----------------

MVPA Finer Panel (3 columns × 6 rows = 18 panels)
- File: figures/panels/mvpa_finer_panel.svg (and .pdf)
- Individual axes saved to figures/: mvpa_finer_RSA_0_strategy_half.svg, etc.

Panel Layout (repeated for each fine-grained dimension):
Row structure (one row per dimension):
- Left column: RSA correlation bars (Experts vs Novices)
- Middle column: SVM decoding accuracy bars (Experts vs Novices)
- Right column: Model RDM (checkmate stimuli only, 20×20 matrix)

Fine-Grained Dimensions (6 total):
1. Strategy (CM) - attack vs defend binary
2. Moves to Mate (CM) - number of moves to checkmate (ordinal)
3. Total Pieces (CM) - total piece count (ordinal)
4. Legal Moves (CM) - number of legal moves available (ordinal)
5. Tactical Motif (CM) - tactical pattern used (categorical)
6. Side to Move (CM) - which side moves next (binary)

Visual Elements:
- Grouped bars: Experts (solid) vs Novices (hatched)
- Bar colors: ROI group colors (matching main analyses)
- Significance stars: FDR-corrected p-values
- X-axis labels: ROI names, colored by significance (gray if p ≥ 0.05)
- Model RDMs: Colored by board type (checkmate only)

Inputs
------
- mvpa_group_stats.pkl from mvpa_finer_group_rsa results directory
  Contains: Per-ROI RSA correlation statistics (Experts vs Novices)
- mvpa_group_stats.pkl from mvpa_finer_group_decoding results directory
  Contains: Per-ROI SVM decoding statistics (Experts vs Novices)
- stimuli.tsv: Board metadata for computing model RDMs
- ROI metadata: Glasser-22 parcellation info (roi_id, pretty_name, color, etc.)

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives (apply_nature_rc, plot_grouped_bars_on_ax, plot_rdm_on_ax)
- common.rsa_utils (create_model_rdm for generating ground truth RDMs)
- modules.mvpa_plot_utils (extract_mvpa_bar_data for processing stats)
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-supplementary/mvpa-finer/92_plot_mvpa_finer_panel.py
"""

import sys
import os
import pickle
from pathlib import Path

# Add repo root and chess-mvpa to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chess-mvpa')))
script_dir = Path(__file__).parent

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata, load_stimulus_metadata
from common.rsa_utils import create_model_rdm
from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    plot_rdm_on_ax,
    compute_stimulus_palette,
    PLOT_PARAMS,
    PLOT_YLIMITS,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
)
from modules.mvpa_plot_utils import extract_mvpa_bar_data


# =============================================================================
# Configuration and Constants
# =============================================================================
# Define fine-grained dimensions to analyze (all on checkmate boards only)

RESULTS_BASE = script_dir / "results"

# Fine-grained targets to plot (those analyzed on checkmate boards)
# All targets end with "_half" suffix indicating checkmate-only analysis
FINE_TARGETS = [
    'strategy_half',       # Attack vs defend strategy (binary)
    'check_n_half',        # Number of moves to checkmate (ordinal)
    'total_pieces_half',   # Total pieces on board (ordinal)
    'legal_moves_half',    # Number of legal moves available (ordinal)
    'motif_half',          # Tactical motif/pattern (categorical)
]

# Human-readable display names for each dimension (CM = checkmate)
TARGET_DISPLAY_NAMES = {
    'strategy_half': 'Strategy (CM)',
    'check_n_half': 'Moves to Mate (CM)',
    'total_pieces_half': 'Total Pieces (CM)',
    'legal_moves_half': 'Legal Moves (CM)',
    'motif_half': 'Tactical Motif (CM)',
}

# Map target names to stimuli.tsv column names
# This mapping allows us to extract feature values for model RDM computation
TARGET_TO_STIMULI_COL = {
    'strategy_half': 'strategy',
    'check_n_half': 'check_n',
    'total_pieces_half': 'total_pieces',
    'legal_moves_half': 'legal_moves',
    'motif_half': 'motif',
}


# =============================================================================
# Helper Functions for Model RDM Computation
# =============================================================================

def load_checkmate_stimuli_column(stim_tsv: Path, target: str) -> np.ndarray:
    """
    Load feature values for checkmate boards only.

    Extracts a specific feature column from stimuli.tsv, filtered to
    checkmate boards only (check_status == 'checkmate').

    Parameters
    ----------
    stim_tsv : Path
        Path to stimuli.tsv containing board metadata
    target : str
        Target name (e.g., 'strategy_half')

    Returns
    -------
    np.ndarray
        Feature values for checkmate boards only (length = 20)

    Raises
    ------
    ValueError
        If column is not found in stimuli.tsv
    """
    # Load full stimuli table
    df = pd.read_csv(stim_tsv, sep='\t')

    # Filter to checkmate boards only (20 boards)
    checkmate_df = df[df['check_status'] == 'checkmate'].copy()

    # Extract feature column based on target name
    col_name = TARGET_TO_STIMULI_COL[target]
    if col_name not in checkmate_df.columns:
        raise ValueError(f"Column '{col_name}' not found in stimuli.tsv")

    return checkmate_df[col_name].values


def compute_model_rdm_for_target(stim_tsv: Path, target: str) -> np.ndarray:
    """
    Compute model RDM for a target dimension (checkmate boards only).

    Creates a representational dissimilarity matrix (RDM) based on ground truth
    feature values. For categorical features, RDM is binary (0 if same, 1 if
    different). For numerical/ordinal features, RDM is absolute difference.

    Parameters
    ----------
    stim_tsv : Path
        Path to stimuli.tsv containing board metadata
    target : str
        Target name (e.g., 'strategy_half')

    Returns
    -------
    np.ndarray
        Model RDM matrix (20×20 for checkmate boards)
        Symmetric matrix where rdm[i,j] = dissimilarity between boards i and j
    """
    # Load feature values for checkmate boards
    feature_values = load_checkmate_stimuli_column(stim_tsv, target)

    # Determine whether feature is categorical or numerical/ordinal
    # Categorical: strategy, motif, side (binary RDM: 0 if same, 1 if different)
    # Numerical/ordinal: total_pieces, legal_moves, check_n (continuous RDM: absolute difference)
    categorical_targets = ['strategy_half', 'motif_half', 'side_half']
    is_categorical = target in categorical_targets

    # Compute RDM using common utility function
    # For categorical: RDM[i,j] = 1 if feature[i] != feature[j], else 0
    # For numerical: RDM[i,j] = |feature[i] - feature[j]|
    rdm = create_model_rdm(feature_values, is_categorical=is_categorical)
    return rdm


# =============================================================================
# Configuration and Results Loading
# =============================================================================
# Find latest MVPA finer results directories and load group statistics

apply_nature_rc()  # Apply Nature journal style to all figures

# Find latest RSA and SVM decoding results directories (separate analyses)
RESULTS_DIR_RSA = find_latest_results_directory(
    RESULTS_BASE,
    pattern='mvpa_finer_group_rsa',
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True
)
RESULTS_DIR_SVM = find_latest_results_directory(
    RESULTS_BASE,
    pattern='mvpa_finer_group_decoding',
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True
)

FIGURES_DIR = RESULTS_DIR_RSA / "figures"

# Initialize logging in RSA results directory (primary output location)
extra = {
    "RESULTS_DIR_RSA": str(RESULTS_DIR_RSA),
    "RESULTS_DIR_SVM": str(RESULTS_DIR_SVM),
}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR_RSA,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="pylustrator_mvpa_finer.log",
)

logger.info("Loading MVPA finer group statistics...")

# Load RSA correlation statistics (per-ROI, per-target, per-group)
# Contains: Expert vs Novice RSA correlations, p-values, CIs
with open(RESULTS_DIR_RSA / "mvpa_group_stats.pkl", "rb") as f:
    rsa_stats = pickle.load(f)

# Load SVM decoding statistics (per-ROI, per-target, per-group)
# Contains: Expert vs Novice SVM accuracies, p-values, CIs
with open(RESULTS_DIR_SVM / "mvpa_group_stats.pkl", "rb") as f:
    svm_stats = pickle.load(f)

# Load ROI metadata for Glasser-22 parcellation
roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])

# Load checkmate stimuli metadata and compute color palette for RDM visualization
# Palette assigns colors based on board properties (e.g., strategy, piece count)
all_stimuli = load_stimulus_metadata(return_all=True)
checkmate_stimuli = all_stimuli[all_stimuli['check'] == 'checkmate'].copy()
stim_colors, stim_alphas = compute_stimulus_palette(checkmate_stimuli)


# =============================================================================
# Figure: MVPA Finer-Grained Panels (Independent Axes for Pylustrator)
# =============================================================================
# Creates 3 panels per fine-grained dimension (RSA, SVM, model RDM)
# Total: 6 dimensions × 3 panels = 18 panels arranged by pylustrator

fig = plt.figure(figsize=(12, len(FINE_TARGETS) * 3))

panel_idx = 0  # Track row index for layout
for target in FINE_TARGETS:
    # -----------------------------------------------------------------------------
    # Data Extraction and Preparation
    # -----------------------------------------------------------------------------
    # Extract barplot data using existing MVPA helper function
    # This function processes group stats and formats data for plotting

    # RSA correlation data (correlation coefficient with model RDM)
    rsa_data = extract_mvpa_bar_data(
        rsa_stats,
        roi_info,
        [target],
        method='rsa_corr',      # RSA correlation method
        subtract_chance=False   # Don't subtract chance (correlations start at 0)
    )

    # SVM decoding data (classification accuracy)
    svm_data = extract_mvpa_bar_data(
        svm_stats,
        roi_info,
        [target],
        method='svm',           # SVM decoding method
        subtract_chance=True    # Subtract chance level (50% for binary)
    )

    # Skip target if data is missing
    if target not in rsa_data or target not in svm_data:
        logger.warning(f"Skipping {target}: missing RSA or SVM data")
        continue

    # Compute model RDM for this target (ground truth dissimilarity matrix)
    model_rdm = compute_model_rdm_for_target(Path(CONFIG['STIMULI_FILE']), target)

    # Extract formatted data structures for plotting
    rsa = rsa_data[target]  # Dict with keys: roi_names, roi_colors, exp_means, nov_means, etc.
    svm = svm_data[target]  # Dict with same structure

    # Get ROI names, colors, and label colors (gray if not significant)
    roi_names = rsa['roi_names']       # List of pretty ROI names
    roi_colors = rsa['roi_colors']     # List of ROI group colors
    label_colors_rsa = rsa['label_colors'] # List of label colors (gray if p ≥ 0.05)
    label_colors_svm = svm['label_colors'] # List of label colors (gray if p ≥ 0.05)
    x = np.arange(len(roi_names))      # X-positions for bars

    # Compute y-axis limits with padding for clean visualization
    ylim_rsa = PLOT_YLIMITS['rsa_neural_rois_finer']  # Centralized finer ROI RSA limits (was -.02, .18)
    ylim_svm = PLOT_YLIMITS['decoding_rois_finer']    # Centralized finer ROI decoding limits (was -.02, .18)

    # -----------------------------------------------------------------------------
    # Panel A: RSA Correlation Barplot (Experts vs Novices)
    # -----------------------------------------------------------------------------
    # Shows per-ROI correlation between neural RDMs and model RDM
    # Grouped bars: Experts (solid) vs Novices (hatched)
    ax_rsa = plt.axes()
    ax_rsa.set_label(f'RSA_{panel_idx}_{target}')

    plot_grouped_bars_on_ax(
        ax=ax_rsa,
        x_positions=x,
        group1_values=rsa['exp_means'],      # Expert mean correlations
        group1_cis=rsa['exp_cis'],           # Expert 95% CIs
        group1_color=roi_colors,             # ROI group colors (solid bars)
        group2_values=rsa['nov_means'],      # Novice mean correlations
        group2_cis=rsa['nov_cis'],           # Novice 95% CIs
        group2_color=roi_colors,             # Same colors (hatched bars)
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=rsa['pvals'],       # FDR-corrected p-values for stars
        ylim=ylim_rsa,                       # Shared y-axis range
        y_label=PLOT_PARAMS['ylabel_correlation_r'],  # 'Correlation (r)'
        subtitle=f'{TARGET_DISPLAY_NAMES[target]}',
        xtick_labels=roi_names,              # ROI names on x-axis
        x_label_colors=label_colors_rsa,         # Color by significance
        x_tick_rotation=30,
        x_tick_align='right',
        visible_spines=['left','bottom'],
        params=PLOT_PARAMS
    )

    # -----------------------------------------------------------------------------
    # Panel B: SVM Decoding Barplot (Experts vs Novices)
    # -----------------------------------------------------------------------------
    # Shows per-ROI classification accuracy (minus chance level)
    # Grouped bars: Experts (solid) vs Novices (hatched)
    ax_svm = plt.axes()
    ax_svm.set_label(f'SVM_{panel_idx}_{target}')

    plot_grouped_bars_on_ax(
        ax=ax_svm,
        x_positions=x,
        group1_values=svm['exp_means'],      # Expert mean accuracies (minus chance)
        group1_cis=svm['exp_cis'],           # Expert 95% CIs
        group1_color=roi_colors,             # ROI group colors (solid bars)
        group2_values=svm['nov_means'],      # Novice mean accuracies (minus chance)
        group2_cis=svm['nov_cis'],           # Novice 95% CIs
        group2_color=roi_colors,             # Same colors (hatched bars)
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=svm['pvals'],       # FDR-corrected p-values for stars
        ylim=ylim_svm,                       # Shared y-axis range
        y_label='Accuracy - chance',         # Chance-subtracted accuracy
        subtitle=f'{TARGET_DISPLAY_NAMES[target]}',
        xtick_labels=roi_names,              # ROI names on x-axis
        x_label_colors=label_colors_svm,         # Color by significance
        x_tick_rotation=30,
        x_tick_align='right',
        visible_spines=['left','bottom'],
        params=PLOT_PARAMS
    )

    # -----------------------------------------------------------------------------
    # Panel C: Model RDM (Checkmate Stimuli Only)
    # -----------------------------------------------------------------------------
    # Shows ground truth dissimilarity matrix for this dimension
    # Matrix is 20×20 (checkmate boards only)
    # Cells colored by stimulus properties using precomputed palette
    ax_rdm = plt.axes()
    ax_rdm.set_label(f'RDM_{panel_idx}_{target}')

    plot_rdm_on_ax(
        ax=ax_rdm,
        rdm=model_rdm,                       # Model RDM (20×20 matrix)
        title='',
        colors=stim_colors,                  # Stimulus-based color palette
        alphas=stim_alphas,                  # Transparency for each stimulus
        show_colorbar=False,
        params=PLOT_PARAMS
    )

    panel_idx += 1  # Increment row index

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
#% end: automatic generated code from pylustrator

# Display figure in pylustrator GUI for interactive layout adjustment
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(cm_to_inches(18.26), cm_to_inches(23.93), forward=True)
plt.figure(1).ax_dict["RDM_0_strategy_half"].set(position=[0.4555, 0.6875, 0.08276, 0.06322])
plt.figure(1).ax_dict["RDM_1_check_n_half"].set(position=[0.4555, 0.4895, 0.08276, 0.06322])
plt.figure(1).ax_dict["RDM_2_total_pieces_half"].set(position=[0.4555, 0.09723, 0.08276, 0.06322])
plt.figure(1).ax_dict["RDM_3_legal_moves_half"].set(position=[0.4555, 0.2934, 0.08276, 0.06322])
plt.figure(1).ax_dict["RDM_4_motif_half"].set(position=[0.4555, 0.8854, 0.08276, 0.06322])
plt.figure(1).ax_dict["RSA_0_strategy_half"].set(position=[0.6065, 0.6762, 0.3821, 0.08568])
plt.figure(1).ax_dict["RSA_1_check_n_half"].set(position=[0.6065, 0.4782, 0.3821, 0.08568])
plt.figure(1).ax_dict["RSA_2_total_pieces_half"].set(position=[0.6065, 0.08599, 0.3821, 0.08568])
plt.figure(1).ax_dict["RSA_3_legal_moves_half"].set(position=[0.6065, 0.2821, 0.3821, 0.08568])
plt.figure(1).ax_dict["RSA_4_motif_half"].set(position=[0.6065, 0.8742, 0.3821, 0.08568])
plt.figure(1).ax_dict["RSA_4_motif_half"].text(0.5148, 1.2943, 'Brain-Model RSA (Checkmate boards only)', transform=plt.figure(1).ax_dict["RSA_4_motif_half"].transAxes, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["RSA_4_motif_half"].texts[1].new
plt.figure(1).ax_dict["SVM_0_strategy_half"].set(position=[0.05909, 0.6744, 0.378, 0.08941])
plt.figure(1).ax_dict["SVM_1_check_n_half"].set(position=[0.05909, 0.4782, 0.378, 0.08568])
plt.figure(1).ax_dict["SVM_2_total_pieces_half"].set(position=[0.05909, 0.08599, 0.378, 0.08568])
plt.figure(1).ax_dict["SVM_3_legal_moves_half"].set(position=[0.05909, 0.2821, 0.378, 0.08568])
plt.figure(1).ax_dict["SVM_4_motif_half"].set(position=[0.05909, 0.8742, 0.378, 0.08568])
plt.figure(1).ax_dict["SVM_4_motif_half"].text(0.2045, 1.2943, 'Brain Decoding (Checkmate board only)', transform=plt.figure(1).ax_dict["SVM_4_motif_half"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["SVM_4_motif_half"].texts[6].new
#% end: automatic generated code from pylustrator
from common import CONFIG as _CONFIG_FOR_SHOW
if _CONFIG_FOR_SHOW['ENABLE_PYLUSTRATOR']:
    plt.show()


# =============================================================================
# Save Figures (Individual Axes and Full Panel)
# =============================================================================
# After arranging axes in pylustrator, save both:
# 1. Individual axes as separate SVG/PDF files (for modular figure assembly)
# 2. Full panel as complete SVG/PDF file (for standalone use)

# Save full panel (complete multi-axis figure)
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'mvpa_finer_panel.pdf')

# Save individual axes (one file per axis, named by prefix + axis label)
save_axes_svgs(fig, FIGURES_DIR, 'mvpa_finer')

logger.info("✓ Panel: MVPA fine-grained dimensions complete")

log_script_end(logger)
