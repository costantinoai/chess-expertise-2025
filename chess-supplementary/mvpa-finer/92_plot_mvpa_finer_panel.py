"""
Pylustrator-driven MVPA-finer panels.

For each fine-grained dimension tested on checkmate boards:
  - Left: RSA barplot (Expert vs Novice)
  - Middle: SVM decoding barplot (Expert vs Novice)
  - Right: Model RDM (checkmate stimuli only)

Reuses plotting functions from common.plotting and chess-mvpa modules.

Usage:
    python 92_plot_mvpa_finer_panel.py
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
from scipy.spatial.distance import pdist, squareform
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata, load_stimulus_metadata
from common.rsa_utils import create_model_rdm
from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    plot_rdm_on_ax,
    set_axis_title,
    compute_ylim_range,
    compute_stimulus_palette,
    style_spines,
    PLOT_PARAMS,
    save_axes_svgs,
    save_panel_svg,
    save_axes_pdfs,
    save_panel_pdf,
)
from modules.mvpa_plot_utils import extract_mvpa_bar_data


# =============================================================================
# Configuration
# =============================================================================

RESULTS_BASE = script_dir / "results"

# Fine-grained targets to plot (those analyzed on checkmate boards)
FINE_TARGETS = [
    'strategy_half',
    'check_n_half',
    'total_pieces_half',
    'legal_moves_half',
    'motif_half',
    'side_half',
]

TARGET_DISPLAY_NAMES = {
    'strategy_half': 'Strategy (CM)',
    'check_n_half': 'Moves to Mate (CM)',
    'total_pieces_half': 'Total Pieces (CM)',
    'legal_moves_half': 'Legal Moves (CM)',
    'motif_half': 'Tactical Motif (CM)',
    'side_half': 'Side to Move (CM)',
}

# Map target names to stimuli.tsv column names
TARGET_TO_STIMULI_COL = {
    'strategy_half': 'strategy',
    'check_n_half': 'check_n',
    'total_pieces_half': 'total_pieces',
    'legal_moves_half': 'legal_moves',
    'motif_half': 'motif',
    'side_half': 'side',
}


# =============================================================================
# Helper functions
# =============================================================================

def load_checkmate_stimuli_column(stim_tsv: Path, target: str) -> np.ndarray:
    """
    Load feature values for checkmate boards only.

    Parameters
    ----------
    stim_tsv : Path
        Path to stimuli.tsv
    target : str
        Target name (e.g., 'strategy_half')

    Returns
    -------
    np.ndarray
        Feature values for checkmate boards only
    """
    df = pd.read_csv(stim_tsv, sep='\t')
    checkmate_df = df[df['check_status'] == 'checkmate'].copy()

    col_name = TARGET_TO_STIMULI_COL[target]
    if col_name not in checkmate_df.columns:
        raise ValueError(f"Column '{col_name}' not found in stimuli.tsv")

    return checkmate_df[col_name].values


def compute_model_rdm_for_target(stim_tsv: Path, target: str) -> np.ndarray:
    """
    Compute model RDM for a target dimension (checkmate boards only).

    Parameters
    ----------
    stim_tsv : Path
        Path to stimuli.tsv
    target : str
        Target name (e.g., 'strategy_half')

    Returns
    -------
    np.ndarray
        Model RDM matrix (20×20 for checkmate boards)
    """
    feature_values = load_checkmate_stimuli_column(stim_tsv, target)

    # Determine whether feature is categorical or numerical
    # Categorical: strategy, motif, side (binary RDM: 0 if same, 1 if different)
    # Numerical/ordinal: total_pieces, legal_moves, check_n (continuous RDM: absolute difference)
    categorical_targets = ['strategy_half', 'motif_half', 'side_half']
    is_categorical = target in categorical_targets

    # Compute RDM using common.rsa_utils.create_model_rdm
    rdm = create_model_rdm(feature_values, is_categorical=is_categorical)
    return rdm


# =============================================================================
# Load results
# =============================================================================

apply_nature_rc()

RESULTS_DIR_RSA = find_latest_results_directory(
    RESULTS_BASE, pattern='*_mvpa_finer_group_rsa', create_subdirs=['figures'], require_exists=True, verbose=True
)
RESULTS_DIR_SVM = find_latest_results_directory(
    RESULTS_BASE, pattern='*_mvpa_finer_group_decoding', create_subdirs=['figures'], require_exists=True, verbose=True
)

FIGURES_DIR = RESULTS_DIR_RSA / "figures"

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
with open(RESULTS_DIR_RSA / "mvpa_group_stats.pkl", "rb") as f:
    rsa_stats = pickle.load(f)
with open(RESULTS_DIR_SVM / "mvpa_group_stats.pkl", "rb") as f:
    svm_stats = pickle.load(f)

roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])

# Load checkmate stimuli and compute color palette for RDM visualization
all_stimuli = load_stimulus_metadata(return_all=True)
checkmate_stimuli = all_stimuli[all_stimuli['check'] == 'checkmate'].copy()
stim_colors, stim_alphas = compute_stimulus_palette(checkmate_stimuli)


# =============================================================================
# Create panels for each fine target
# =============================================================================

fig = plt.figure(figsize=(12, len(FINE_TARGETS) * 3))

panel_idx = 0
for target in FINE_TARGETS:
    # Extract barplot data using existing function
    rsa_data = extract_mvpa_bar_data(rsa_stats, roi_info, [target], method='rsa_corr', subtract_chance=False)
    svm_data = extract_mvpa_bar_data(svm_stats, roi_info, [target], method='svm', subtract_chance=True)

    if target not in rsa_data or target not in svm_data:
        logger.warning(f"Skipping {target}: missing RSA or SVM data")
        continue

    # Compute model RDM
    model_rdm = compute_model_rdm_for_target(Path(CONFIG['STIMULI_FILE']), target)

    rsa = rsa_data[target]
    svm = svm_data[target]

    roi_names = rsa['roi_names']
    roi_colors = rsa['roi_colors']
    label_colors = rsa['label_colors']
    x = np.arange(len(roi_names))

    # Compute ylims
    ylim_rsa = compute_ylim_range(
        list(rsa['exp_means']) + list(rsa['nov_means']), padding_pct=0.15
    )
    ylim_svm = compute_ylim_range(
        list(svm['exp_means']) + list(svm['nov_means']), padding_pct=0.15
    )

    # Panel A: RSA barplot
    ax_rsa = plt.axes()
    ax_rsa.set_label(f'RSA_{panel_idx}_{target}')

    plot_grouped_bars_on_ax(
        ax=ax_rsa,
        x_positions=x,
        group1_values=rsa['exp_means'],
        group1_cis=rsa['exp_cis'],
        group1_color=roi_colors,
        group2_values=rsa['nov_means'],
        group2_cis=rsa['nov_cis'],
        group2_color=roi_colors,
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=rsa['pvals'],
        ylim=ylim_rsa,
        params=PLOT_PARAMS
    )

    ax_rsa.set_ylabel(PLOT_PARAMS['ylabel_correlation_r'], fontsize=PLOT_PARAMS['font_size_label'])
    ax_rsa.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
    set_axis_title(ax_rsa, title=f'{TARGET_DISPLAY_NAMES[target]} RSA')

    if panel_idx == 0:
        ax_rsa.legend(frameon=False, loc='upper right', ncol=1, fontsize=PLOT_PARAMS['font_size_legend'])

    style_spines(ax_rsa, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    ax_rsa.set_xlim(-0.5, len(roi_names) - 0.5)
    ax_rsa.set_xticks(x)
    ax_rsa.set_xticklabels(roi_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    for ticklabel, color in zip(ax_rsa.get_xticklabels(), label_colors):
        ticklabel.set_color(color)

    # Panel B: SVM decoding barplot
    ax_svm = plt.axes()
    ax_svm.set_label(f'SVM_{panel_idx}_{target}')

    plot_grouped_bars_on_ax(
        ax=ax_svm,
        x_positions=x,
        group1_values=svm['exp_means'],
        group1_cis=svm['exp_cis'],
        group1_color=roi_colors,
        group2_values=svm['nov_means'],
        group2_cis=svm['nov_cis'],
        group2_color=roi_colors,
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=svm['pvals'],
        ylim=ylim_svm,
        params=PLOT_PARAMS
    )

    ax_svm.set_ylabel('Accuracy - chance', fontsize=PLOT_PARAMS['font_size_label'])
    ax_svm.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
    set_axis_title(ax_svm, title=f'{TARGET_DISPLAY_NAMES[target]} Decoding')

    style_spines(ax_svm, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    ax_svm.set_xlim(-0.5, len(roi_names) - 0.5)
    ax_svm.set_xticks(x)
    ax_svm.set_xticklabels(roi_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    for ticklabel, color in zip(ax_svm.get_xticklabels(), label_colors):
        ticklabel.set_color(color)

    # Panel C: Model RDM
    ax_rdm = plt.axes()
    ax_rdm.set_label(f'RDM_{panel_idx}_{target}')

    plot_rdm_on_ax(
        ax=ax_rdm,
        rdm=model_rdm,
        title=f'{TARGET_DISPLAY_NAMES[target]} Model RDM',
        colors=stim_colors,
        alphas=stim_alphas,
        show_colorbar=(panel_idx == len(FINE_TARGETS) - 1),  # Show colorbar on last panel
        params=PLOT_PARAMS
    )

    panel_idx += 1

fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}

#% start: automatic generated code from pylustrator
#% end: automatic generated code from pylustrator
plt.show()

save_axes_svgs(fig, FIGURES_DIR, 'mvpa_finer')
save_axes_pdfs(fig, FIGURES_DIR, 'mvpa_finer')
save_panel_svg(fig, FIGURES_DIR / 'panels' / 'mvpa_finer_panel.svg')
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'mvpa_finer_panel.pdf')

log_script_end(logger)
