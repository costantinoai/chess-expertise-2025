"""
Pylustrator-driven MVPA panels — Decoding only.

Figure: ROI SVM decoding (Expert vs Novice) for three targets
  - Visual Similarity, Strategy, Checkmate

Arrange interactively in pylustrator and save to inject layout code.

Usage:
    python 93_plot_mvpa_decoding.py
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
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    set_axis_title,
    compute_ylim_range,
    format_roi_labels_and_colors,
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

RESULTS_DIR_NAME = None
RESULTS_BASE = script_dir / "results"

MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']
SVM_TITLES = {
    'visual_similarity': 'Visual Similarity Decoding',
    'strategy': 'Strategy Decoding',
    'checkmate': 'Checkmate Decoding',
}


# =============================================================================
# Load results
# =============================================================================

RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern="*_mvpa_group_decoding",
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=["figures"],
    require_exists=True,
    verbose=True,
)

FIGURES_DIR = RESULTS_DIR / "figures"

extra = {"RESULTS_DIR": str(RESULTS_DIR), "FIGURES_DIR": str(FIGURES_DIR)}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="pylustrator_mvpa_decoding.log",
)

logger.info("Loading MVPA decoding group statistics...")
with open(RESULTS_DIR / "mvpa_group_stats.pkl", "rb") as f:
    group_stats = pickle.load(f)

roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])
apply_nature_rc()

# Figure: SVM decoding (3 axes, no layout)
svm_data = extract_mvpa_bar_data(group_stats, roi_info, MAIN_TARGETS, method='svm', subtract_chance=True)

fig1 = plt.figure(1)

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

    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x,
        group1_values=data['exp_means'],
        group1_cis=data['exp_cis'],
        group1_color=roi_colors,
        group2_values=data['nov_means'],
        group2_cis=data['nov_cis'],
        group2_color=roi_colors,
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=data['pvals'],
        ylim=(0, 0.25),
        params=PLOT_PARAMS
    )

    ax.set_ylabel('Accuracy - chance', fontsize=PLOT_PARAMS['font_size_label'])
    ax.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
    set_axis_title(ax, title=SVM_TITLES[tgt])

    if idx == 0:
        ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=PLOT_PARAMS['font_size_legend'])

    style_spines(ax, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    ax.set_xlim(-0.5, len(roi_names) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    for ticklabel, color in zip(ax.get_xticklabels(), label_colors):
        ticklabel.set_color(color)

fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["SVM_1_visual_similarity"].set(position=[0.044, 0.7789, 0.4626, 0.1855])
plt.figure(1).ax_dict["SVM_2_strategy"].set(position=[0.044, 0.4513, 0.4626, 0.1855])
plt.figure(1).ax_dict["SVM_3_checkmate"].set(position=[0.044, 0.1236, 0.4626, 0.1855])
#% end: automatic generated code from pylustrator
plt.show()

save_axes_svgs(fig1, FIGURES_DIR, 'mvpa_svm')
save_axes_pdfs(fig1, FIGURES_DIR, 'mvpa_svm')
save_panel_svg(fig1, FIGURES_DIR / 'panels' / 'mvpa_svm_panel.svg')
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'mvpa_svm_panel.pdf')

log_script_end(logger)
