"""
Generate Subcortical RSA Figure Panel (Pylustrator)
====================================================

Creates publication-ready bar plots for subcortical ROI-based RSA analysis.
Mirrors chess-mvpa/92_plot_mvpa_rsa.py exactly in styling, using centralized
PLOT_YLIMITS and PLOT_PARAMS. No pial surface maps (subcortical structures
are not surface-rendered).

Figures Produced
----------------
- subcortical_rsa__RSA_1_visual_similarity.svg
- subcortical_rsa__RSA_2_strategy.svg
- subcortical_rsa__RSA_3_checkmate.svg
- panels/subcortical_rsa_panel.pdf

Inputs
------
- subcortical_group_stats.pkl (rsa_corr block)
- ROI metadata from CONFIG['ROI_CABNP']
"""

import pickle
from pathlib import Path
script_dir = Path(__file__).parent


from common import CONFIG

if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import numpy as np
import matplotlib.pyplot as plt
from common import setup_script, log_script_end
from common.bids_utils import load_roi_metadata
from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    PLOT_PARAMS,
    PLOT_YLIMITS,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
    embed_figure_on_ax,
    create_roi_group_legend,
)
from analyses.mvpa.plot_utils import extract_mvpa_bar_data


# =============================================================================
# Configuration — matches cortical 92_plot_mvpa_rsa.py exactly
# =============================================================================

MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']

RSA_TITLES = {
    'visual_similarity': 'Visual Similarity RSA',
    'strategy': 'Strategy RSA',
    'checkmate': 'Checkmate RSA',
}


# =============================================================================
# Load results
# =============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='subcortical_rois',
    output_subdirs=['figures'],
    log_name='pylustrator_subcortical_rsa.log',
)
RESULTS_DIR = results_dir
FIGURES_DIR = dirs['figures']

logger.info("Loading subcortical RSA group statistics...")
with open(RESULTS_DIR / "subcortical_group_stats.pkl", "rb") as f:
    group_stats = pickle.load(f)

roi_info = load_roi_metadata(CONFIG["ROI_CABNP"])

apply_nature_rc()


# =============================================================================
# RSA Correlation Barplots — same styling as cortical 92_plot_mvpa_rsa.py
# =============================================================================

rsa_data = extract_mvpa_bar_data(
    group_stats, roi_info, MAIN_TARGETS,
    method='rsa_corr', subtract_chance=False,
)

fig1 = plt.figure(1)

for idx, tgt in enumerate(MAIN_TARGETS):
    if tgt not in rsa_data:
        continue

    data = rsa_data[tgt]
    x = np.arange(len(data['roi_names']))

    ax = plt.axes()
    ax.set_label(f'RSA_{idx+1}_{tgt}')

    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x,
        group1_values=data['exp_means'],
        group1_cis=data['exp_cis'],
        group1_color=data['roi_colors'],
        group2_values=data['nov_means'],
        group2_cis=data['nov_cis'],
        group2_color=data['roi_colors'],
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=data['pvals'],
        ylim=PLOT_YLIMITS['rsa_subcortical'],   # Tighter range for subcortical
        y_label=PLOT_PARAMS['ylabel_correlation_r'],
        subtitle=RSA_TITLES[tgt],
        xtick_labels=data['roi_names'],
        x_label_colors=data['label_colors'],
        x_tick_rotation=30,
        x_tick_align='right',
        show_legend=(idx == 0),
        legend_loc='upper right',
        visible_spines=['left', 'bottom'],
        params=PLOT_PARAMS,
    )

# Add ROI group legend (horizontal, bottom)
ax_legend = plt.axes()
ax_legend.set_label('ROI_Legend')
legend_fig = create_roi_group_legend(
    roi_metadata_path=CONFIG['ROI_CABNP'] / 'region_info.tsv',
    output_path=None,
    single_row=True,
    colorblind=False,
)
embed_figure_on_ax(ax_legend, legend_fig, title='')

fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(cm_to_inches(11.43), cm_to_inches(16.00), forward=True)
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set_position([0.119538, 0.76, 0.665730, 0.148551])
plt.figure(1).ax_dict["RSA_2_strategy"].set_position([0.119538, 0.46, 0.665730, 0.148551])
plt.figure(1).ax_dict["RSA_3_checkmate"].set_position([0.119538, 0.16, 0.665730, 0.148551])
plt.figure(1).ax_dict["ROI_Legend"].set_position([-0.03, -0.01, 1.06, 0.12])
plt.figure(1).text(0.365780, 0.982855, 'Subcortical Brain-Model RSA', transform=plt.figure(1).transFigure, fontsize=7., weight='bold')
#% end: automatic generated code from pylustrator

if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()

save_axes_svgs(fig1, FIGURES_DIR, 'subcortical_rsa')
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'subcortical_rsa_panel.pdf')

logger.info("Panel: Subcortical RSA complete")
log_script_end(logger)
