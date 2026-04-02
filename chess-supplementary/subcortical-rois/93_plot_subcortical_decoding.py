"""
Generate Subcortical Decoding + RSA Figure Panel (Pylustrator)
==============================================================

Mirrors chess-mvpa/93_plot_mvpa_decoding.py for subcortical ROIs. Creates a
combined panel with SVM decoding (left column) and RSA correlations (right
column) for the three target dimensions. Uses centralized PLOT_YLIMITS and
PLOT_PARAMS for consistency with cortical panels.

NOTE: Pickle is used for reading the group stats artifact, matching the
existing cortical pipeline format (mvpa_group_stats.pkl).

Figures Produced
----------------
- subcortical_svm__SVM_1/2/3_*.svg: SVM decoding bars
- subcortical_svm__RSA_1/2/3_*.svg: RSA correlation bars
- panels/subcortical_svm_panel.pdf: Combined 2x3 panel
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
    plot_grouped_boxplots_on_ax,
    compute_whisker_ylim,
    PLOT_PARAMS,
    PLOT_YLIMITS,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
    embed_figure_on_ax,
    create_roi_group_legend,
)
from analyses.mvpa.plot_utils import extract_mvpa_bar_data


MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']

TARGET_TITLES = {
    'visual_similarity': 'Visual Similarity',
    'strategy': 'Strategy',
    'checkmate': 'Checkmate',
}

# =============================================================================
# Load results
# =============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='subcortical_rois',
    output_subdirs=['figures'],
    log_name='pylustrator_subcortical_decoding.log',
)
RESULTS_DIR = results_dir
FIGURES_DIR = dirs['figures']

logger.info("Loading subcortical SVM + RSA group statistics...")
# Read unified artifact pickle (same format as cortical mvpa_group_stats.pkl)
with open(RESULTS_DIR / "subcortical_group_stats.pkl", "rb") as f:
    group_stats = pickle.load(f)

roi_info = load_roi_metadata(CONFIG["ROI_CABNP"])
apply_nature_rc()

svm_data = extract_mvpa_bar_data(group_stats, roi_info, MAIN_TARGETS, method='svm', subtract_chance=True)
rsa_data = extract_mvpa_bar_data(group_stats, roi_info, MAIN_TARGETS, method='rsa_corr', subtract_chance=False)

# Load per-subject subcortical data for boxplots
import pandas as pd
from common.io_utils import find_subject_tsvs
from common.bids_utils import get_participants_with_expertise
from analyses.mvpa.io import build_group_dataframe

roi_col_names = roi_info['roi_name'].tolist()
participants, _ = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)

rsa_files = find_subject_tsvs(Path(CONFIG['BIDS_MVPA_RSA_SUBCORTICAL']))
rsa_df = build_group_dataframe(rsa_files, participants, roi_col_names)
rsa_subject = {t: {'experts': [], 'novices': []} for t in MAIN_TARGETS}
for t in MAIN_TARGETS:
    for _, row in rsa_df[rsa_df['target'] == t].iterrows():
        g = 'experts' if row['expert'] else 'novices'
        rsa_subject[t][g].append(row[roi_col_names].values.astype(float))
for t in MAIN_TARGETS:
    for g in ('experts', 'novices'):
        rsa_subject[t][g] = np.array(rsa_subject[t][g])

svm_files = find_subject_tsvs(Path(CONFIG['BIDS_MVPA_DECODING_SUBCORTICAL']))
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

svm_boxplot_ylim = compute_whisker_ylim(
    *[svm_subject[t][g] for t in MAIN_TARGETS for g in ('experts', 'novices')]
)
rsa_boxplot_ylim = compute_whisker_ylim(
    *[rsa_subject[t][g] for t in MAIN_TARGETS for g in ('experts', 'novices')]
)

fig1 = plt.figure(1)

# --- SVM panels ---
for idx, tgt in enumerate(MAIN_TARGETS):
    if tgt not in svm_data:
        continue
    data = svm_data[tgt]
    x = np.arange(len(data['roi_names']))
    ax = plt.axes(); ax.set_label(f'SVM_{idx+1}_{tgt}')

    plot_grouped_boxplots_on_ax(
        ax=ax, x_positions=x,
        group1_data=svm_subject[tgt]['experts'], group2_data=svm_subject[tgt]['novices'],
        group1_color=data['roi_colors'], group2_color=data['roi_colors'],
        group1_label='Experts', group2_label='Novices',
        comparison_pvals=data['pvals'], ylim=svm_boxplot_ylim,
        y_label='Accuracy - chance', subtitle=TARGET_TITLES[tgt],
        xtick_labels=data['roi_names'], x_label_colors=data['label_colors'],
        x_tick_rotation=30, x_tick_align='right',
        show_legend=(idx == 0), legend_loc='upper right',
        visible_spines=['left', 'bottom'], params=PLOT_PARAMS,
    )

# --- RSA panels ---
for idx, tgt in enumerate(MAIN_TARGETS):
    if tgt not in rsa_data:
        continue
    data = rsa_data[tgt]
    x = np.arange(len(data['roi_names']))
    ax = plt.axes(); ax.set_label(f'RSA_{idx+1}_{tgt}')

    plot_grouped_boxplots_on_ax(
        ax=ax, x_positions=x,
        group1_data=rsa_subject[tgt]['experts'], group2_data=rsa_subject[tgt]['novices'],
        group1_color=data['roi_colors'], group2_color=data['roi_colors'],
        group1_label='Experts', group2_label='Novices',
        comparison_pvals=data['pvals'], ylim=rsa_boxplot_ylim,
        y_label=PLOT_PARAMS['ylabel_correlation_r'], subtitle=TARGET_TITLES[tgt],
        xtick_labels=data['roi_names'], x_label_colors=data['label_colors'],
        x_tick_rotation=30, x_tick_align='right',
        show_legend=(idx == 0), legend_loc='upper right',
        visible_spines=['left', 'bottom'], params=PLOT_PARAMS,
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

# =============================================================================
# Pylustrator Layout
# =============================================================================

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(18.280000/2.54, 11.440000/2.54, forward=True)
plt.figure(1).ax_dict["ROI_Legend"].set(position=[-0.005881, 0.08207, 1.039, 0.1966])
plt.figure(1).ax_dict["ROI_Legend"].set_position([-0.001248, -0.068241, 1.029054, 0.228807])
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set(position=[0.549, 0.791, 0.417, 0.156])
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set_position([0.548451, 0.756775, 0.416583, 0.181546])
plt.figure(1).ax_dict["RSA_2_strategy"].set(position=[0.549, 0.5385, 0.417, 0.156])
plt.figure(1).ax_dict["RSA_2_strategy"].set_position([0.548451, 0.462876, 0.416583, 0.181546])
plt.figure(1).ax_dict["RSA_3_checkmate"].set(position=[0.549, 0.2859, 0.417, 0.156])
plt.figure(1).ax_dict["RSA_3_checkmate"].set_position([0.548451, 0.168977, 0.416583, 0.181546])
plt.figure(1).ax_dict["SVM_1_visual_similarity"].set_position([0.060939, 0.756775, 0.416583, 0.181546])
plt.figure(1).ax_dict["SVM_2_strategy"].set(position=[0.061, 0.5385, 0.417, 0.156])
plt.figure(1).ax_dict["SVM_2_strategy"].set_position([0.060939, 0.462876, 0.416583, 0.181546])
plt.figure(1).ax_dict["SVM_3_checkmate"].set(position=[0.061, 0.2859, 0.417, 0.156])
plt.figure(1).ax_dict["SVM_3_checkmate"].set_position([0.060939, 0.168977, 0.416583, 0.181546])
plt.figure(1).text(0.27, 0.98, 'Subcortical Decoding', transform=plt.figure(1).transFigure, fontsize=7., weight='bold', ha='center')
plt.figure(1).texts[0].set_position([0.269730, 0.976725])
plt.figure(1).text(0.76, 0.98, 'Subcortical Brain-Model RSA', transform=plt.figure(1).transFigure, fontsize=7., weight='bold', ha='center')
plt.figure(1).texts[1].set_position([0.759239, 0.976725])
#% end: automatic generated code from pylustrator

if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()

save_axes_svgs(fig1, FIGURES_DIR, 'subcortical_svm')
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'subcortical_svm_panel.pdf')

logger.info("Panel: Subcortical SVM + RSA complete")
log_script_end(logger)
