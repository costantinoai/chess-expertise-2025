"""
Pylustrator-driven layout for behavioral RSA figures.

This script builds all behavioral panels using our existing primitives and
styling, but does NOT arrange them into a grid. Use pylustrator to interactively
position and size the axes, then save to inject the arrangement code.

Panels created (axes labels in parentheses):
- Behavioral RDM (Experts)        [A1_RDM_Experts]
- Behavioral RDM (Novices)        [A2_RDM_Novices]
- Directional preference (Experts)[A3_DirPref_Experts]
- Directional preference (Novices)[A4_DirPref_Novices]
- MDS embedding (Experts)         [B1_MDS_Experts]
- MDS embedding (Novices)         [B2_MDS_Novices]
- Stimulus selection (Experts)    [C1_Choice_Experts]
- Stimulus selection (Novices)    [C2_Choice_Novices]
- RSA model correlations          [D_RSA_Models]

Usage:
    python 91_plot_behavioral_panels.py
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
import pandas as pd
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
    save_panel_svg,
    save_axes_pdfs,
    save_panel_pdf,
)


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR_NAME = None
RESULTS_BASE = script_dir / "results"

RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern="*_behavioral_rsa",
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=["figures"],
    require_exists=True,
    verbose=True,
)

FIGURES_DIR = RESULTS_DIR / "figures"


# =============================================================================
# Setup
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
# Load results (same as 02_plot_behavioral_results.py)
# =============================================================================

logger.info("Loading behavioral results...")

expert_rdm = np.load(RESULTS_DIR / "expert_behavioral_rdm.npy")
novice_rdm = np.load(RESULTS_DIR / "novice_behavioral_rdm.npy")
expert_dsm = np.load(RESULTS_DIR / "expert_directional_dsm.npy")
novice_dsm = np.load(RESULTS_DIR / "novice_directional_dsm.npy")
expert_mds_coords = np.load(RESULTS_DIR / "expert_mds_coords.npy")
novice_mds_coords = np.load(RESULTS_DIR / "novice_mds_coords.npy")

with open(RESULTS_DIR / "pairwise_data.pkl", "rb") as f:
    pairwise_data = pickle.load(f)
expert_pairwise = pairwise_data["expert_pairwise"]
novice_pairwise = pairwise_data["novice_pairwise"]

with open(RESULTS_DIR / "correlation_results.pkl", "rb") as f:
    correlation_results = pickle.load(f)
expert_corrs = correlation_results["expert"]
novice_corrs = correlation_results["novice"]
exp_p_fdr_map = correlation_results.get("expert_p_fdr")
nov_p_fdr_map = correlation_results.get("novice_p_fdr")

stimuli_df = load_stimulus_metadata()
strat_colors, strat_alphas = compute_stimulus_palette(stimuli_df)

# Global symmetric color scale for RDM/DSM panels
global_max_abs = max(
    np.abs(expert_rdm).max(),
    np.abs(novice_rdm).max(),
    np.abs(expert_dsm).max(),
    np.abs(novice_dsm).max()
)
behavioral_vmin = -global_max_abs
behavioral_vmax = global_max_abs

logger.info("Behavioral data loaded successfully\n")


# =============================================================================
# Style
# =============================================================================

apply_nature_rc()


# =============================================================================
# Create figure and independent axes (pylustrator will arrange)
# =============================================================================

fig = plt.figure(1)

# Panel A1: Behavioral RDM (Experts)
ax_A1 = plt.axes(); ax_A1.set_label('A1_RDM_Experts')
plot_rdm_on_ax(
    ax=ax_A1,
    rdm=expert_rdm,
    colors=strat_colors,
    alphas=strat_alphas,
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
    show_colorbar=False,
)
set_axis_title(ax_A1, title="Behavioral RDM", subtitle="Experts")
ax_A1.set_xticks([]); ax_A1.set_yticks([])

# Panel A2: Behavioral RDM (Novices)
ax_A2 = plt.axes(); ax_A2.set_label('A2_RDM_Novices')
plot_rdm_on_ax(
    ax=ax_A2,
    rdm=novice_rdm,
    colors=strat_colors,
    alphas=strat_alphas,
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
    show_colorbar=False,
)
set_axis_title(ax_A2, title="Behavioral RDM", subtitle="Novices")
ax_A2.set_xticks([]); ax_A2.set_yticks([])

# Panel A3: Directional preference (Experts)
ax_A3 = plt.axes(); ax_A3.set_label('A3_DirPref_Experts')
plot_rdm_on_ax(
    ax=ax_A3,
    rdm=expert_dsm,
    colors=strat_colors,
    alphas=strat_alphas,
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
    show_colorbar=False,
)
set_axis_title(ax_A3, title="Directional preference", subtitle="Experts")
ax_A3.set_xticks([]); ax_A3.set_yticks([])

# Panel A4: Directional preference (Novices)
ax_A4 = plt.axes(); ax_A4.set_label('A4_DirPref_Novices')
plot_rdm_on_ax(
    ax=ax_A4,
    rdm=novice_dsm,
    colors=strat_colors,
    alphas=strat_alphas,
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
    show_colorbar=False,
)
set_axis_title(ax_A4, title="Directional preference", subtitle="Novices")
ax_A4.set_xticks([]); ax_A4.set_yticks([])

# Panel B1: MDS embedding (Experts)
ax_B1 = plt.axes(); ax_B1.set_label('B1_MDS_Experts')
plot_2d_embedding_on_ax(
    ax=ax_B1,
    coords=expert_mds_coords,
    point_colors=strat_colors,
    point_alphas=strat_alphas,
    params=PLOT_PARAMS,
    hide_tick_marks=True,
    x_label='Dimension 1',
    y_label='Dimension 2',
)
set_axis_title(ax_B1, title="MDS embedding", subtitle="Experts")

# Panel B2: MDS embedding (Novices)
ax_B2 = plt.axes(); ax_B2.set_label('B2_MDS_Novices')
plot_2d_embedding_on_ax(
    ax=ax_B2,
    coords=novice_mds_coords,
    point_colors=strat_colors,
    point_alphas=strat_alphas,
    params=PLOT_PARAMS,
    hide_tick_marks=True,
    x_label='Dimension 1',
    y_label='Dimension 2',
)
set_axis_title(ax_B2, title="MDS embedding", subtitle="Novices")

# Panel C1: Stimulus selection frequency (Experts)
ax_C1 = plt.axes(); ax_C1.set_label('C1_Choice_Experts')
if 'count' in expert_pairwise.columns:
    frequency_exp = expert_pairwise.groupby('better')['count'].sum().sort_index()
else:
    frequency_exp = expert_pairwise['better'].value_counts().sort_index()

legend_items = [
    ('Checkmate', strat_colors[0], 0.7),
    ('Non-checkmate', strat_colors[len(strat_colors)//2], 0.7),
]

plot_counts_on_ax(
    ax=ax_C1,
    x_values=frequency_exp.index,
    counts=frequency_exp.values,
    colors=strat_colors[:len(frequency_exp)],
    alphas=strat_alphas[:len(frequency_exp)],
    xlabel='Stimulus ID',
    ylabel='Selection count',
    title='Stimulus selection frequency',
    subtitle='Experts',
    legend=legend_items,
    params=PLOT_PARAMS,
)

# Panel C2: Stimulus selection frequency (Novices)
ax_C2 = plt.axes(); ax_C2.set_label('C2_Choice_Novices')
if 'count' in novice_pairwise.columns:
    frequency_nov = novice_pairwise.groupby('better')['count'].sum().sort_index()
else:
    frequency_nov = novice_pairwise['better'].value_counts().sort_index()

plot_counts_on_ax(
    ax=ax_C2,
    x_values=frequency_nov.index,
    counts=frequency_nov.values,
    colors=strat_colors[:len(frequency_nov)],
    alphas=strat_alphas[:len(frequency_nov)],
    xlabel='Stimulus ID',
    ylabel='Selection count',
    title='Stimulus selection frequency',
    subtitle='Novices',
    legend=legend_items,
    params=PLOT_PARAMS,
)

# Panel D: RSA model correlations
ax_D = plt.axes(); ax_D.set_label('D_RSA_Models')

# Extract and reorder correlations to MODEL_ORDER
r_exp = [res[1] for res in expert_corrs]
ci_exp = [(res[3], res[4]) for res in expert_corrs]
# Prefer FDR-corrected p-values if available
if isinstance(exp_p_fdr_map, dict):
    p_exp = [exp_p_fdr_map.get(res[0], res[2]) for res in expert_corrs]
else:
    p_exp = [res[2] for res in expert_corrs]

r_nov = [res[1] for res in novice_corrs]
ci_nov = [(res[3], res[4]) for res in novice_corrs]
if isinstance(nov_p_fdr_map, dict):
    p_nov = [nov_p_fdr_map.get(res[0], res[2]) for res in novice_corrs]
else:
    p_nov = [res[2] for res in novice_corrs]

column_labels = [res[0] for res in expert_corrs]
idx_order = [column_labels.index(lbl) for lbl in MODEL_ORDER]

r_exp_ordered = [r_exp[i] for i in idx_order]
ci_exp_ordered = [ci_exp[i] for i in idx_order]
p_exp_ordered = [p_exp[i] for i in idx_order]

r_nov_ordered = [r_nov[i] for i in idx_order]
ci_nov_ordered = [ci_nov[i] for i in idx_order]
p_nov_ordered = [p_nov[i] for i in idx_order]

x = np.arange(len(MODEL_LABELS_PRETTY))
plot_grouped_bars_on_ax(
    ax=ax_D,
    x_positions=x,
    group1_values=r_exp_ordered,
    group1_cis=ci_exp_ordered,
    group1_color=COLORS_EXPERT_NOVICE['expert'],
    group2_values=r_nov_ordered,
    group2_cis=ci_nov_ordered,
    group2_color=COLORS_EXPERT_NOVICE['novice'],
    group1_label="Experts",
    group2_label="Novices",
    group1_pvals=p_exp_ordered,
    group2_pvals=p_nov_ordered,
    ylim=(-0.2, 1.0),
)

ax_D.axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)
ax_D.set_xticks(x)
ax_D.set_xticklabels(MODEL_LABELS_PRETTY, rotation=0, ha='center',
                     fontsize=PLOT_PARAMS['font_size_tick'])
ax_D.set_ylabel(PLOT_PARAMS['ylabel_correlation_r'], fontsize=PLOT_PARAMS['font_size_label'])
ax_D.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
ax_D.legend(fontsize=PLOT_PARAMS['font_size_legend'], frameon=False, loc='upper left')
set_axis_title(ax_D, title="Behavioral-model RSA", subtitle="FDR corrected")


# Provide an axis dictionary for pylustrator convenience (fail fast)
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}


# =============================================================================
# Show pylustrator window
# =============================================================================

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(16.820000/2.54, 15.610000/2.54, forward=True)
plt.figure(1).ax_dict["A1_RDM_Experts"].set(position=[0.24, 0.7455, 0.1532, 0.1912])
plt.figure(1).ax_dict["A1_RDM_Experts"].set_position([0.350733, 0.681630, 0.219633, 0.236670])
plt.figure(1).ax_dict["A1_RDM_Experts"].texts[0].set(position=(0.5, 1.064))
plt.figure(1).ax_dict["A1_RDM_Experts"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["A2_RDM_Novices"].set(position=[0.24, 0.5035, 0.1532, 0.1912])
plt.figure(1).ax_dict["A2_RDM_Novices"].set_position([0.350733, 0.376982, 0.219633, 0.236670])
plt.figure(1).ax_dict["A2_RDM_Novices"].texts[0].set(position=(0.5, 1.067))
plt.figure(1).ax_dict["A2_RDM_Novices"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["A3_DirPref_Experts"].set(position=[0.05331, 0.7455, 0.1532, 0.1912])
plt.figure(1).ax_dict["A3_DirPref_Experts"].set_position([0.079160, 0.681630, 0.219633, 0.236670])
plt.figure(1).ax_dict["A3_DirPref_Experts"].texts[0].set(position=(0.5, 1.065))
plt.figure(1).ax_dict["A3_DirPref_Experts"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["A4_DirPref_Novices"].set(position=[0.05331, 0.5035, 0.1532, 0.1912])
plt.figure(1).ax_dict["A4_DirPref_Novices"].set_position([0.079160, 0.376982, 0.219633, 0.236670])
plt.figure(1).ax_dict["A4_DirPref_Novices"].texts[0].set(position=(0.5, 1.067))
plt.figure(1).ax_dict["A4_DirPref_Novices"].texts[1].set(position=(0.5, 1.016))
plt.figure(1).ax_dict["B1_MDS_Experts"].set(position=[0.6537, 0.6794, 0.2967, 0.2408])
plt.figure(1).ax_dict["B1_MDS_Experts"].texts[0].set(position=(0.5, 1.069))
plt.figure(1).ax_dict["B1_MDS_Experts"].texts[1].set(position=(0.5, 1.018))
plt.figure(1).ax_dict["B2_MDS_Novices"].set(position=[0.6537, 0.3751, 0.2967, 0.2408])
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
plt.figure(1).text(0.0295, 0.9591, 'a', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.042866, 0.948480])
plt.figure(1).text(0.6635, 0.9591, 'b', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.965172, 0.948480])
plt.figure(1).text(0.0295, 0.4715, 'c', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.042866, 0.334723])
plt.figure(1).text(0.6635, 0.4715, 'd', transform=plt.figure(1).transFigure, fontsize=8., weight='bold')  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.965172, 0.334723])
#% end: automatic generated code from pylustrator
plt.show()

# Save each axis separately first, then full panel
fig = plt.gcf()
save_axes_svgs(fig, FIGURES_DIR, 'behavioral')
save_axes_pdfs(fig, FIGURES_DIR, 'behavioral')
save_panel_svg(fig, FIGURES_DIR / 'panels' / 'behavioral_rsa_panel.svg')
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'behavioral_rsa_panel.pdf')

log_script_end(logger)
