"""
Pylustrator-driven manifold panels (two separate figures).

Figure 1: Combined PR barplots (top: group means; bottom: expert−novice differences)
Figure 2: PR matrix, PCA loadings, PCA 2D projection, feature importance

All plots are created using our existing primitives/helpers; no subplot layout
is defined here. Arrange interactively in pylustrator and save to inject layout code.

Usage:
    python 91_plot_manifold_panels.py
"""

import sys
import os
import pickle
from pathlib import Path
# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

# Import pylustrator BEFORE creating figures
import pylustrator
pylustrator.start()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.plotting import (
    apply_nature_rc,
    set_axis_title,
    plot_matrix_on_ax,
    plot_grouped_bars_on_ax,
    plot_2d_embedding_on_ax,
    style_spines,
    COLORS_EXPERT_NOVICE,
    PLOT_PARAMS,
    save_axes_svgs,
    save_panel_svg,
)


# =============================================================================
# Configuration and results
# =============================================================================

RESULTS_DIR_NAME = None
RESULTS_BASE = script_dir / "results"

RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern="*_manifold",
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
    log_name="pylustrator_manifold_panels.log",
)

logger.info("Loading manifold results...")
with open(RESULTS_DIR / "pr_results.pkl", "rb") as f:
    results = pickle.load(f)

summary_stats = results['summary_stats']
stats_results = results['stats_results']
roi_info = results['roi_info']
roi_labels = results['roi_labels']
classifier = results['classifier']
pca2d = results.get('pca2d')
pr_matrix_pack = results.get('pr_matrix')

if pca2d is None or pr_matrix_pack is None:
    raise RuntimeError("Missing pca2d or pr_matrix in results. Re-run 01_manifold_analysis.py.")

logger.info("Data loaded successfully")

apply_nature_rc()


# =============================================================================
# Figure 1: Combined PR barplots (two independent axes)
# =============================================================================

fig1 = plt.figure(1)

# Merge to get pretty names and colors
summary_with_info = summary_stats.merge(
    roi_info[['roi_id', 'pretty_name', 'color']],
    left_on='ROI_Label', right_on='roi_id', how='left'
)
stats_with_info = stats_results.merge(
    roi_info[['roi_id', 'pretty_name', 'color']],
    left_on='ROI_Label', right_on='roi_id', how='left'
)

# Prepare data ordered by ROI label
expert_data = summary_with_info[summary_with_info['group'] == 'expert'].sort_values('ROI_Label')
novice_data = summary_with_info[summary_with_info['group'] == 'novice'].sort_values('ROI_Label')

roi_names = expert_data['pretty_name'].tolist()
roi_names = [name.replace("\n", " ") for name in roi_names]
roi_colors = expert_data['color'].tolist()

sig_col = 'p_val_fdr'
pvals = stats_with_info.sort_values('ROI_Label')[sig_col].tolist()
is_significant = [p < config['ALPHA'] for p in pvals]

x = np.arange(len(roi_names))

# Top bar axis: group means
ax_bars_top = plt.axes(); ax_bars_top.set_label('Bars_Top_Mean_PR')

exp_vals = expert_data['mean_PR'].tolist()
exp_cis = list(zip(expert_data['ci_low'].values, expert_data['ci_high'].values))
nov_vals = novice_data['mean_PR'].tolist()
nov_cis = list(zip(novice_data['ci_low'].values, novice_data['ci_high'].values))

plot_grouped_bars_on_ax(
    ax=ax_bars_top,
    x_positions=x,
    group1_values=exp_vals,
    group1_cis=exp_cis,
    group1_color=roi_colors,
    group2_values=nov_vals,
    group2_cis=nov_cis,
    group2_color=roi_colors,
    group1_label='Expert',
    group2_label='Novice',
    comparison_pvals=pvals,
    params=PLOT_PARAMS,
)

ax_bars_top.set_ylabel('Mean PR (95% CI)', fontsize=PLOT_PARAMS['font_size_label'])
ax_bars_top.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
set_axis_title(ax_bars_top, title='Participation Ratio', subtitle='FDR p < .05')
style_spines(ax_bars_top, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
ax_bars_top.set_xlim(-0.5, len(roi_names) - 0.5)
ax_bars_top.set_xticks(x)
ax_bars_top.set_xticklabels([])

# Bottom bar axis: differences
ax_bars_bottom = plt.axes(); ax_bars_bottom.set_label('Bars_Bottom_Diff_PR')

diff_vals = stats_with_info.sort_values('ROI_Label')['mean_diff'].tolist()
diff_cis = list(zip(stats_with_info.sort_values('ROI_Label')['ci95_low'].values,
                    stats_with_info.sort_values('ROI_Label')['ci95_high'].values))

plot_grouped_bars_on_ax(
    ax=ax_bars_bottom,
    x_positions=x,
    group1_values=diff_vals,
    group1_cis=diff_cis,
    group1_color=roi_colors,
    group1_pvals=pvals,
    bar_width_multiplier=2.0,
    params=PLOT_PARAMS,
)

ax_bars_bottom.set_ylabel('ΔPR (Expert − Novice) (95% CI)', fontsize=PLOT_PARAMS['font_size_label'])
ax_bars_bottom.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
set_axis_title(ax_bars_bottom, title='Participation Ratio Difference', subtitle='')
style_spines(ax_bars_bottom, visible_spines=['left'], params=PLOT_PARAMS)
ax_bars_bottom.set_xlim(-0.5, len(roi_names) - 0.5)
ax_bars_bottom.set_xticks(x)
ax_bars_bottom.set_xticklabels(roi_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
for ticklabel, sig, color in zip(ax_bars_bottom.get_xticklabels(), is_significant, roi_colors):
    ticklabel.set_color(color if sig else '#999999')

logger.info("✓ Figure 1 axes ready: combined PR barplots")


# =============================================================================
# Figure 2: Matrix, loadings, PCA 2D, feature importance (independent axes)
# =============================================================================

fig2 = plt.figure(2)

# A: PR matrix
ax_A = plt.axes(); ax_A.set_label('A_PR_Matrix')
matrix = pr_matrix_pack['matrix']
n_experts = pr_matrix_pack['n_experts']
plot_matrix_on_ax(
    ax=ax_A,
    matrix=matrix,
    title='PR Profiles across Subjects',
    subtitle=None,
    cmap='mako',
    show_colorbar=False,
    xticklabels=None,
    yticklabels=None,
    square=False,
    params=PLOT_PARAMS,
)
ax_A.set_xlabel('')
ax_A.set_ylabel('Subjects (Experts on top)', fontsize=PLOT_PARAMS['font_size_tick'])
if 0 < n_experts < matrix.shape[0]:
    ax_A.axhline(n_experts, color='black', linewidth=PLOT_PARAMS['plot_linewidth'])

# D: PCA loadings heatmap
ax_D = plt.axes(); ax_D.set_label('D_PCA_Loadings')
from common.plotting.colors import _make_brain_cmap
brain_cmap = _make_brain_cmap()
loadings = pca2d['components']
max_abs = np.abs(loadings).max()
roi_name_map = dict(zip(roi_info['roi_id'].values, roi_info['pretty_name'].values))
roi_pretty_names = [roi_name_map.get(int(lbl), str(lbl)) for lbl in roi_labels]
roi_pretty_names = [name.replace("\n", " ") for name in roi_pretty_names]
plot_matrix_on_ax(
    ax=ax_D,
    matrix=loadings,
    title='ROI Contributions to PCA Components',
    subtitle=None,
    cmap=brain_cmap,
    vmin=-max_abs,
    vmax=max_abs,
    center=0,
    show_colorbar=False,
    xticklabels=roi_pretty_names,
    yticklabels=['PC 1', 'PC 2'],
    square=False,
    params=PLOT_PARAMS,
)
ax_D.set_xlabel('')
for ticklabel in ax_D.get_xticklabels():
    ticklabel.set_color('gray')

# B: PCA 2D projection
ax_B = plt.axes(); ax_B.set_label('B_PCA_Projection')
coords = pca2d['coords']
expl = pca2d['explained']
labels = pca2d['labels']
bnd = pca2d['boundary']
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
x_range = x_max - x_min
y_range = y_max - y_min
padding = 0.05
xlim = (x_min - padding * x_range, x_max + padding * x_range)
ylim = (y_min - padding * y_range, y_max + padding * y_range)
point_colors = [COLORS_EXPERT_NOVICE['expert'] if lbl == 1 else COLORS_EXPERT_NOVICE['novice'] for lbl in labels]
point_alphas = [0.7] * len(labels)
plot_2d_embedding_on_ax(
    ax=ax_B,
    coords=coords,
    point_colors=point_colors,
    point_alphas=point_alphas,
    xlim=xlim,
    ylim=ylim,
    fill={
        'xx': bnd['xx'],
        'yy': bnd['yy'],
        'Z': bnd['Z'],
        'colors': [COLORS_EXPERT_NOVICE['novice'], COLORS_EXPERT_NOVICE['expert']],
        'alpha': 0.15,
        'levels': [0, 0.5, 1],
    },
    params=PLOT_PARAMS,
)
set_axis_title(ax_B, title='PCA Projection of PR Profiles', subtitle='')
ax_B.set_xlabel(f'PC 1 ({expl[0]:.1f}% var)', fontsize=PLOT_PARAMS['font_size_label'])
ax_B.set_ylabel(f'PC 2 ({expl[1]:.1f}% var)', fontsize=PLOT_PARAMS['font_size_label'])

# C: Feature importance
ax_C = plt.axes(); ax_C.set_label('C_Feature_Importance')
weights = classifier.coef_[0]
top_n = 10
abs_weights = np.abs(weights)
top_idx = np.argsort(abs_weights)[-top_n:]
top_weights = weights[top_idx]
roi_name_map2 = dict(zip(roi_info['roi_id'], roi_info['pretty_name']))
top_names = [roi_name_map2.get(i+1, f'ROI_{i+1}') for i in top_idx]
top_names = [name.replace("\n", " ") for name in top_names]
top_colors = [COLORS_EXPERT_NOVICE['novice'] if w < 0 else COLORS_EXPERT_NOVICE['expert'] for w in top_weights]
y_pos = np.arange(len(top_names))
ax_C.barh(
    y_pos,
    top_weights,
    color=top_colors,
    alpha=PLOT_PARAMS.get('bar_alpha', 0.7),
    edgecolor='black',
    linewidth=PLOT_PARAMS['plot_linewidth']
)
ax_C.set_yticks(y_pos)
ax_C.set_yticklabels(top_names, fontsize=PLOT_PARAMS['font_size_label'], color="gray")
ax_C.yaxis.tick_right()
ax_C.yaxis.set_label_position("right")
ax_C.set_xlabel('Weight in Original ROI Space', fontsize=PLOT_PARAMS['font_size_label'])
ax_C.tick_params(axis='x', labelsize=PLOT_PARAMS['font_size_tick'])
ax_C.set_xlim(-1.0, 1.0)
ax_C.set_ylim(-1, 10.0)
ax_C.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax_C.axvline(x=0, color='black', linestyle='-', linewidth=PLOT_PARAMS['plot_linewidth'], alpha=PLOT_PARAMS.get('line_alpha', 0.5))
set_axis_title(ax_C, title='Top 10 Contributions to Classification', subtitle='')
from matplotlib.font_manager import FontProperties
arrow_font = FontProperties(family=PLOT_PARAMS["font_family"], size=PLOT_PARAMS['font_size_title'])
ax_C.text(0.15, 1.04, "← Higher PR predictive of Novices", ha="center", va="center",
          transform=ax_C.transAxes, fontproperties=arrow_font,
          color=COLORS_EXPERT_NOVICE['novice'])
ax_C.text(0.85, 1.04, "Higher PR predictive of Experts →", ha="center", va="center",
          transform=ax_C.transAxes, fontproperties=arrow_font,
          color=COLORS_EXPERT_NOVICE['expert'])

logger.info("✓ Figure 2 axes ready: matrix, loadings, PCA projection, feature importance")


# =============================================================================
# Figure 3: PR vs Voxel size scatters (Experts, Novices, ΔPR vs avg voxels)
# =============================================================================

fig3 = plt.figure(3)

voxel_corr = results.get('voxel_corr', {})
group_avg = voxel_corr.get('group_avg')
diff_data = voxel_corr.get('diff_data')

if group_avg is not None and diff_data is not None:
    stats_vox = voxel_corr.get('stats', {})
    def _add_regression(ax, xvals, slope, intercept, r=None, p=None):
        if len(xvals) == 0:
            return
        x0, x1 = float(np.min(xvals)), float(np.max(xvals))
        y0 = intercept + slope * x0
        y1 = intercept + slope * x1
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=PLOT_PARAMS['plot_linewidth'], alpha=0.7, zorder=1)
        if r is not None and p is not None:
            ax.text(0.98, 0.02, f"r={r:.2f}, p={p:.3f}", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=PLOT_PARAMS['font_size_tick'], color='#666666')

    # Experts
    ax_E = plt.axes(); ax_E.set_label('E_PR_vs_Voxels_Experts')
    ga_exp = group_avg[group_avg['group'] == 'expert']
    ms = PLOT_PARAMS.get('marker_size', 1.0) * 30
    ax_E.scatter(ga_exp['n_voxels'], ga_exp['PR'], c=ga_exp['color'], s=ms, alpha=0.8, edgecolors='none')
    ax_E.set_xlabel('ROI voxel count', fontsize=PLOT_PARAMS['font_size_label'])
    ax_E.set_ylabel('Mean PR', fontsize=PLOT_PARAMS['font_size_label'])
    ax_E.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
    style_spines(ax_E, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    set_axis_title(ax_E, title='PR vs ROI size', subtitle='Experts')
    stE = stats_vox.get('expert', {})
    if stE:
        _add_regression(ax_E, ga_exp['n_voxels'].values, stE.get('slope', 0.0), stE.get('intercept', 0.0), stE.get('r', None), stE.get('p', None))

    # Novices
    ax_F = plt.axes(); ax_F.set_label('F_PR_vs_Voxels_Novices')
    ga_nov = group_avg[group_avg['group'] == 'novice']
    ms = PLOT_PARAMS.get('marker_size', 1.0) * 30
    ax_F.scatter(ga_nov['n_voxels'], ga_nov['PR'], c=ga_nov['color'], s=ms, alpha=0.8, edgecolors='none')
    ax_F.set_xlabel('ROI voxel count', fontsize=PLOT_PARAMS['font_size_label'])
    ax_F.set_ylabel('Mean PR', fontsize=PLOT_PARAMS['font_size_label'])
    ax_F.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
    style_spines(ax_F, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    set_axis_title(ax_F, title='PR vs ROI size', subtitle='Novices')
    stN = stats_vox.get('novice', {})
    if stN:
        _add_regression(ax_F, ga_nov['n_voxels'].values, stN.get('slope', 0.0), stN.get('intercept', 0.0), stN.get('r', None), stN.get('p', None))

    # Differences (Expert − Novice)
    ax_G = plt.axes(); ax_G.set_label('G_PRdiff_vs_VoxelsAvg')
    ms = PLOT_PARAMS.get('marker_size', 1.0) * 30
    ax_G.scatter(diff_data['n_voxels_avg'], diff_data['PR_diff'], c=diff_data['color'], s=ms, alpha=0.8, edgecolors='none')
    ax_G.axhline(0, color='black', linewidth=PLOT_PARAMS['plot_linewidth'], alpha=PLOT_PARAMS.get('line_alpha', 0.5))
    ax_G.set_xlabel('ROI voxel count (avg E/N)', fontsize=PLOT_PARAMS['font_size_label'])
    ax_G.set_ylabel('ΔPR (Expert − Novice)', fontsize=PLOT_PARAMS['font_size_label'])
    ax_G.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
    style_spines(ax_G, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    set_axis_title(ax_G, title='ΔPR vs ROI size', subtitle='')
    stD = stats_vox.get('diff', {})
    if stD:
        _add_regression(ax_G, diff_data['n_voxels_avg'].values, stD.get('slope', 0.0), stD.get('intercept', 0.0), stD.get('r', None), stD.get('p', None))

logger.info("✓ Figure 3 axes ready: PR vs voxel size scatters")


# For pylustrator convenience: label axes dicts (fail fast on error)
fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}
fig2.ax_dict = {ax.get_label(): ax for ax in fig2.axes}
fig3.ax_dict = {ax.get_label(): ax for ax in fig3.axes}


# =============================================================================
# Show pylustrator window
# =============================================================================

#% start: automatic generated code from pylustrator
plt.figure(2).ax_dict = {ax.get_label(): ax for ax in plt.figure(2).axes}
import matplotlib as mpl
getattr(plt.figure(2), '_pylustrator_init', lambda: ...)()
plt.figure(2).ax_dict["A_PR_Matrix"].set(position=[0.0903, 0.5049, 0.292, 0.4608])
plt.figure(2).ax_dict["B_PCA_Projection"].set(position=[0.4251, 0.756, 0.2102, 0.2096])
plt.figure(2).ax_dict["C_Feature_Importance"].set(position=[0.4251, 0.5049, 0.2098, 0.1935])
plt.figure(2).ax_dict["C_Feature_Importance"].spines[['right', 'top']].set_visible(False)
plt.figure(2).ax_dict["C_Feature_Importance"].texts[0].set(position=(0.15, 1.025), fontsize=6.)
plt.figure(2).ax_dict["C_Feature_Importance"].texts[1].set(position=(0.85, 1.025), fontsize=6.)
plt.figure(2).ax_dict["D_PCA_Loadings"].set(position=[0.0876, 0.4467, 0.292, 0.02706])
plt.figure(2).ax_dict["D_PCA_Loadings"].text(0.4908, 1.3920, 'ROI Contributions to PCA Components', transform=plt.figure(2).ax_dict["D_PCA_Loadings"].transAxes, ha='center', fontsize=7., weight='bold')  # id=plt.figure(2).ax_dict["D_PCA_Loadings"].texts[0].new
plt.figure(2).ax_dict["D_PCA_Loadings"].title.set(visible=False)
#% end: automatic generated code from pylustrator
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["Bars_Bottom_Diff_PR"].set(position=[0.1886, 0.2645, 0.5077, 0.2393])
plt.figure(1).ax_dict["Bars_Bottom_Diff_PR"].title.set(visible=False)
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].set(position=[0.1886, 0.5289, 0.5077, 0.1775])
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].texts[14].set(position=(0.4999, 1.023))
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].texts[15].set(position=(0.4999, 0.952))
plt.figure(1).text(0.4424, 0.5084, 'Participation Ratio Difference', transform=plt.figure(1).transFigure, ha='center', fontsize=8., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.4424, 0.4952, 'FDR p < .05', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[1].new
#% end: automatic generated code from pylustrator
#% start: automatic generated code from pylustrator
plt.figure(3).ax_dict = {ax.get_label(): ax for ax in plt.figure(3).axes}
import matplotlib as mpl
getattr(plt.figure(3), '_pylustrator_init', lambda: ...)()
plt.figure(3).ax_dict["E_PR_vs_Voxels_Experts"].set(position=[0.04688, 0.7026, 0.2676, 0.2557], ylim=(17., 32.))
plt.figure(3).ax_dict["E_PR_vs_Voxels_Experts"].texts[0].set(position=(0.5, 1.06))
plt.figure(3).ax_dict["F_PR_vs_Voxels_Novices"].set(position=[0.3841, 0.7026, 0.2676, 0.2557], ylim=(17., 32.))
plt.figure(3).ax_dict["F_PR_vs_Voxels_Novices"].texts[0].set(position=(0.5, 1.06))
plt.figure(3).ax_dict["F_PR_vs_Voxels_Novices"].texts[1].set(position=(0.5, 1.008))
plt.figure(3).ax_dict["G_PRdiff_vs_VoxelsAvg"].set(position=[0.7155, 0.7026, 0.2676, 0.2557], ylim=(-7., 5.))
plt.figure(3).text(0.8462, 0.9605, 'Experts - Novices', transform=plt.figure(3).transFigure, ha='center')  # id=plt.figure(3).texts[0].new
#% end: automatic generated code from pylustrator
plt.show()

# After arranging in pylustrator, save each axis separately (and the full panels)
save_axes_svgs(fig1, FIGURES_DIR, 'manifold_bars')
save_axes_svgs(fig2, FIGURES_DIR, 'manifold_matrix_pca')
save_axes_svgs(fig3, FIGURES_DIR, 'manifold_pr_voxels')

save_panel_svg(fig1, FIGURES_DIR / 'panels' / 'manifold_bars_panel.svg')
save_panel_svg(fig2, FIGURES_DIR / 'panels' / 'manifold_matrix_pca_panel.svg')
save_panel_svg(fig3, FIGURES_DIR / 'panels' / 'manifold_pr_voxels_panel.svg')

log_script_end(logger)
