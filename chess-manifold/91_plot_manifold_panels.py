"""
Generate Manifold PR Figure Panels (Pylustrator)
================================================

Creates publication-ready multi-panel figures for the manifold (Participation Ratio)
analysis. Uses pylustrator for interactive layout arrangement. The script builds
independent axes using standardized plotting primitives and then saves both
individual axes (SVG/PDF) and assembled panels (SVG/PDF) into the current
manifold results directory.

Figures Produced
----------------

Panel 1: Group PR Bars (Experts vs Novices)
- File: figures/panels/manifold_bars_panel.svg (and .pdf)
- Axes saved to figures/: manifold_bars_Bars_Top_Mean_PR.*, manifold_bars_Bars_Bottom_Diff_PR.*, manifold_bars_ROI_Groups_Legend.*
- Content: Top panel shows group mean PR with 95% CIs per ROI (Experts vs Novices);
  bottom panel shows ΔPR (Expert − Novice) with 95% CIs and FDR significance.

Panel 2: Matrix + PCA + Feature Importance
- File: figures/panels/manifold_matrix_pca_panel.svg (and .pdf)
- Axes saved to figures/: manifold_matrix_pca_A_PR_Matrix.*, manifold_matrix_pca_B_PCA_Projection.*, manifold_matrix_pca_C_Feature_Importance.*, manifold_matrix_pca_D_PCA_Loadings.*
- Content: Subject×ROI PR matrix (experts on top), 2D PCA projection with
  decision boundary, top-10 classifier feature importances in ROI space,
  and PCA component loadings per ROI.

Panel 3: PR vs ROI Size
- File: figures/panels/manifold_pr_voxels_panel.svg (and .pdf)
- Axes saved to figures/: manifold_pr_voxels_E_PR_vs_Voxels_Experts.*, manifold_pr_voxels_F_PR_vs_Voxels_Novices.*, manifold_pr_voxels_G_PRdiff_vs_VoxelsAvg.*
- Content: Scatter plots of PR versus ROI voxel counts for Experts and Novices,
  and ΔPR versus average ROI size, each with regression line and r/p annotation.

Inputs
------
- pr_results.pkl (from 01_manifold_analysis.py), containing at minimum:
  - summary_stats: per-ROI, per-group PR summary with CIs
  - stats_results: ROI-level group comparisons with FDR-corrected p-values
  - roi_info: ROI metadata with pretty names and colors
  - roi_labels: ordered ROI IDs
  - classifier: trained logistic regression (coef_ used for feature importance)
  - pca2d: dict with 'coords', 'explained', 'labels', 'boundary', 'components'
  - pr_matrix: dict with 'matrix' (subjects×ROIs) and 'n_experts'
  - voxel_corr: dict with 'group_avg', 'diff_data', and 'stats'

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- common.plotting primitives and style (apply_nature_rc, bars, heatmaps, scatter)
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-manifold/91_plot_manifold_panels.py
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

import matplotlib.pyplot as plt
import numpy as np

from common import setup_script, log_script_end
from common.plotting import (
    apply_nature_rc,
    set_axis_title,
    plot_matrix_on_ax,
    plot_grouped_bars_on_ax,
    plot_2d_embedding_on_ax,
    style_spines,
    COLORS_EXPERT_NOVICE,
    PLOT_PARAMS,
    CMAP_BRAIN,
    CMAP_SEQUENTIAL,
    cm_to_inches,
    save_axes_svgs,
    save_panel_pdf,
    create_roi_group_legend,
    embed_figure_on_ax,
    compute_ylim_range,
    format_roi_labels_and_colors,
    draw_regression_line,
)
from modules.plotting import (
    compute_limits_with_padding,
    pca_axis_labels_from_explained,
    plot_topn_feature_importance_on_ax,
)


# =============================================================================
# Configuration and results
# =============================================================================

RESULTS_DIR_NAME = None
RESULTS_BASE = script_dir / "results"

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='manifold',
    output_subdirs=['figures'],
    log_name='pylustrator_manifold_panels.log',
)
RESULTS_DIR = results_dir
FIGURES_DIR = dirs['figures']


# =============================================================================
# Common Data Loading
# =============================================================================
# Data used across multiple panels: summary_stats, stats_results, roi_info,
# roi_labels, classifier, pca2d, pr_matrix_pack

logger.info("Loading manifold results...")
with open(RESULTS_DIR / "pr_results.pkl", "rb") as f:
    results = pickle.load(f)

# Extract data tables from results pickle
# - summary_stats: per-ROI, per-group PR summary with mean, CI bounds (for barplots)
# - stats_results: ROI-level group comparisons with mean_diff, CIs, and FDR p-values
# - roi_info: ROI metadata including pretty_name, color, group/family
# - roi_labels: ordered list of ROI IDs used in matrices
# - classifier: trained logistic regression model (coef_ = feature importance)
# - pca2d: dict with PCA projection ('coords', 'explained', 'labels', 'boundary', 'components')
# - pr_matrix_pack: dict with PR matrix (subjects×ROIs) and 'n_experts' for split line
summary_stats = results['summary_stats']        # DataFrame: ROI_Label, group, mean_PR, ci_low, ci_high
stats_results = results['stats_results']        # DataFrame: ROI_Label, mean_diff, ci95_low, ci95_high, p_val_fdr
roi_info = results['roi_info']                  # DataFrame: roi_id, pretty_name, color, group/family
roi_labels = results['roi_labels']              # List[int]: ordered ROI IDs for matrix columns
classifier = results['classifier']              # LogisticRegression: trained model
pca2d = results.get('pca2d')                   # Dict: PCA 2D projection data
pr_matrix_pack = results.get('pr_matrix')      # Dict: PR matrix (subjects×ROIs) + n_experts

# Fail fast if critical data is missing
if pca2d is None or pr_matrix_pack is None:
    raise RuntimeError("Missing pca2d or pr_matrix in results. Re-run 01_manifold_analysis.py.")

logger.info("Data loaded successfully")

apply_nature_rc()


# =============================================================================
# Panel 1: Combined PR barplots (bars)
# =============================================================================
# This panel shows:
# - Top panel: Mean PR per ROI for Experts vs Novices (grouped bars with 95% CIs)
# - Bottom panel: ΔPR (Expert − Novice) per ROI (single bars with 95% CIs)
# - Both panels show FDR-corrected significance stars
# - X-axis labels colored by ROI group, gray if not significant

# Data preparation specific to this panel
# Establish consistent ROI order across all panels (sorted by ROI_Label)
# This ensures bars, labels, and colors align correctly
order_rois = stats_results.sort_values('ROI_Label')['ROI_Label'].tolist()

# Get formatted ROI names, colors, and label colors (gray for non-significant)
# Uses DRY helper that merges roi_info with stats and applies significance coloring
roi_names, roi_colors, label_colors = format_roi_labels_and_colors(
    stats_results.sort_values('ROI_Label'), roi_info, alpha=CONFIG['ALPHA']
)

# Extract FDR-corrected p-values for significance annotation (stars on bars)
pvals = stats_results.sort_values('ROI_Label')['p_val_fdr'].tolist()

# Create x-positions for bars (0, 1, 2, ..., n_rois-1)
x = np.arange(len(order_rois))

# Align summary_stats (mean PR, CIs) to the consistent ROI order
# summary_stats has separate rows for 'expert' and 'novice' groups
# We reindex to match order_rois, ensuring bars appear in correct positions
expert_data = (
    summary_stats[summary_stats['group'] == 'expert']
    .set_index('ROI_Label')
    .reindex(order_rois)  # Align to order_rois
)
novice_data = (
    summary_stats[summary_stats['group'] == 'novice']
    .set_index('ROI_Label')
    .reindex(order_rois)  # Align to order_rois
)

# Build figure and axes
fig1 = plt.figure(1)

# -----------------------------------------------------------------------------
# Panel 1A: Top bars - Mean PR per ROI (Experts vs Novices)
# -----------------------------------------------------------------------------
# Shows grouped bars comparing Expert (solid) vs Novice (hatched) mean PR
# Bars use ROI group colors; significance stars show FDR p < 0.05
ax_bars_top = plt.axes(); ax_bars_top.set_label('Bars_Top_Mean_PR')

# Extract mean PR values and 95% CIs for each group
exp_vals = expert_data['mean_PR'].tolist()
exp_cis = list(zip(expert_data['ci_low'].values, expert_data['ci_high'].values))
nov_vals = novice_data['mean_PR'].tolist()
nov_cis = list(zip(novice_data['ci_low'].values, novice_data['ci_high'].values))

# Plot grouped bars with significance annotation
plot_grouped_bars_on_ax(
    ax=ax_bars_top,
    x_positions=x,
    group1_values=exp_vals,              # Expert mean PR
    group1_cis=exp_cis,                  # Expert 95% CIs
    group1_color=roi_colors,             # ROI group colors (solid bars)
    group2_values=nov_vals,              # Novice mean PR
    group2_cis=nov_cis,                  # Novice 95% CIs
    group2_color=roi_colors,             # Same colors (hatched bars)
    group1_label='Expert',               # Legend label
    group2_label='Novice',               # Legend label
    comparison_pvals=pvals,              # FDR p-values for significance stars
    y_label='Mean PR (95% CI)',
    title='Participation Ratio',
    subtitle='FDR p < .05',
    hide_xticklabels=True,               # No x-labels on top panel
    show_legend=True,                    # Show Expert/Novice legend (solid/hatched)
    visible_spines=['left', 'bottom'],
    params=PLOT_PARAMS,
)

# -----------------------------------------------------------------------------
# Panel 1B: Bottom bars - ΔPR (Expert − Novice) per ROI
# -----------------------------------------------------------------------------
# Shows single bars for the difference in PR between Experts and Novices
# X-axis labels show ROI names, colored by group (gray if not significant)
ax_bars_bottom = plt.axes(); ax_bars_bottom.set_label('Bars_Bottom_Diff_PR')

# Align stats_results to order_rois and extract difference values and CIs
stats_ordered = stats_results.set_index('ROI_Label').reindex(order_rois)
diff_vals = stats_ordered['mean_diff'].tolist()      # Expert − Novice difference
diff_cis = list(zip(stats_ordered['ci95_low'].values,
                    stats_ordered['ci95_high'].values))  # 95% CIs for difference

# Plot single-group bars with ROI labels on x-axis
plot_grouped_bars_on_ax(
    ax=ax_bars_bottom,
    x_positions=x,
    group1_values=diff_vals,             # ΔPR values
    group1_cis=diff_cis,                 # 95% CIs
    group1_color=roi_colors,             # ROI group colors
    group1_pvals=pvals,                  # FDR p-values for significance stars
    bar_width_multiplier=2.0,            # Wider bars (single group)
    y_label='ΔPR (Expert − Novice) (95% CI)',
    title='Participation Ratio Difference',
    subtitle='FDR p < .05',
    xtick_labels=roi_names,              # Show ROI names
    x_label_colors=label_colors,         # Color by significance (gray if p ≥ 0.05)
    x_tick_rotation=30,
    x_tick_align='right',
    visible_spines=['left'],             # No bottom spine (labels below)
    params=PLOT_PARAMS,
)

# -----------------------------------------------------------------------------
# Panel 1C: ROI groups legend
# -----------------------------------------------------------------------------
# Shows all ROI group families with their colors (horizontal layout)
# Positioned with zorder=-1 so it appears behind other plot elements
ax_roi_legend = plt.axes(); ax_roi_legend.set_label('ROI_Groups_Legend')
roi_legend_fig = create_roi_group_legend(
    single_row=True,                     # Horizontal layout
    params=PLOT_PARAMS
)
embed_figure_on_ax(ax_roi_legend, roi_legend_fig, title='')  # Zorder set centrally in legend

# Setup ax_dict for pylustrator
fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}

# Pylustrator layout code for this panel
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(cm_to_inches(8.90), cm_to_inches(8.74), forward=True)
plt.figure(1).ax_dict["Bars_Bottom_Diff_PR"].set(position=[0.1116, 0.2957, 0.867, 0.2766])
plt.figure(1).ax_dict["Bars_Bottom_Diff_PR"].texts[14].set(position=(0.5, 1.151))
plt.figure(1).ax_dict["Bars_Bottom_Diff_PR"].texts[15].set(position=(0.5, 1.065))
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].legend(loc=(0.6604, 0.9411), frameon=False, ncols=2)
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].set(position=[0.1116, 0.6717, 0.867, 0.2766])
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].texts[14].set(position=(0.5, 1.073))
plt.figure(1).ax_dict["Bars_Top_Mean_PR"].texts[15].set(position=(0.5, 0.9845))
plt.figure(1).ax_dict["ROI_Groups_Legend"].set(position=[-0.02044, -0.03719, 1.123, 0.1567])
#% end: automatic generated code from pylustrator

# Save this panel
save_axes_svgs(fig1, FIGURES_DIR, 'manifold_bars')
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'manifold_bars_panel.pdf')

logger.info("✓ Panel 1: Combined PR barplots complete")


# =============================================================================
# Panel 2: Matrix, loadings, PCA 2D, feature importance (matrix_pca)
# =============================================================================
# This panel shows the manifold analysis:
# - Panel A: PR matrix (subjects × ROIs) with experts on top
# - Panel B: PCA 2D projection with decision boundary
# - Panel C: Top-10 feature importance from classifier
# - Panel D: PCA component loadings (ROI contributions to PC1 and PC2)

# Data preparation specific to this panel
# Extract PR matrix and expert/novice split index
matrix = pr_matrix_pack['matrix']          # Shape: (n_subjects, n_rois)
n_experts = pr_matrix_pack['n_experts']    # Row index where experts end

# Extract PCA projection data
coords = pca2d['coords']        # Shape: (n_subjects, 2) - PC1 and PC2 coordinates
expl = pca2d['explained']       # Variance explained by PC1 and PC2
labels = pca2d['labels']        # Group labels (1 = expert, 0 = novice)
bnd = pca2d['boundary']         # Decision boundary mesh (xx, yy, Z)

# Extract PCA component loadings (shape: 2 × n_rois)
loadings = pca2d['components']  # Row 0 = PC1 weights, Row 1 = PC2 weights

# Map ROI IDs to pretty names for x-axis labels
roi_name_map = dict(zip(roi_info['roi_id'].values, roi_info['pretty_name'].values))
roi_pretty_names = [roi_name_map.get(int(lbl), str(lbl)) for lbl in roi_labels]
roi_pretty_names = [name.replace("\n", " ") for name in roi_pretty_names]

# Extract classifier coefficients (feature importance in original ROI space)
weights = classifier.coef_[0]  # Shape: (n_rois,) - one weight per ROI

# Build figure and axes
fig2 = plt.figure(2)

# -----------------------------------------------------------------------------
# Panel 2A: PR Matrix (Subjects × ROIs)
# -----------------------------------------------------------------------------
# Heatmap showing PR value for each subject (rows) across all ROIs (columns)
# Experts positioned in top rows, separated by horizontal line
ax_A = plt.axes(); ax_A.set_label('A_PR_Matrix')

plot_matrix_on_ax(
    ax=ax_A,
    matrix=matrix,                         # Subjects × ROIs PR values
    title='PR Profiles across Subjects',
    subtitle=None,
    cmap=CMAP_SEQUENTIAL,                  # Sequential colormap (not diverging)
    show_colorbar=False,                   # Colorbar shown separately
    xticklabels=None,                      # Too many ROIs to label
    yticklabels=None,                      # Too many subjects to label
    square=False,                          # Allow rectangular (not square cells)
    params=PLOT_PARAMS,
)
ax_A.set_xlabel('')
ax_A.set_ylabel('Subjects (Experts on top)', fontsize=PLOT_PARAMS['font_size_tick'])

# Draw horizontal line separating experts from novices
if 0 < n_experts < matrix.shape[0]:
    ax_A.axhline(n_experts, color='black', linewidth=PLOT_PARAMS['plot_linewidth'])

# -----------------------------------------------------------------------------
# Panel 2D: PCA Component Loadings (2 × n_rois)
# -----------------------------------------------------------------------------
# Shows how each ROI contributes to PC1 and PC2 (component weights)
# Positive values = ROI increases along that PC; negative = ROI decreases
ax_D = plt.axes(); ax_D.set_label('D_PCA_Loadings')

# Compute symmetric color range (centered at 0 for diverging colormap)
vmin, vmax = compute_ylim_range(loadings, symmetric=True, padding_pct=0.0)

plot_matrix_on_ax(
    ax=ax_D,
    matrix=loadings,                       # 2 rows (PC1, PC2) × n_rois columns
    title='ROI Contributions to PCA Components',
    subtitle=None,
    cmap=CMAP_BRAIN,                       # Diverging colormap (blue-white-red)
    vmin=vmin,                             # Symmetric range
    vmax=vmax,
    center=0,                              # Center colormap at 0
    show_colorbar=False,                   # Colorbar shown separately
    xticklabels=roi_pretty_names,          # Show ROI names on x-axis
    yticklabels=['PC 1', 'PC 2'],         # Component labels on y-axis
    square=False,
    params=PLOT_PARAMS,
)
ax_D.set_xlabel('')
# Gray out x-axis labels (ROI names shown below in other panels)
for ticklabel in ax_D.get_xticklabels():
    ticklabel.set_color('lightgray')

# -----------------------------------------------------------------------------
# Panel 2B: PCA 2D Projection with Decision Boundary
# -----------------------------------------------------------------------------
# Scatter plot of subjects in 2D PCA space, colored by group
# Shows classifier decision boundary (shaded regions)
ax_B = plt.axes(); ax_B.set_label('B_PCA_Projection')

# Compute axis limits with 5% padding
xlim, ylim = compute_limits_with_padding(coords, padding=0.05)

# Assign colors based on group (expert = blue, novice = orange)
point_colors = [COLORS_EXPERT_NOVICE['expert'] if lbl == 1 else COLORS_EXPERT_NOVICE['novice'] for lbl in labels]
point_alphas = [0.7] * len(labels)

# Format axis labels with variance explained (e.g., "PC 1 (45.2% var)")
x_label, y_label = pca_axis_labels_from_explained(expl)

plot_2d_embedding_on_ax(
    ax=ax_B,
    coords=coords,                         # Subject positions in PC1-PC2 space
    title='PCA Projection of PR Profiles',
    subtitle='',
    point_colors=point_colors,             # Expert/Novice colors
    point_alphas=point_alphas,
    xlim=xlim,
    ylim=ylim,
    x_label=x_label,                       # PC1 with % variance
    y_label=y_label,                       # PC2 with % variance
    fill={                                 # Decision boundary shading
        'xx': bnd['xx'],                   # Mesh grid x-coordinates
        'yy': bnd['yy'],                   # Mesh grid y-coordinates
        'Z': bnd['Z'],                     # Probability values (0 to 1)
        'colors': [COLORS_EXPERT_NOVICE['novice'], COLORS_EXPERT_NOVICE['expert']],
        'alpha': 0.15,                     # Transparent background
        'levels': [0, 0.5, 1],             # Contour levels (decision boundary at 0.5)
    },
    params=PLOT_PARAMS,
)

# -----------------------------------------------------------------------------
# Panel 2C: Feature Importance (Top 10 ROIs)
# -----------------------------------------------------------------------------
# Horizontal barplot showing top-10 ROI weights from logistic regression classifier
# Positive weights = ROI predicts Expert; Negative weights = ROI predicts Novice
ax_C = plt.axes(); ax_C.set_label('C_Feature_Importance')

# Use module-level helper to plot top-10 ROIs by absolute weight
plot_topn_feature_importance_on_ax(ax_C, weights, roi_info, top_n=10, params=PLOT_PARAMS,
                                   title='Top 10 Contributions to Classification')

# Setup ax_dict for pylustrator
fig2.ax_dict = {ax.get_label(): ax for ax in fig2.axes}

# Pylustrator layout code for this panel
#% start: automatic generated code from pylustrator
plt.figure(2).ax_dict = {ax.get_label(): ax for ax in plt.figure(2).axes}
getattr(plt.figure(2), '_pylustrator_init', lambda: ...)()
plt.figure(2).set_size_inches(cm_to_inches(18.29), cm_to_inches(13.31), forward=True)
plt.figure(2).ax_dict["A_PR_Matrix"].set(position=[0.1148, 0.3461, 0.3517, 0.6051])
plt.figure(2).ax_dict["A_PR_Matrix"].text(-0.0313, 1.0304, 'a', transform=plt.figure(2).ax_dict["A_PR_Matrix"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(2).ax_dict["A_PR_Matrix"].texts[0].new
plt.figure(2).ax_dict["B_PCA_Projection"].set(position=[0.5306, 0.6619, 0.288, 0.2465])
plt.figure(2).ax_dict["B_PCA_Projection"].text(-0.0477, 1.1114, 'b', transform=plt.figure(2).ax_dict["B_PCA_Projection"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(2).ax_dict["B_PCA_Projection"].texts[0].new
plt.figure(2).ax_dict["C_Feature_Importance"].set(position=[0.5311, 0.2863, 0.288, 0.2755], xlim=(-1.1, 1.1))
plt.figure(2).ax_dict["C_Feature_Importance"].spines[['right', 'top']].set_visible(False)
plt.figure(2).ax_dict["C_Feature_Importance"].texts[0].set(position=(0.1742, 1.027), fontsize=6.)
plt.figure(2).ax_dict["C_Feature_Importance"].texts[1].set(position=(0.8235, 1.027), fontsize=6.)
plt.figure(2).ax_dict["C_Feature_Importance"].text(-0.0477, 1.1218, 'c', transform=plt.figure(2).ax_dict["C_Feature_Importance"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(2).ax_dict["C_Feature_Importance"].texts[2].new
plt.figure(2).ax_dict["D_PCA_Loadings"].set(position=[0.1148, 0.2468, 0.3517, 0.03663])
#% end: automatic generated code from pylustrator

# Save this panel
save_axes_svgs(fig2, FIGURES_DIR, 'manifold_matrix_pca')
save_panel_pdf(fig2, FIGURES_DIR / 'panels' / 'manifold_matrix_pca_panel.pdf')

logger.info("✓ Panel 2: Matrix, loadings, PCA projection, feature importance complete")


# =============================================================================
# Panel 3: PR vs Voxel size scatters (pr_voxels)
# =============================================================================
# This panel shows the relationship between ROI size (voxel count) and PR:
# - Panel E: Experts - scatter of PR vs voxel count with regression line
# - Panel F: Novices - scatter of PR vs voxel count with regression line
# - Panel G: Differences - scatter of ΔPR vs average voxel count with regression line
# Each point = one ROI, colored by ROI group

# Data preparation specific to this panel
# Extract voxel correlation data (computed in 01_manifold_analysis.py)
# - group_avg: per-ROI, per-group average PR and voxel counts
# - diff_data: per-ROI differences (Expert - Novice PR) and average voxel counts
# - stats: correlation statistics (slope, intercept, r, p) for each group
voxel_corr = results.get('voxel_corr', {})
group_avg = voxel_corr.get('group_avg')      # DataFrame: ROI_Label, group, PR, n_voxels, color
diff_data = voxel_corr.get('diff_data')      # DataFrame: ROI_Label, PR_diff, n_voxels_avg, color

# Build figure and axes
fig3 = plt.figure(3)

# Only create scatter plots if voxel correlation data exists
if group_avg is not None and diff_data is not None:
    stats_vox = voxel_corr.get('stats', {})  # Dict: 'expert', 'novice', 'diff' → regression stats

    # -------------------------------------------------------------------------
    # Panel 3E: Experts - PR vs ROI size
    # -------------------------------------------------------------------------
    # Scatter plot showing relationship between ROI voxel count and mean PR
    # Each point = one ROI; color = ROI group
    ax_E = plt.axes(); ax_E.set_label('E_PR_vs_Voxels_Experts')

    # Filter group_avg to experts only
    ga_exp = group_avg[group_avg['group'] == 'expert']
    ms = PLOT_PARAMS.get('marker_size', 12.0)

    # Scatter: x = voxel count, y = mean PR, color = ROI group
    ax_E.scatter(
        ga_exp['n_voxels'], ga_exp['PR'],
        c=ga_exp['color'], s=ms, marker='o',
        alpha=PLOT_PARAMS.get('marker_alpha', 0.8),
        edgecolors='black', linewidths=PLOT_PARAMS['plot_linewidth'], zorder=2
    )
    ax_E.set_xlabel('ROI voxel count', fontsize=PLOT_PARAMS['font_size_label'])
    ax_E.set_ylabel('Mean PR', fontsize=PLOT_PARAMS['font_size_label'])
    ax_E.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
    style_spines(ax_E, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    set_axis_title(ax_E, title='PR vs ROI size', subtitle='Experts')

    # Draw regression line with r and p annotation (if stats available)
    stE = stats_vox.get('expert', {})
    if stE:
        draw_regression_line(
            ax_E, ga_exp['n_voxels'].values,
            stE.get('slope', 0.0), stE.get('intercept', 0.0),
            stE.get('r', None), stE.get('p', None), params=PLOT_PARAMS
        )

    # -------------------------------------------------------------------------
    # Panel 3F: Novices - PR vs ROI size
    # -------------------------------------------------------------------------
    # Same as Panel E, but for novices
    ax_F = plt.axes(); ax_F.set_label('F_PR_vs_Voxels_Novices')

    # Filter group_avg to novices only
    ga_nov = group_avg[group_avg['group'] == 'novice']
    ms = PLOT_PARAMS.get('marker_size', 12.0)

    # Scatter: x = voxel count, y = mean PR, color = ROI group
    ax_F.scatter(
        ga_nov['n_voxels'], ga_nov['PR'],
        c=ga_nov['color'], s=ms, marker='o',
        alpha=PLOT_PARAMS.get('marker_alpha', 0.8),
        edgecolors='black', linewidths=PLOT_PARAMS['plot_linewidth'], zorder=2
    )
    ax_F.set_xlabel('ROI voxel count', fontsize=PLOT_PARAMS['font_size_label'])
    ax_F.set_ylabel('Mean PR', fontsize=PLOT_PARAMS['font_size_label'])
    ax_F.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
    style_spines(ax_F, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    set_axis_title(ax_F, title='PR vs ROI size', subtitle='Novices')

    # Draw regression line with r and p annotation (if stats available)
    stN = stats_vox.get('novice', {})
    if stN:
        draw_regression_line(
            ax_F, ga_nov['n_voxels'].values,
            stN.get('slope', 0.0), stN.get('intercept', 0.0),
            stN.get('r', None), stN.get('p', None), params=PLOT_PARAMS
        )

    # -------------------------------------------------------------------------
    # Panel 3G: Differences - ΔPR vs average ROI size
    # -------------------------------------------------------------------------
    # Scatter plot showing relationship between average voxel count and ΔPR
    # X-axis = average of expert and novice voxel counts per ROI
    # Y-axis = Expert PR - Novice PR (positive = experts have higher PR)
    ax_G = plt.axes(); ax_G.set_label('G_PRdiff_vs_VoxelsAvg')
    ms = PLOT_PARAMS.get('marker_size', 12.0)

    # Scatter: x = average voxel count, y = ΔPR, color = ROI group
    ax_G.scatter(
        diff_data['n_voxels_avg'], diff_data['PR_diff'],
        c=diff_data['color'], s=ms, marker='o',
        alpha=PLOT_PARAMS.get('marker_alpha', 0.8),
        edgecolors='black', linewidths=PLOT_PARAMS['plot_linewidth'], zorder=2
    )

    # Add horizontal line at y=0 (no difference between groups)
    ax_G.axhline(0, color='black', linewidth=PLOT_PARAMS['plot_linewidth'],
                 alpha=PLOT_PARAMS.get('line_alpha', 0.5))

    ax_G.set_xlabel('ROI voxel count (avg E/N)', fontsize=PLOT_PARAMS['font_size_label'])
    ax_G.set_ylabel('ΔPR (Expert − Novice)', fontsize=PLOT_PARAMS['font_size_label'])
    ax_G.tick_params(labelsize=PLOT_PARAMS['font_size_tick'])
    style_spines(ax_G, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    set_axis_title(ax_G, title='ΔPR vs ROI size', subtitle='')

    # Draw regression line with r and p annotation (if stats available)
    stD = stats_vox.get('diff', {})
    if stD:
        draw_regression_line(
            ax_G, diff_data['n_voxels_avg'].values,
            stD.get('slope', 0.0), stD.get('intercept', 0.0),
            stD.get('r', None), stD.get('p', None), params=PLOT_PARAMS
        )

# Setup ax_dict for pylustrator
fig3.ax_dict = {ax.get_label(): ax for ax in fig3.axes}

# Pylustrator layout code for this panel
#% start: automatic generated code from pylustrator
plt.figure(3).ax_dict = {ax.get_label(): ax for ax in plt.figure(3).axes}
getattr(plt.figure(3), '_pylustrator_init', lambda: ...)()
plt.figure(3).set_size_inches(cm_to_inches(18.30), cm_to_inches(5.15), forward=True)
plt.figure(3).ax_dict["E_PR_vs_Voxels_Experts"].set(position=[0.06032, 0.1735, 0.244, 0.7121])
plt.figure(3).ax_dict["E_PR_vs_Voxels_Experts"].texts[0].set(position=(0.5, 1.084))
plt.figure(3).ax_dict["E_PR_vs_Voxels_Experts"].texts[1].set(position=(0.5, 1.018))
plt.figure(3).ax_dict["F_PR_vs_Voxels_Novices"].set(position=[0.391, 0.1735, 0.244, 0.7121])
plt.figure(3).ax_dict["F_PR_vs_Voxels_Novices"].texts[0].set(position=(0.5, 1.084))
plt.figure(3).ax_dict["F_PR_vs_Voxels_Novices"].texts[1].set(position=(0.5, 1.018))
plt.figure(3).ax_dict["G_PRdiff_vs_VoxelsAvg"].set(position=[0.7217, 0.1735, 0.244, 0.7121])
#% end: automatic generated code from pylustrator

# Save this panel
save_axes_svgs(fig3, FIGURES_DIR, 'manifold_pr_voxels')
save_panel_pdf(fig3, FIGURES_DIR / 'panels' / 'manifold_pr_voxels_panel.pdf')

logger.info("✓ Panel 3: PR vs voxel size scatters complete")


# =============================================================================
# Show all figures for interactive editing
# =============================================================================
if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()

log_script_end(logger)
