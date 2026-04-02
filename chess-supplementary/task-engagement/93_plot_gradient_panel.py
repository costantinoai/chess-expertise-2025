"""
Perceptual-to-Relational Gradient — Supplementary Figure
=========================================================

Produces a single panel showing:
  Row 1: Bivariate (solid) vs Partial (hatched) Spearman r for each feature,
         per group. Features ordered along the perceptual→relational gradient.
  Row 2: Variance partitioning stacked bars (Perceptual / Structural /
         Strategic-Relational blocks) per group.

Figures Produced
----------------
- figures/panels/gradient_panel.pdf
- figures/gradient_panel.svg

Inputs
------
- feature_correlations.csv
- feature_partial_correlations.csv
- feature_variance_partitioning.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

from common.script_utils import setup_script
from common.logging_utils import log_script_end
from common.plotting import (
    apply_nature_rc,
    PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    figure_size,
    save_panel_pdf,
    style_spines,
    cm_to_inches,
)

# ============================================================================
# Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='novice_diagnostics',
    output_subdirs=['figures'],
    log_name='93_plot_gradient_panel.log',
)
figures_dir = dirs['figures']
apply_nature_rc()

PP = PLOT_PARAMS
EXP_COL = COLORS_EXPERT_NOVICE['expert']
NOV_COL = COLORS_EXPERT_NOVICE['novice']

# Feature order (must match 04_quantify_preference_drivers.py)
FEATURE_ORDER = [
    ('image_entropy',     'Image\nentropy'),
    ('edge_density',      'Edge\ndensity'),
    ('piece_count',       'Piece\ncount'),
    ('officer_count',     'Officer\ncount'),
    ('center_occupation', 'Center\noccup.'),
    ('king_advantage',    'King\nadvantage'),
    ('attack_advantage',  'Attack\nadvantage'),
    ('is_checkmate',      'Checkmate\nstatus'),
]

BLOCK_COLORS = {
    'Perceptual': '#3498DB',
    'Structural': '#2ECC71',
    'Strategic-Relational': '#E74C3C',
}

# ============================================================================
# Load data
# ============================================================================

biv_df = pd.read_csv(results_dir / 'feature_correlations.csv')
part_df = pd.read_csv(results_dir / 'feature_partial_correlations.csv')
vp_df = pd.read_csv(results_dir / 'feature_variance_partitioning.csv')

logger.info("Loaded gradient analysis results")

# ============================================================================
# Figure
# ============================================================================

logger.info("Plotting gradient panel...")

fig = plt.figure(figsize=figure_size(columns=2, height_mm=95))

gs = GridSpec(2, 2, figure=fig,
             width_ratios=[3, 1],
             height_ratios=[1, 1],
             hspace=0.35, wspace=0.25,
             left=0.09, right=0.96, top=0.90, bottom=0.10)

n_feat = len(FEATURE_ORDER)
x = np.arange(n_feat)
bar_w = 0.28

# Block boundary positions (between features, in x-axis coords)
# Perceptual: features 0-1, Structural: 2-4, Strategic-Relational: 5-7
block_edges = [(-0.5, 1.5), (1.5, 4.5), (4.5, 7.5)]

# --- Left column: Bivariate vs Partial correlations ---

for g_idx, (group, group_color, group_label) in enumerate([
    ('expert', EXP_COL, 'Experts'),
    ('novice', NOV_COL, 'Novices'),
]):
    ax = fig.add_subplot(gs[g_idx, 0])

    # Bivariate r + CIs
    gdf_biv = biv_df[biv_df['group'] == group].set_index('feature')
    r_biv, ci_biv, sig_biv = [], [], []
    for fk, _ in FEATURE_ORDER:
        if fk in gdf_biv.index:
            r_biv.append(gdf_biv.loc[fk, 'spearman_r'])
            ci_biv.append((gdf_biv.loc[fk, 'ci_low'], gdf_biv.loc[fk, 'ci_high']))
            sig_biv.append(bool(gdf_biv.loc[fk, 'significant_fdr']))
        else:
            r_biv.append(0); ci_biv.append((0, 0)); sig_biv.append(False)

    # Partial r + CIs
    gdf_part = part_df[part_df['group'] == group].set_index('feature')
    r_part, ci_part, sig_part = [], [], []
    for fk, _ in FEATURE_ORDER:
        if fk in gdf_part.index:
            r_part.append(gdf_part.loc[fk, 'partial_r'])
            ci_part.append((gdf_part.loc[fk, 'ci_low'], gdf_part.loc[fk, 'ci_high']))
            sig_part.append(bool(gdf_part.loc[fk, 'significant_fdr']))
        else:
            r_part.append(0); ci_part.append((0, 0)); sig_part.append(False)

    # Bootstrap CI error bars (asymmetric: bar height = r, errors = CI bounds - r)
    biv_err_lo = [r - ci[0] for r, ci in zip(r_biv, ci_biv)]
    biv_err_hi = [ci[1] - r for r, ci in zip(r_biv, ci_biv)]
    part_err_lo = [r - ci[0] for r, ci in zip(r_part, ci_part)]
    part_err_hi = [ci[1] - r for r, ci in zip(r_part, ci_part)]

    # Block shading (behind bars, aligned to feature boundaries)
    for (x0, x1), (bname, bcol) in zip(block_edges, BLOCK_COLORS.items()):
        ax.axvspan(x0, x1, alpha=0.06, color=bcol, zorder=0)

    # Bivariate bars (solid) with CI error bars
    colors_biv = [group_color if s else '#CCCCCC' for s in sig_biv]
    ax.bar(x - bar_w / 2, r_biv, bar_w, color=colors_biv, edgecolor='none',
           alpha=PP['bar_alpha'], label='Bivariate', zorder=2)
    ax.errorbar(x - bar_w / 2, r_biv, yerr=[biv_err_lo, biv_err_hi],
                fmt='none', color='black', elinewidth=PP['errorbar_linewidth'],
                capsize=PP['errorbar_capsize'] * 0.6, zorder=3)

    # Partial bars (hatched) with CI error bars
    colors_part = [group_color if s else '#CCCCCC' for s in sig_part]
    bars_part = ax.bar(x + bar_w / 2, r_part, bar_w, color=colors_part,
                       edgecolor=group_color, linewidth=0.4,
                       alpha=PP['bar_alpha'] * 0.6, label='Partial', zorder=2)
    for bar in bars_part:
        bar.set_hatch('///')
    ax.errorbar(x + bar_w / 2, r_part, yerr=[part_err_lo, part_err_hi],
                fmt='none', color='black', elinewidth=PP['errorbar_linewidth'],
                capsize=PP['errorbar_capsize'] * 0.6, zorder=3)

    # Significance stars (above CI upper bound)
    for i, (rv, ci, sig) in enumerate(zip(r_biv, ci_biv, sig_biv)):
        if sig:
            ax.text(i - bar_w / 2, ci[1] + 0.02, '*', ha='center',
                    fontsize=PP['font_size_annotation'], fontweight='bold',
                    color=group_color, zorder=4)
    for i, (rv, ci, sig) in enumerate(zip(r_part, ci_part, sig_part)):
        if sig:
            ax.text(i + bar_w / 2, ci[1] + 0.02, '*', ha='center',
                    fontsize=PP['font_size_annotation'], fontweight='bold',
                    color=group_color, zorder=3)

    feat_labels = [fl for _, fl in FEATURE_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=PP['font_size_tick'] - 1)
    ax.set_ylabel('Spearman r', fontsize=PP['font_size_label'])
    ax.set_title(group_label, fontsize=PP['font_size_title'], fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.4, zorder=1)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlim(-0.6, n_feat - 0.4)

    # Explicit legend with group color (not gray from first non-sig bar)
    from matplotlib.patches import Patch as _P
    biv_patch = _P(facecolor=group_color, alpha=PP['bar_alpha'], label='Bivariate')
    part_patch = _P(facecolor=group_color, alpha=PP['bar_alpha'] * 0.6,
                    edgecolor=group_color, linewidth=0.4, label='Partial')
    part_patch.set_hatch('///')
    ax.legend(handles=[biv_patch, part_patch], fontsize=PP['font_size_legend'],
              loc='upper left', frameon=False)
    style_spines(ax)

# --- Right column: Variance partitioning stacked bars (side by side) ---

for g_idx, (group, group_label) in enumerate([('expert', 'Experts'), ('novice', 'Novices')]):
    ax = fig.add_subplot(gs[g_idx, 1])
    gdf_vp = vp_df[vp_df['group'] == group]

    bottom = 0
    for _, row in gdf_vp.iterrows():
        color = BLOCK_COLORS[row['block']]
        ax.bar(0, row['delta_r2'], bottom=bottom, color=color,
               edgecolor='white', linewidth=0.5, width=0.45,
               alpha=PP['bar_alpha'])
        if row['delta_r2'] > 0.03:
            ax.text(0, bottom + row['delta_r2'] / 2,
                    f"{row['delta_r2']:.2f}",
                    ha='center', va='center', fontsize=PP['font_size_tick'] - 0.5,
                    color='white', fontweight='bold')
        bottom += row['delta_r2']

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel('R²', fontsize=PP['font_size_label'])
    ax.set_title(f'Variance', fontsize=PP['font_size_tick'], fontweight='bold')
    ax.set_ylim(0, 1.05)
    style_spines(ax)

# Block legend between the two rows, right-aligned
legend_handles = [Patch(facecolor=c, label=n, alpha=PP['bar_alpha'])
                  for n, c in BLOCK_COLORS.items()]
fig.legend(handles=legend_handles, loc='lower right', ncol=1, frameon=False,
           fontsize=PP['font_size_legend'], bbox_to_anchor=(0.96, 0.42))

# Gradient arrow below x-axis of bottom-left panel
fig.text(0.45, 0.02, 'Perceptual  ───────→  Relational',
         ha='center', fontsize=PP['font_size_tick'], style='italic',
         color='grey', transform=fig.transFigure)

# Panel labels
for i, ax in enumerate(fig.axes):
    ax.text(-0.10, 1.08, chr(ord('a') + i), transform=ax.transAxes,
            fontsize=PP['font_size_panel_label'], fontweight='bold', va='top')

fig.suptitle('Perceptual-to-relational feature gradient',
             fontsize=PP['font_size_title'], fontweight='bold', y=0.97)

# Create ax_dict for pylustrator convenience
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}

# =============================================================================
# Pylustrator Auto-Generated Layout Code
# =============================================================================

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(15.090000/2.54, 8.780000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.06619, 0.5733, 0.5698, 0.303])
plt.figure(1).axes[0].set_position([0.080095, 0.575135, 0.685875, 0.325490])
plt.figure(1).axes[1].set(position=[0.08009, 0.06889, 0.6859, 0.3255])
plt.figure(1).axes[2].set(position=[0.6878, 0.5742, 0.07069, 0.3036])
plt.figure(1).axes[2].set_position([0.828283, 0.576099, 0.085085, 0.326134])
plt.figure(1).axes[2].texts[1].set(position=(-0.5599, 1.113))
plt.figure(1).axes[3].set(position=[0.8283, 0.06889, 0.08508, 0.3261])
plt.figure(1).axes[3].texts[3].set(position=(-0.5599, 1.108))
plt.figure(1).axes[3].title.set(visible=False)
plt.figure(1).texts[0].set(visible=False)
plt.figure(1).texts[0].set_position([0.543808, -0.015936])
plt.figure(1).texts[1].set(position=(0.5368, 0.9872))
#% end: automatic generated code from pylustrator

# Display in pylustrator GUI for interactive adjustment
if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()

# =============================================================================
# Save
# =============================================================================

save_panel_pdf(fig, figures_dir / 'panels' / 'gradient_panel.pdf')
fig.savefig(figures_dir / 'gradient_panel.svg', format='svg', bbox_inches='tight')
logger.info("Saved gradient panel")
plt.close(fig)

log_script_end(logger)
