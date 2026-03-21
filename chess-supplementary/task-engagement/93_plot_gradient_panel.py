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
- gradient_bivariate_correlations.csv
- gradient_partial_correlations.csv
- gradient_variance_partitioning.csv
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from common import CONFIG
from common.script_utils import setup_script
from common.logging_utils import log_script_end
from common.plotting import (
    apply_nature_rc,
    PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    figure_size,
    save_panel_pdf,
    style_spines,
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

# Feature order (must match 05_perceptual_relational_gradient.py)
FEATURE_ORDER = [
    ('image_entropy',     'Image\nentropy'),
    ('edge_density',      'Edge\ndensity'),
    ('piece_count',       'Piece\ncount'),
    ('officer_count',     'Officer\ncount'),
    ('center_occupation', 'Center\noccup.'),
    ('king_exposure',     'King\nexposure'),
    ('attack_coverage',   'Attack\ncoverage'),
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

biv_df = pd.read_csv(results_dir / 'gradient_bivariate_correlations.csv')
part_df = pd.read_csv(results_dir / 'gradient_partial_correlations.csv')
vp_df = pd.read_csv(results_dir / 'gradient_variance_partitioning.csv')

logger.info("Loaded gradient analysis results")

# ============================================================================
# Figure
# ============================================================================

logger.info("Plotting gradient panel...")

fig = plt.figure(figsize=figure_size(columns=2, height_mm=110))

gs = GridSpec(2, 3, figure=fig,
             width_ratios=[2.5, 2.5, 1],
             height_ratios=[1, 1],
             hspace=0.50, wspace=0.30,
             left=0.09, right=0.97, top=0.88, bottom=0.14)

n_feat = len(FEATURE_ORDER)
x = np.arange(n_feat)
bar_w = 0.35

# --- Top-left and bottom-left: Bivariate vs Partial correlations ---

for g_idx, (group, group_color, group_label) in enumerate([
    ('expert', EXP_COL, 'Experts'),
    ('novice', NOV_COL, 'Novices'),
]):
    ax = fig.add_subplot(gs[g_idx, 0:2])

    # Bivariate r
    gdf_biv = biv_df[biv_df['group'] == group].set_index('feature')
    r_biv = [gdf_biv.loc[fk, 'spearman_r'] if fk in gdf_biv.index else 0
             for fk, _ in FEATURE_ORDER]
    sig_biv = [gdf_biv.loc[fk, 'significant_fdr'] if fk in gdf_biv.index else False
               for fk, _ in FEATURE_ORDER]

    # Partial r
    gdf_part = part_df[part_df['group'] == group].set_index('feature')
    r_part = [gdf_part.loc[fk, 'partial_r'] if fk in gdf_part.index else 0
              for fk, _ in FEATURE_ORDER]
    sig_part = [bool(gdf_part.loc[fk, 'significant_fdr']) if fk in gdf_part.index else False
                for fk, _ in FEATURE_ORDER]

    # Bivariate bars (solid)
    colors_biv = [group_color if s else '#CCCCCC' for s in sig_biv]
    ax.bar(x - bar_w / 2, r_biv, bar_w, color=colors_biv, edgecolor='none',
           alpha=PP['bar_alpha'], label='Bivariate')

    # Partial bars (hatched)
    colors_part = [group_color if s else '#CCCCCC' for s in sig_part]
    bars_part = ax.bar(x + bar_w / 2, r_part, bar_w, color=colors_part,
                       edgecolor=group_color, linewidth=0.5,
                       alpha=PP['bar_alpha'] * 0.6, label='Partial')
    for bar in bars_part:
        bar.set_hatch('///')

    # Significance stars
    for i, (rv, sig) in enumerate(zip(r_biv, sig_biv)):
        if sig:
            y_pos = rv + 0.03
            ax.text(i - bar_w / 2, y_pos, '*', ha='center',
                    fontsize=PP['font_size_annotation'], fontweight='bold',
                    color=group_color)
    for i, (rv, sig) in enumerate(zip(r_part, sig_part)):
        if sig:
            y_pos = rv + 0.03
            ax.text(i + bar_w / 2, y_pos, '*', ha='center',
                    fontsize=PP['font_size_annotation'], fontweight='bold',
                    color=group_color)

    feat_labels = [fl for _, fl in FEATURE_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=PP['font_size_tick'] - 1)
    ax.set_ylabel('Spearman r', fontsize=PP['font_size_label'])
    ax.set_title(group_label, fontsize=PP['font_size_title'], fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylim(-0.4, 1.05)
    ax.legend(fontsize=PP['font_size_legend'], loc='upper left', frameon=False)
    style_spines(ax)

    # Block spans (light background shading)
    block_spans = [(0, 1.5, BLOCK_COLORS['Perceptual']),
                   (1.5, 4.5, BLOCK_COLORS['Structural']),
                   (4.5, 7.5, BLOCK_COLORS['Strategic-Relational'])]
    for x0, x1, col in block_spans:
        ax.axvspan(x0 - 0.5, x1, alpha=0.04, color=col, zorder=0)

# --- Right column: Variance partitioning stacked bars ---

for g_idx, (group, group_label) in enumerate([('expert', 'Experts'), ('novice', 'Novices')]):
    ax = fig.add_subplot(gs[g_idx, 2])
    gdf_vp = vp_df[vp_df['group'] == group]

    bottom = 0
    for _, row in gdf_vp.iterrows():
        color = BLOCK_COLORS[row['block']]
        ax.bar(0, row['delta_r2'], bottom=bottom, color=color,
               edgecolor='white', linewidth=0.5, width=0.5,
               alpha=PP['bar_alpha'])
        if row['delta_r2'] > 0.03:
            ax.text(0, bottom + row['delta_r2'] / 2,
                    f".{int(row['delta_r2']*100):02d}",
                    ha='center', va='center', fontsize=PP['font_size_tick'],
                    color='white', fontweight='bold')
        bottom += row['delta_r2']

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel('R²', fontsize=PP['font_size_label'])
    ax.set_title(f'Variance\n({group_label})',
                 fontsize=PP['font_size_tick'], fontweight='bold')
    ax.set_ylim(0, max(1.0, bottom + 0.05))
    style_spines(ax)

# Block legend at bottom
legend_handles = [Patch(facecolor=c, label=n, alpha=PP['bar_alpha'])
                  for n, c in BLOCK_COLORS.items()]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False,
           fontsize=PP['font_size_legend'], bbox_to_anchor=(0.5, 0.0))

# Gradient arrow
fig.text(0.42, 0.08, 'Perceptual  ←―――――――→  Relational',
         ha='center', fontsize=PP['font_size_label'], style='italic',
         transform=fig.transFigure)

# Panel labels
for i, ax in enumerate(fig.axes):
    ax.text(-0.08, 1.08, chr(ord('a') + i), transform=ax.transAxes,
            fontsize=PP['font_size_panel_label'], fontweight='bold', va='top')

fig.suptitle('Perceptual-to-relational feature gradient',
             fontsize=PP['font_size_title'], fontweight='bold', y=0.96)

save_panel_pdf(fig, figures_dir / 'panels' / 'gradient_panel.pdf')
fig.savefig(figures_dir / 'gradient_panel.svg', format='svg', bbox_inches='tight')
logger.info("Saved gradient panel")
plt.close(fig)

log_script_end(logger)
