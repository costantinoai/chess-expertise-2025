#!/usr/bin/env python3
"""
Board Preference Feature Drivers — Supplementary Figure
========================================================

Single panel (2 rows x 4 cols):
  Row 1: Experts — top-3 most preferred boards, key feature scatter (checkmate)
  Row 2: Novices — top-3 most preferred boards, key feature scatter (officers)

Board titles use full nomenclature: S{id} C/NC SY{strategy} P{visual}.
Board borders colored by check status. Scatter highlights top-3 and bottom-3
at full alpha; remaining boards at half alpha.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

from common import CONFIG
from common.script_utils import setup_script
from common.logging_utils import log_script_end
from common.plotting import (
    apply_nature_rc,
    PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    COLORS_CHECKMATE_NONCHECKMATE,
    figure_size,
    save_panel_pdf,
    style_spines,
)
from common.bids_utils import load_stimulus_metadata

# ============================================================================
# Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='novice_diagnostics',
    output_subdirs=['figures'],
    log_name='92_plot_preference_features.log',
)
figures_dir = dirs['figures']
apply_nature_rc()

PP = PLOT_PARAMS
EXP_COL = COLORS_EXPERT_NOVICE['expert']
NOV_COL = COLORS_EXPERT_NOVICE['novice']
C_CM = COLORS_CHECKMATE_NONCHECKMATE['checkmate']
C_NC = COLORS_CHECKMATE_NONCHECKMATE['non_checkmate']
STIMULI_DIR = CONFIG['EXTERNAL_DATA_ROOT'] / "stimuli"

# ============================================================================
# Load data
# ============================================================================

board_group = pd.read_csv(results_dir / 'board_preference_group.csv')
stim_df = load_stimulus_metadata(return_all=True)
ck = 'check' if 'check' in stim_df.columns else 'check_status'
feat_df = pd.read_csv(results_dir / 'feature_matrix.csv')

def get_ranked(group):
    gdf = board_group[board_group['group'] == group]
    freqs = {}
    for _, r in gdf.iterrows():
        freqs[int(r['c_stim_id'])] = r['c_freq']
        freqs[int(r['nc_stim_id'])] = r['nc_freq']
    bdf = stim_df.copy()
    bdf['pref_freq'] = bdf['stim_id'].map(freqs)
    return bdf.sort_values('pref_freq', ascending=False).reset_index(drop=True)

ranked = {g: get_ranked(g) for g in ['expert', 'novice']}


def board_label(row):
    """Full nomenclature: S{id} C/NC SY{strategy} P{visual}."""
    sid = int(row['stim_id'])
    tag = 'C' if row[ck] == 'checkmate' else 'NC'
    sy = int(row['strategy']) if pd.notna(row.get('strategy')) else '?'
    vis = int(row['visual']) if pd.notna(row.get('visual')) else '?'
    return f"S{sid} {tag} SY{sy} P{vis}"


# ============================================================================
# Figure
# ============================================================================

logger.info("Plotting preference features panel...")

fig = plt.figure(figsize=(18 / 2.54, 11 / 2.54))

# 4 sub-rows (2 per group) x 4 cols; col 3 = scatter spanning 2 sub-rows
gs = GridSpec(
    4, 4,
    figure=fig,
    width_ratios=[1, 1, 1, 1.8],
    height_ratios=[1, 1, 1, 1],
    hspace=0.08, wspace=0.08,
    left=0.06, right=0.97, top=0.92, bottom=0.08,
)

# First pass: collect all pref values to set shared ylim for scatters
all_pref = np.concatenate([feat_df['pref_expert'].dropna().values,
                           feat_df['pref_novice'].dropna().values])
shared_ylim = (max(0, np.floor(all_pref.min() * 20) / 20 - 0.02),
               np.ceil(all_pref.max() * 20) / 20 + 0.02)

scatter_axes = []
first_board_axes = []  # track first board axis per group for label placement

for g_idx, (group, group_label, key_feat, key_label) in enumerate([
    ('expert', 'Experts', 'is_checkmate', 'Checkmate status (0/1)'),
    ('novice', 'Novices', 'officer_count', 'Officer count (N+B+R+Q)'),
]):
    bdf = ranked[group]
    top3 = bdf.head(3)
    bot3 = bdf.tail(3)
    r_top = g_idx * 2
    r_bot = g_idx * 2 + 1

    # --- Top 3 boards (sub-row 0, cols 0-2) ---
    for i, (_, row) in enumerate(top3.iterrows()):
        ax = fig.add_subplot(gs[r_top, i])
        if i == 0:
            first_board_axes.append(ax)
        img_path = STIMULI_DIR / row['filename']
        is_cm = row[ck] == 'checkmate'
        if img_path.exists():
            ax.imshow(plt.imread(str(img_path)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color(C_CM if is_cm else C_NC)
        lbl = board_label(row)
        ax.set_title(f'{lbl}  f={row["pref_freq"]:.2f}',
                     fontsize=PP['font_size_tick'] - 1, color='black', pad=1)
        if i == 0:
            ax.set_ylabel('Top 3', fontsize=PP['font_size_tick'], rotation=0,
                          labelpad=18, va='center')

    # --- Bottom 3 boards (sub-row 1, cols 0-2) ---
    for i, (_, row) in enumerate(bot3.iterrows()):
        ax = fig.add_subplot(gs[r_bot, i])
        img_path = STIMULI_DIR / row['filename']
        is_cm = row[ck] == 'checkmate'
        if img_path.exists():
            ax.imshow(plt.imread(str(img_path)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color(C_CM if is_cm else C_NC)
        lbl = board_label(row)
        ax.set_title(f'{lbl}  f={row["pref_freq"]:.2f}',
                     fontsize=PP['font_size_tick'] - 1, color='black', pad=1)
        if i == 0:
            ax.set_ylabel('Bot 3', fontsize=PP['font_size_tick'], rotation=0,
                          labelpad=18, va='center')

    # --- Key scatter (col 3, spans both sub-rows) ---
    ax_sc = fig.add_subplot(gs[r_top:r_bot + 1, 3])
    scatter_axes.append(ax_sc)
    pref_col = f'pref_{group}'
    valid = feat_df.dropna(subset=[key_feat, pref_col]).copy()
    cm_mask = valid['is_checkmate'] == 1

    ax_sc.scatter(valid.loc[cm_mask, key_feat], valid.loc[cm_mask, pref_col],
                  c=C_CM, s=PP['marker_size'] * 1.5, alpha=0.7,
                  edgecolors='white', linewidths=0.3, zorder=3, label='Checkmate')
    ax_sc.scatter(valid.loc[~cm_mask, key_feat], valid.loc[~cm_mask, pref_col],
                  c=C_NC, s=PP['marker_size'] * 1.5, alpha=0.7,
                  edgecolors='white', linewidths=0.3, zorder=3, label='Non-checkmate')

    r_val, p_val = stats.spearmanr(valid[key_feat], valid[pref_col])
    z = np.polyfit(valid[key_feat], valid[pref_col], 1)
    x_line = np.linspace(valid[key_feat].min(), valid[key_feat].max(), 50)
    ax_sc.plot(x_line, np.polyval(z, x_line), '--', color='grey',
               linewidth=PP['reference_line_width'])

    star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    p_str = 'p < .001' if p_val < 0.001 else f'p = {p_val:.3f}'
    ax_sc.text(0.04, 0.96, f'r = {r_val:.2f}, {p_str}{" " + star if star else ""}',
               transform=ax_sc.transAxes, ha='left', va='top',
               fontsize=PP['font_size_tick'], color='black')

    ax_sc.set_xlabel(key_label, fontsize=PP['font_size_label'])
    ax_sc.set_ylabel('Selection frequency', fontsize=PP['font_size_label'])
    ax_sc.set_ylim(shared_ylim)
    ax_sc.axhline(0.5, color='grey', linestyle=':',
                  linewidth=PP['reference_line_width'],
                  alpha=PP['comparison_line_alpha'])
    ax_sc.legend(fontsize=PP['font_size_legend'], loc='lower right', frameon=False)
    style_spines(ax_sc)

# Suptitle — placed well above the top row
fig.suptitle('Board preference feature drivers', fontsize=PP['font_size_title'],
             fontweight='bold', y=1.04)

# Force layout before reading positions
fig.canvas.draw()

# Panel labels and group titles using tracked axes
letter = 0
for g_idx, group_label in enumerate(['Experts', 'Novices']):
    # (a/c) label + group title above top-left board
    pos = first_board_axes[g_idx].get_position()
    fig.text(pos.x0 - 0.04, pos.y1 + 0.04, chr(ord('a') + letter),
             fontsize=PP['font_size_panel_label'], fontweight='bold',
             va='bottom', transform=fig.transFigure)
    fig.text(pos.x0 + 0.02, pos.y1 + 0.04, group_label,
             fontsize=PP['font_size_title'], fontweight='bold',
             color='black', va='bottom', transform=fig.transFigure)
    letter += 1

    # (b/d) label above scatter
    pos_sc = scatter_axes[g_idx].get_position()
    fig.text(pos_sc.x0 - 0.04, pos_sc.y1 + 0.04, chr(ord('a') + letter),
             fontsize=PP['font_size_panel_label'], fontweight='bold',
             va='bottom', transform=fig.transFigure)
    letter += 1

# Save
save_panel_pdf(fig, figures_dir / 'panels' / 'preference_features_panel.pdf')
fig.savefig(figures_dir / 'preference_features_panel.svg', format='svg', bbox_inches='tight')
logger.info("Saved preference features panel")
plt.close(fig)

log_script_end(logger)
