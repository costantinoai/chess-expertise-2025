#!/usr/bin/env python3
"""
Skill-Gradient Analysis — Supplementary Figure

Generates one combined 3x3 figure:
  Row 1 (a-c): Elo vs neural measures (experts only) — PR, RSA checkmate, RSA strategy
  Rows 2-3 (d-i): Move accuracy vs neural measures (all participants) — all 6 metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# Enable repo root imports
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
    results_pattern='skill_gradient',
    output_subdirs=['figures'],
    log_name='91_plot_skill_gradient.log',
)
figures_dir = dirs['figures']
apply_nature_rc()

EXPERT_COLOR = COLORS_EXPERT_NOVICE['expert']
NOVICE_COLOR = COLORS_EXPERT_NOVICE['novice']
PP = PLOT_PARAMS
RNG = np.random.default_rng(CONFIG['RANDOM_SEED'])


# ============================================================================
# Load data
# ============================================================================

elo_all = pd.read_csv(results_dir / 'elo_correlations_all.csv')

# Load enriched per-subject table from BIDS derivatives
enriched_file = Path(CONFIG['BIDS_SKILL_GRADIENT']) / 'familiarisation_subject_enriched.csv'
anon_df = pd.read_csv(enriched_file)
if 'participant_id' in anon_df.columns:
    anon_df = anon_df.drop(columns=['participant_id'])

logger.info(f"Loaded Elo correlations: {len(elo_all)} rows")
logger.info(f"Loaded subject data from derivatives: {len(anon_df)} subjects")

# Expert subset for Elo scatter plots (row 1)
expert_subj_all = anon_df[
    (anon_df['group'] == 'expert') & anon_df['rating'].notna()
].copy()
expert_subj_all = expert_subj_all.rename(columns={'rating': 'elo'})
logger.info(f"Experts with Elo: {len(expert_subj_all)}")

# All-participant data for familiarisation scatter plots (rows 2-3)
fam_subj = anon_df


# ============================================================================
# Plotting helper — matched to familiarisation 91_plot style
# ============================================================================

def plot_correlation(ax, x_arr, y_arr, xlabel, ylabel, title,
                     group_labels=None, group_colors=None, single_color=None):
    """Scatter with BLACK regression line + grey CI band.

    Matches the familiarisation correlation style exactly:
    - Black regression line (linewidth = base * 2)
    - Black CI band (alpha = error_band_alpha)
    - White-edged markers
    - r/p annotation bottom-right with significance stars
    """
    valid = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    xv, yv = x_arr[valid], y_arr[valid]

    # Points with white edges
    if group_labels is not None and group_colors is not None:
        gl = group_labels[valid] if hasattr(group_labels, '__getitem__') else group_labels.values[valid]
        for g, gc in group_colors.items():
            mask = gl == g
            if mask.any():
                ax.scatter(xv[mask], yv[mask], color=gc,
                           s=PP['marker_size'], alpha=PP['scatter_alpha'],
                           edgecolors='white', linewidths=0.3, zorder=3)
    elif single_color is not None:
        ax.scatter(xv, yv, color=single_color,
                   s=PP['marker_size'], alpha=PP['scatter_alpha'],
                   edgecolors='white', linewidths=0.3, zorder=3)
    else:
        ax.scatter(xv, yv, color='black',
                   s=PP['marker_size'], alpha=PP['scatter_alpha'],
                   edgecolors='white', linewidths=0.3, zorder=3)

    if len(xv) >= 5:
        slope, intercept, r, p, se = stats.linregress(xv, yv)
        x_sorted = np.linspace(xv.min(), xv.max(), 100)
        y_pred = intercept + slope * x_sorted

        # 95% CI band
        n = len(xv)
        x_mean = xv.mean()
        ss_x = np.sum((xv - x_mean) ** 2)
        residuals = yv - (intercept + slope * xv)
        mse = np.sum(residuals ** 2) / (n - 2)
        se_line = np.sqrt(mse * (1.0 / n + (x_sorted - x_mean) ** 2 / ss_x))
        t_crit = stats.t.ppf(0.975, n - 2)

        # BLACK line and band
        ax.fill_between(x_sorted, y_pred - t_crit * se_line,
                        y_pred + t_crit * se_line,
                        color='grey', alpha=0.15, zorder=1)
        ax.plot(x_sorted, y_pred, color='black',
                linewidth=PP['base_linewidth'] * 2, zorder=2)

        # Bottom-right annotation with significance stars
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(0.98, 0.04, f'r={r:.2f}, p={p:.3f}{sig}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=PP['font_size_tick'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    style_spines(ax)


def add_group_legend(ax, loc='upper right'):
    """Expert/novice legend matching familiarisation style."""
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=EXPERT_COLOR,
               markersize=4, label='Expert'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NOVICE_COLOR,
               markersize=4, label='Novice'),
    ]
    ax.legend(handles=handles, loc=loc, frameon=False,
              fontsize=PP['font_size_legend'])


# ============================================================================
# Combined figure: Elo (row 1, experts) + Move accuracy (rows 2-3, all), 3x3
# ============================================================================

logger.info("Plotting combined skill-gradient figure (3x3)")

# Familiarisation data for rows 2-3 (all participants)
fam_valid = fam_subj[fam_subj['move_acc_all_cm'].notna()].copy()
group_labels = np.array(fam_valid['group'].values)
group_colors = {'expert': EXPERT_COLOR, 'novice': NOVICE_COLOR}

figsize = figure_size(columns=2, height_mm=220)
fig, axes = plt.subplots(3, 3, figsize=figsize, squeeze=False)

# --- Row 1: Elo correlations (experts only, 3 key metrics) ---
elo_metrics = [
    ('mean_pr', 'Mean PR', 'Elo vs PR'),
    ('rsa_checkmate', 'RSA checkmate (r)', 'Elo vs RSA checkmate'),
    ('rsa_strategy', 'RSA strategy (r)', 'Elo vs RSA strategy'),
]

for j, (col, ylabel, title) in enumerate(elo_metrics):
    plot_correlation(
        axes[0, j],
        expert_subj_all['elo'].values, expert_subj_all[col].values,
        xlabel='Elo rating', ylabel=ylabel,
        title=title, single_color=EXPERT_COLOR,
    )

# --- Rows 2-3: Move accuracy vs neural metrics (all participants) ---
neural_metrics = [
    ('mean_pr', 'Participation ratio', 'Move acc. vs PR'),
    ('rsa_checkmate', 'RSA checkmate (r)', 'Move acc. vs RSA checkmate'),
    ('rsa_strategy', 'RSA strategy (r)', 'Move acc. vs RSA strategy'),
    ('rsa_visual_similarity', 'RSA visual sim. (r)', 'Move acc. vs RSA visual sim.'),
    ('dec_checkmate', 'Decoding checkmate', 'Move acc. vs Dec. checkmate'),
    ('dec_strategy', 'Decoding strategy', 'Move acc. vs Dec. strategy'),
]

for i, (col, ylabel, title) in enumerate(neural_metrics):
    row, c = divmod(i, 3)
    ax = axes[row + 1, c]
    if col in fam_valid.columns:
        plot_correlation(
            ax,
            fam_valid['move_acc_all_cm'].values,
            fam_valid[col].values,
            xlabel='Move accuracy', ylabel=ylabel, title=title,
            group_labels=group_labels, group_colors=group_colors,
        )

# Panel labels (a-i)
for i, ax in enumerate(axes.flat):
    ax.text(-0.12, 1.06, chr(ord('a') + i), transform=ax.transAxes,
            fontsize=PP['font_size_panel_label'], fontweight='bold',
            va='top', ha='left')

# Layout
fig.subplots_adjust(hspace=0.55)

# Row suptitles — use axes transforms so position is robust to bbox_inches='tight'
axes[0, 1].text(0.5, 1.30, 'Elo rating (experts only, n = 20)',
                transform=axes[0, 1].transAxes, ha='center', va='bottom',
                fontsize=PP['font_size_title'], fontweight='bold')
axes[1, 1].text(0.5, 1.30, 'Familiarisation move accuracy (all participants, n = 38)',
                transform=axes[1, 1].transAxes, ha='center', va='bottom',
                fontsize=PP['font_size_title'], fontweight='bold')

# Legend in first panel of row 2
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=EXPERT_COLOR,
           markersize=4, label='Expert'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=NOVICE_COLOR,
           markersize=4, label='Novice'),
]
axes[1, 0].legend(handles=handles, loc='upper right', frameon=False,
                  fontsize=PP['font_size_legend'])

save_panel_pdf(fig, figures_dir / 'skill_gradient_panel.pdf')
fig.savefig(figures_dir / 'skill_gradient_panel.svg',
            format='svg', bbox_inches='tight')
logger.info("Saved skill_gradient_panel")
plt.close(fig)


# ============================================================================

log_script_end(logger)
