#!/usr/bin/env python3
"""
Skill-Gradient Analysis — Supplementary Figure

Generates one combined 3x3 figure:
  Row 1 (a-c): Elo vs neural measures (experts only) — PR, RSA checkmate, RSA strategy
  Rows 2-3 (d-i): Move accuracy vs neural measures (all participants) — all 6 metrics
"""

import sys
from pathlib import Path

import pickle  # nosec B403 — trusted internal analysis outputs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# Enable repo root imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common import CONFIG
from common.script_utils import setup_script
from common.logging_utils import log_script_end
from utils import compute_subject_mean_pr, load_bids_tsvs
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

# Load enriched familiarisation data (produced by 01_skill_gradient.py)
fam_subj_file = results_dir / 'familiarisation_subject_enriched.csv'
fam_subj = pd.read_csv(fam_subj_file)

# Load participants for Elo (all 20 experts)
participants = pd.read_csv(CONFIG['BIDS_PARTICIPANTS'], sep='\t')
experts = participants[(participants['group'] == 'expert') & participants['rating'].notna()]
elo_map = dict(zip(experts['participant_id'], experts['rating']))
expert_ids = set(elo_map.keys())

logger.info(f"Loaded Elo correlations: {len(elo_all)} rows")
logger.info(f"Loaded familiarisation data: {len(fam_subj)} subjects")
logger.info(f"Experts with Elo: {len(expert_ids)}")

# PR: load from pickle (same source as analysis script)
pr_pkl = Path(CONFIG['REPO_ROOT']) / 'chess-manifold/results/manifold/pr_results.pkl'
with open(pr_pkl, 'rb') as f:  # nosec B301 — trusted internal pkl
    pr_data = pickle.load(f)  # nosec B301
pr_long = pr_data['pr_long_format']
expert_mean_pr = compute_subject_mean_pr(pr_long, subject_ids=expert_ids)

# RSA: load from BIDS derivatives
rsa_df = load_bids_tsvs(CONFIG['BIDS_MVPA_RSA'])
rsa_roi_cols = [c for c in rsa_df.columns if c not in ['subject', 'target']]
rsa_cm = rsa_df[(rsa_df['subject'].isin(expert_ids)) & (rsa_df['target'] == 'checkmate')].copy()
rsa_cm['rsa_checkmate'] = rsa_cm[rsa_roi_cols].astype(float).mean(axis=1)
rsa_st = rsa_df[(rsa_df['subject'].isin(expert_ids)) & (rsa_df['target'] == 'strategy')].copy()
rsa_st['rsa_strategy'] = rsa_st[rsa_roi_cols].astype(float).mean(axis=1)
expert_rsa = rsa_cm[['subject', 'rsa_checkmate']].rename(columns={'subject': 'participant_id'})
expert_rsa = expert_rsa.merge(
    rsa_st[['subject', 'rsa_strategy']].rename(columns={'subject': 'participant_id'}),
    on='participant_id', how='outer',
)

# Combine into expert_subj_all (all 20 experts)
expert_subj_all = expert_mean_pr.merge(expert_rsa, on='participant_id', how='outer')
expert_subj_all['elo'] = expert_subj_all['participant_id'].map(elo_map)
expert_subj_all = expert_subj_all[expert_subj_all['elo'].notna()]
logger.info(f"Expert Elo-row data: {len(expert_subj_all)} subjects")


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
