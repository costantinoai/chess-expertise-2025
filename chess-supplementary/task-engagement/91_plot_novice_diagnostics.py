#!/usr/bin/env python3
"""
Task Engagement and Novice Diagnostics — Supplementary Figure

Generates one combined panel (2 rows):
  Row 1 (a-d): Response rate, Transitivity, Checkmate preference, Visual pairs consistency
  Row 2 (e-f): Board preference profiles — Experts, Novices
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# Enable repo root imports
from common import CONFIG
from common.script_utils import setup_script
from common.logging_utils import log_script_end
from common.stats_utils import compute_mean_ci_and_ttest_vs_value
from common.plotting import (
    apply_nature_rc,
    PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    figure_size,
    save_panel_pdf,
    style_spines,
    set_axis_title,
)


# ============================================================================
# Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='novice_diagnostics',
    output_subdirs=['figures'],
    log_name='91_plot_novice_diagnostics.log',
)
figures_dir = dirs['figures']
apply_nature_rc()

EXPERT_COLOR = COLORS_EXPERT_NOVICE['expert']
NOVICE_COLOR = COLORS_EXPERT_NOVICE['novice']
from common.plotting import COLORS_CHECKMATE_NONCHECKMATE
C_CM = COLORS_CHECKMATE_NONCHECKMATE['checkmate']
C_NC = COLORS_CHECKMATE_NONCHECKMATE['non_checkmate']
PP = PLOT_PARAMS
RNG = np.random.default_rng(CONFIG['RANDOM_SEED'])


# ============================================================================
# Load data
# ============================================================================

resp_rate = pd.read_csv(results_dir / 'response_rate.csv')
check_pref = pd.read_csv(results_dir / 'checkmate_preference.csv')
transitivity = pd.read_csv(results_dir / 'transitivity.csv')
board_profile = pd.read_csv(results_dir / 'board_preference_profile.csv')
board_group = pd.read_csv(results_dir / 'board_preference_group.csv')

logger.info(f"Loaded novice diagnostic results from {results_dir}")


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_group_comparison(ax, expert_vals, novice_vals, ylabel, title,
                          chance=None, ylim_top=1.12):
    """Grouped bar chart with individual points, CI, and significance.

    Style matches main manuscript bar plots (PLOT_PARAMS).
    """
    bar_width = 0.5
    x = [0, 1]
    groups = [expert_vals.dropna(), novice_vals.dropna()]
    colors = [EXPERT_COLOR, NOVICE_COLOR]
    labels = ['Exp.', 'Nov.']

    for i, (vals, color, label) in enumerate(zip(groups, colors, labels)):
        mean = vals.mean()
        sem = stats.sem(vals) if len(vals) > 1 else 0
        ci = sem * stats.t.ppf(0.975, len(vals) - 1) if len(vals) > 1 else 0

        ax.bar(x[i], mean, width=bar_width, color=color,
               alpha=PP['bar_alpha'], edgecolor=PP['bar_edgecolor'],
               linewidth=PP['bar_linewidth'], zorder=2, label=label)

        ax.errorbar(x[i], mean, yerr=ci, fmt='none', color='black',
                    elinewidth=PP['errorbar_linewidth'],
                    capsize=PP['errorbar_capsize'], zorder=3)

        jitter = RNG.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x[i] + jitter, vals, color=color,
                   s=PP['marker_size'] * 0.7,
                   alpha=PP['scatter_alpha'], edgecolors='white',
                   linewidths=0.2, zorder=4)

    if chance is not None:
        ax.axhline(y=chance, color='grey', linestyle='--',
                   linewidth=PP['reference_line_width'],
                   alpha=PP['comparison_line_alpha'], zorder=1)

    # Significance bracket between groups
    if len(groups[0]) > 1 and len(groups[1]) > 1:
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        y_bar = max(groups[0].max(), groups[1].max()) + 0.03
        ax.plot([0, 0, 1, 1], [y_bar - 0.01, y_bar, y_bar, y_bar - 0.01],
                color='black', linewidth=PP['base_linewidth'], zorder=5)
        ax.text(0.5, y_bar + 0.005, sig, ha='center', va='bottom',
                fontsize=PP['font_size_annotation'], fontweight='bold',
                color='black', zorder=6)

    # Within-group vs chance stars
    for i, vals in enumerate(groups):
        if len(vals) > 1 and chance is not None:
            mean, _, ci_high, t_ch, p_ch = compute_mean_ci_and_ttest_vs_value(vals, popmean=chance)
            if p_ch < 0.05:
                star = '***' if p_ch < 0.001 else '**' if p_ch < 0.01 else '*'
                ax.text(x[i], ci_high + 0.02, star, ha='center', va='bottom',
                        fontsize=PP['font_size_annotation'], fontweight='bold',
                        color='black', zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(bottom=None, top=ylim_top)
    ax.set_xlim(-0.7, 1.7)
    style_spines(ax)


def plot_board_preference(ax, board_group_df, group, color, title):
    """Board preference scatter for a single group with regression + CI.

    Stats annotation in top-left.
    """
    gdata = board_group_df[board_group_df['group'] == group]
    c_freq = gdata['c_freq'].values
    nc_freq = gdata['nc_freq'].values
    valid = ~np.isnan(c_freq) & ~np.isnan(nc_freq)
    xv, yv = c_freq[valid], nc_freq[valid]

    # Axis limits from full data range
    all_vals = np.concatenate([
        board_group_df['c_freq'].dropna().values,
        board_group_df['nc_freq'].dropna().values,
    ])
    data_min = max(0, all_vals.min() - 0.05)
    data_max = min(1, all_vals.max() + 0.05)

    # Identity line
    ax.plot([data_min, data_max], [data_min, data_max], color='grey',
            linestyle='--', linewidth=PP['reference_line_width'],
            alpha=PP['comparison_line_alpha'], zorder=0)

    # Points
    ax.scatter(xv, yv, color=color,
               s=PP['marker_size'] * 2, alpha=0.7,
               edgecolors='white', linewidths=0.3, zorder=3)

    # Correlation annotation (no regression line or CI band)
    if len(xv) >= 3:
        r, p = stats.pearsonr(xv, yv)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(0.04, 0.96, f'r={r:.2f}, p={p:.3f}{sig}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=PP['font_size_tick'], color='black')

    ax.set_xlabel('Checkmate freq.')
    ax.set_ylabel('Non-checkmate freq.')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)
    ax.set_aspect('equal')
    style_spines(ax)


# ============================================================================
# Combined panel (3x2)
# ============================================================================

logger.info("Plotting combined diagnostic panel (2 rows: 4 bars + 2 scatters)")

# Row 1: 4 bar plots (a-d) — use narrower bars for compact 4-across layout
# Row 2: 2 scatter plots (e-f) — centered, roughly square
figsize = figure_size(columns=2, height_mm=120)
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3], hspace=0.5, wspace=0.55)

# Row 1: 4 bar plots
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[0, 3])

# Row 2: 2 scatter plots centred (span middle columns)
ax_e = fig.add_subplot(gs[1, 0:2])
ax_f = fig.add_subplot(gs[1, 2:4])

bar_axes = [ax_a, ax_b, ax_c, ax_d]
scatter_axes = [ax_e, ax_f]
all_axes = bar_axes + scatter_axes

# (a) Response rate
exp_rr = resp_rate.loc[resp_rate['group'] == 'expert', 'response_rate']
nov_rr = resp_rate.loc[resp_rate['group'] == 'novice', 'response_rate']
plot_group_comparison(ax_a, exp_rr, nov_rr,
                      ylabel='Response rate', title='Response rate',
                      ylim_top=1.12)

# (b) Response transitivity
exp_tr = transitivity.loc[transitivity['group'] == 'expert', 'prop_transitive'].dropna()
nov_tr = transitivity.loc[transitivity['group'] == 'novice', 'prop_transitive'].dropna()
plot_group_comparison(ax_b, exp_tr, nov_tr,
                      ylabel='Transitivity',
                      title='Transitivity',
                      ylim_top=0.65)

# (c) Checkmate preference
exp_cp = check_pref.loc[check_pref['group'] == 'expert', 'prop_check_preferred'].dropna()
nov_cp = check_pref.loc[check_pref['group'] == 'novice', 'prop_check_preferred'].dropna()
plot_group_comparison(ax_c, exp_cp, nov_cp,
                      ylabel='P(check preferred)',
                      title='Checkmate pref.',
                      chance=0.5, ylim_top=1.12)

# (d) Visual pairs consistency
exp_cnc = board_profile.loc[board_profile['group'] == 'expert', 'cnc_r'].dropna()
nov_cnc = board_profile.loc[board_profile['group'] == 'novice', 'cnc_r'].dropna()
plot_group_comparison(ax_d, exp_cnc, nov_cnc,
                      ylabel='C-NC within-pair r',
                      title='Pairs consistency',
                      chance=0, ylim_top=1.12)

# (e) Board preference — Experts
plot_board_preference(ax_e, board_group, 'expert', EXPERT_COLOR,
                      'Board preference (Experts)')

# (f) Board preference — Novices
plot_board_preference(ax_f, board_group, 'novice', NOVICE_COLOR,
                      'Board preference (Novices)')

# Panel labels (a-f)
for i, ax in enumerate(all_axes):
    ax.text(-0.14, 1.08, chr(ord('a') + i), transform=ax.transAxes,
            fontsize=PP['font_size_panel_label'], fontweight='bold',
            va='top', ha='left')

# Row suptitles — centered across full figure width, y from axes transforms
# Compute figure-center y just above each row's panel titles
fig.canvas.draw()  # force layout so get_position() is valid

# Row 1 suptitle: above bar plots, centered on full figure
bbox_row1 = ax_b.get_position()
fig.text(0.5, bbox_row1.y1 + 0.08, 'Task engagement diagnostics',
         ha='center', va='bottom', fontsize=PP['font_size_title'],
         fontweight='bold', transform=fig.transFigure)

# Row 2 suptitle: above scatter plots, centered on full figure
bbox_row2 = ax_e.get_position()
fig.text(0.5, bbox_row2.y1 + 0.06, 'Board preference profiles within visual pairs',
         ha='center', va='bottom', fontsize=PP['font_size_title'],
         fontweight='bold', transform=fig.transFigure)

# Legend — framed box, between panels e and f
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=EXPERT_COLOR,
           markersize=4, label='Expert'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=NOVICE_COLOR,
           markersize=4, label='Novice'),
]
# Position between the two scatter plots (right edge of ax_e / left edge of ax_f)
bbox_e = ax_e.get_position()
bbox_f = ax_f.get_position()
legend_x = (bbox_e.x1 + bbox_f.x0) / 2 - 0.04
legend_y = (bbox_e.y0 + bbox_e.y1) / 2 + 0.10
fig.legend(handles=legend_handles, loc='center', ncol=1, frameon=True,
           fontsize=PP['font_size_legend'], edgecolor='black',
           fancybox=False, framealpha=1.0,
           bbox_to_anchor=(legend_x, legend_y))

save_panel_pdf(fig, figures_dir / 'novice_diagnostics_panel.pdf')
fig.savefig(figures_dir / 'novice_diagnostics_panel.svg',
            format='svg', bbox_inches='tight')
logger.info("Saved combined panel (3x2)")
plt.close(fig)


# ============================================================================
# Panel 2: Response rate by condition (2-AFC verification)
# ============================================================================
# Shows that response rates are uniform across trial conditions (advantageous,
# non-advantageous, same-status), confirming 2-AFC task design. Within each
# bar, a darker portion shows P(choose current), revealing the discriminative
# signal: experts show strong checkmate sensitivity, novices do not.

logger.info("Plotting response rate by condition panel...")

# Reload events for condition analysis
stim = pd.read_csv(CONFIG['STIMULI_FILE'], sep="\t")
check_ids_set = set(stim.loc[stim['check_status'] == 'checkmate', 'stim_id'].astype(int))
BIDS_ROOT = Path(CONFIG['BIDS_ROOT'])
ppt = pd.read_csv(CONFIG['BIDS_PARTICIPANTS'], sep="\t")
lw = PP['base_linewidth']

all_ev = []
for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
    func_dir = sub_dir / "func"
    if not func_dir.exists():
        continue
    sub_id = sub_dir.name
    for ev_file in sorted(func_dir.glob("*_events.tsv")):
        df = pd.read_csv(ev_file, sep="\t")
        df['subject'] = sub_id
        run_str = ev_file.stem.split("run-")[1].split("_")[0]
        df['run'] = int(run_str)
        all_ev.append(df)

ev_all = pd.concat(all_ev, ignore_index=True)
ev_all = ev_all.merge(ppt[['participant_id', 'group']],
                       left_on='subject', right_on='participant_id', how='left')
ev_all = ev_all[ev_all['subject'] != 'sub-04']

# Compute per-subject per-condition stats
cond_rows = []
for sub_id in ev_all['subject'].unique():
    sub_ev = ev_all[ev_all['subject'] == sub_id]
    group = sub_ev['group'].iloc[0]
    for run, grp in sub_ev.groupby('run'):
        grp = grp.sort_values('onset').reset_index(drop=True)
        for i in range(1, len(grp)):
            curr = int(grp.iloc[i]['stim_id'])
            prev = int(grp.iloc[i - 1]['stim_id'])
            pref = str(grp.iloc[i]['preference'])
            c_c = curr in check_ids_set
            c_p = prev in check_ids_set
            has = pref not in ('n/a', 'nan') and pd.notna(grp.iloc[i]['preference'])

            if c_c and not c_p:
                cond = 'Advantageous'
            elif not c_c and c_p:
                cond = 'Non-advantageous'
            else:
                cond = 'Same status'

            cond_rows.append({
                'subject': sub_id, 'group': group, 'condition': cond,
                'responded': int(has),
                'chose_current': int(has and pref == 'current_preferred'),
            })

cond_df = pd.DataFrame(cond_rows)

# Per-subject per-condition aggregation
sub_cond = (cond_df.groupby(['subject', 'group', 'condition'])
            .agg(n_trials=('responded', 'count'),
                 n_responded=('responded', 'sum'),
                 n_current=('chose_current', 'sum'))
            .reset_index())
sub_cond['resp_rate'] = sub_cond['n_responded'] / sub_cond['n_trials']
sub_cond['p_current'] = sub_cond['n_current'] / sub_cond['n_responded'].clip(lower=1)

# Figure
fig2 = plt.figure(2)
fig2.set_size_inches(figure_size(columns=2, height_mm=70))

gs2 = fig2.add_gridspec(1, 2, wspace=0.35, left=0.08, right=0.97,
                          bottom=0.18, top=0.85)

cond_order = ['Advantageous', 'Non-advantageous', 'Same status']
cond_short = ['Adv.', 'Non-adv.', 'Same']

for gi, (g, g_color, g_label) in enumerate([
    ('expert', EXPERT_COLOR, 'Experts'),
    ('novice', NOVICE_COLOR, 'Novices'),
]):
    ax = fig2.add_subplot(gs2[0, gi])
    ax.set_label(f'resprate_{g_label}')

    gd = sub_cond[sub_cond['group'] == g]
    x = np.arange(len(cond_order))
    bar_w = 0.6

    for ci, cond in enumerate(cond_order):
        cd = gd[gd['condition'] == cond]
        rr_mean = cd['resp_rate'].mean()
        rr_sem = cd['resp_rate'].sem()
        pc_mean = cd['p_current'].mean()

        # Full bar = response rate (lighter)
        ax.bar(x[ci], rr_mean, bar_w, color=g_color, alpha=PP['bar_alpha'] * 0.5,
               edgecolor='black', linewidth=lw)
        # Inner bar = P(current) portion of the response rate (darker)
        ax.bar(x[ci], rr_mean * pc_mean, bar_w, color=g_color,
               alpha=PP['bar_alpha'], edgecolor='none')
        # Error bar on total response rate
        ax.errorbar(x[ci], rr_mean, yerr=rr_sem, fmt='none', color='black',
                    capsize=2, linewidth=lw)
        # Annotate P(current)
        ax.text(x[ci], rr_mean + rr_sem + 0.02,
                f'{pc_mean:.2f}', ha='center', va='bottom',
                fontsize=PP['font_size_annotation'], color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(cond_short, fontsize=PP['font_size_tick'])
    ax.set_ylabel('Response rate', fontsize=PP['font_size_label'])
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=lw, alpha=0.4)
    set_axis_title(ax, title='Response rate by condition', subtitle=g_label)
    style_spines(ax)

    # Legend only on first panel
    if gi == 0:
        legend_elements = [
            Patch(facecolor=g_color, alpha=PP['bar_alpha'], edgecolor='black',
                  linewidth=lw, label='P(choose current)'),
            Patch(facecolor=g_color, alpha=PP['bar_alpha'] * 0.5, edgecolor='black',
                  linewidth=lw, label='P(choose previous)'),
        ]
        ax.legend(handles=legend_elements, fontsize=PP['font_size_legend'],
                  frameon=False, loc='upper right')

# Panel labels
for i, ax in enumerate(fig2.axes):
    ax.text(-0.14, 1.08, chr(ord('a') + i), transform=ax.transAxes,
            fontsize=PP['font_size_panel_label'], fontweight='bold',
            va='top', ha='left')

#% start: automatic generated code from pylustrator
plt.figure(2).ax_dict = {ax.get_label(): ax for ax in plt.figure(2).axes}
getattr(plt.figure(2), '_pylustrator_init', lambda: ...)()
#% end: automatic generated code from pylustrator

if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()

save_panel_pdf(fig2, figures_dir / 'response_rate_by_condition_panel.pdf')
fig2.savefig(figures_dir / 'response_rate_by_condition_panel.svg',
             format='svg', bbox_inches='tight')
logger.info("Saved response rate by condition panel")
plt.close(fig2)


# ============================================================================

log_script_end(logger)
