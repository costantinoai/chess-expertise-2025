"""
Marginal Split-Half Reliability — Supplementary Figure Panel

Single publication panel showing the marginal reliability analysis:
  Row 1: Experts    Row 2: Novices
  (a) Pairwise comparison frequency per pair (histogram, C-NC vs same-cat)
  (b) Board presentations (bar per stimulus, checkmate/non-checkmate colored)
  (c) Marginal split-half scatter (half1 vs half2 selection frequency)

All per-half statistics are from one representative split (seed=42).

Colors: checkmate = gold (#fdb338), non-checkmate = blue (#025196),
        expert = green (#228833), novice = magenta (#aa3377).

Reads results from 02_marginal_split_half.py.
"""

from pathlib import Path

from common import CONFIG

if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from common import setup_script, log_script_end
from common.plotting import (
    apply_nature_rc, PLOT_PARAMS, COLORS_EXPERT_NOVICE,
    COLORS_CHECKMATE_NONCHECKMATE,
    style_spines, save_axes_svgs, save_panel_pdf, cm_to_inches,
    set_axis_title,
)

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='marginal_split_half',
    output_subdirs=['figures'],
    log_name='91_plot_reliability_panels.log',
)
FIGURES_DIR = dirs['figures']

apply_nature_rc()
PP = PLOT_PARAMS
c_exp = COLORS_EXPERT_NOVICE['expert']
c_nov = COLORS_EXPERT_NOVICE['novice']
c_cm = COLORS_CHECKMATE_NONCHECKMATE['checkmate']
c_nc = COLORS_CHECKMATE_NONCHECKMATE['non_checkmate']
lw = PP['base_linewidth']
ms = PP['marker_size']
N_STIM = 40

# ============================================================================
# Load results
# ============================================================================

logger.info("Loading marginal split-half results...")

pair_freq_exp = pd.read_csv(results_dir / "pair_frequency_experts.csv")
pair_freq_nov = pd.read_csv(results_dir / "pair_frequency_novices.csv")
board_pres_exp = pd.read_csv(results_dir / "board_presentations_experts.csv")
board_pres_nov = pd.read_csv(results_dir / "board_presentations_novices.csv")
scatter_exp = pd.read_csv(results_dir / "marginal_scatter_experts.csv")
scatter_nov = pd.read_csv(results_dir / "marginal_scatter_novices.csv")

# ============================================================================
# Figure: 2 rows × 3 columns
# ============================================================================

fig = plt.figure(1)
fig.set_size_inches(cm_to_inches(17.8), cm_to_inches(11.0))

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.65, wspace=0.45,
                       left=0.08, right=0.97, bottom=0.11, top=0.88)

datasets = [
    (pair_freq_exp, board_pres_exp, scatter_exp, c_exp, 'Experts'),
    (pair_freq_nov, board_pres_nov, scatter_nov, c_nov, 'Novices'),
]

all_axes = []

for row, (pair_freq, board_pres, scatter, group_color, group) in enumerate(datasets):

    # ── (a) Pairwise comparison frequency (per half, seed=42) ─────────
    ax_a = fig.add_subplot(gs[row, 0])
    ax_a.set_label(f'a_PairFreq_{group}')

    h1_freq = pair_freq['h1_frequency']
    freq_cn = h1_freq[pair_freq['is_cross_category'].astype(bool)]
    freq_same = h1_freq[~pair_freq['is_cross_category'].astype(bool)]

    bins = np.arange(-0.5, h1_freq.max() + 1.5, 1)
    ax_a.hist(freq_cn, bins=bins, color=c_cm, alpha=0.55,
              edgecolor='none', label='C-NC', zorder=2)
    ax_a.hist(freq_same, bins=bins, color=c_nc, alpha=0.40,
              edgecolor='none', label='Same-cat', zorder=1)

    ax_a.axvline(h1_freq.mean(), color='black',
                 linestyle='--', linewidth=PP['plot_linewidth'], alpha=0.7,
                 label=f'Mean={h1_freq.mean():.0f}')

    ax_a.set_xlabel('Comparisons per pair', fontsize=PP['font_size_label'])
    ax_a.set_ylabel('Number of pairs', fontsize=PP['font_size_label'])
    set_axis_title(ax_a, title='Pairwise comparison freq. (half-sample)',
                   subtitle=group)
    ax_a.legend(fontsize=PP['font_size_legend'], frameon=False,
                loc='upper right')
    style_spines(ax_a)
    all_axes.append(ax_a)

    # ── (b) Board presentations (per half, seed=42) ───────────────────
    ax_b = fig.add_subplot(gs[row, 1])
    ax_b.set_label(f'b_BoardPres_{group}')

    x_boards = board_pres['stim_id'].values
    n_pres = board_pres['h1_presentations'].values
    bar_colors = [c_cm if cs == 'checkmate' else c_nc
                  for cs in board_pres['check_status']]

    ax_b.bar(x_boards, n_pres, width=0.8, color=bar_colors,
             alpha=PP['bar_alpha'], edgecolor='black', linewidth=lw * 0.3)

    ax_b.axhline(n_pres.mean(), color='black', linestyle='--',
                 linewidth=PP['plot_linewidth'], alpha=0.7)

    ax_b.set_xlabel('Stimulus ID', fontsize=PP['font_size_label'])
    ax_b.set_ylabel('Presentations', fontsize=PP['font_size_label'])
    ax_b.set_xlim(0, N_STIM + 1)
    set_axis_title(ax_b, title='Board presentations (half-sample)',
                   subtitle=f'{group} (M={n_pres.mean():.0f})')
    style_spines(ax_b)
    all_axes.append(ax_b)

    # ── (c) Marginal split-half scatter (seed=42) ─────────────────────
    ax_c = fig.add_subplot(gs[row, 2])
    ax_c.set_label(f'c_MargScatter_{group}')

    for cs, color_cs, label_cs in [('checkmate', c_cm, 'Checkmate'),
                                     ('non_checkmate', c_nc, 'Non-checkmate')]:
        mask = scatter['check_status'] == cs
        ax_c.scatter(scatter.loc[mask, 'half1_freq'],
                     scatter.loc[mask, 'half2_freq'],
                     color=color_cs, s=ms * 1.5, alpha=0.75,
                     edgecolors='white', linewidths=0.3, zorder=3,
                     label=label_cs)

    # Identity line
    ax_c.plot([0, 1], [0, 1], color='#555555', linestyle='-',
              linewidth=lw, alpha=0.4, zorder=1)

    # Correlation annotation
    r_val, p_val = stats.spearmanr(scatter['half1_freq'], scatter['half2_freq'])
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    ax_c.text(0.05, 0.95, f'r = {r_val:.2f}{sig}', transform=ax_c.transAxes,
              fontsize=PP['font_size_annotation'], ha='left', va='top')

    ax_c.set_xlabel('Half 1 selection freq.', fontsize=PP['font_size_label'])
    ax_c.set_ylabel('Half 2 selection freq.', fontsize=PP['font_size_label'])
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
    ax_c.set_aspect('equal')
    set_axis_title(ax_c, title='Marginal split-half (half-sample)',
                   subtitle=group)
    ax_c.legend(fontsize=PP['font_size_legend'], frameon=False,
                loc='lower right', markerscale=0.8)
    style_spines(ax_c)
    all_axes.append(ax_c)

# ── Panel labels ──────────────────────────────────────────────────────
labels = 'abcdef'
for i, ax in enumerate(all_axes):
    ax.text(-0.16, 1.10, labels[i], transform=ax.transAxes,
            fontsize=PP['font_size_panel_label'], fontweight='bold',
            va='top', ha='left')

# ── Pylustrator ───────────────────────────────────────────────────────
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["a_PairFreq_Experts"].set(position=[0.05485, 0.6123, 0.278, 0.3209], xlim=(0., 20.), ylim=(0., 80.))
plt.figure(1).ax_dict["a_PairFreq_Novices"].set(position=[0.05485, 0.1056, 0.278, 0.3209], xlim=(0., 20.), ylim=(0., 80.))
plt.figure(1).ax_dict["b_BoardPres_Experts"].set(position=[0.4158, 0.6123, 0.278, 0.3209], yticks=[0., 50., 100., 150., 200.], yticklabels=['0', '50', '100', '150', '200'], ylim=(0., 200.))
plt.figure(1).ax_dict["b_BoardPres_Novices"].set(position=[0.4158, 0.1056, 0.278, 0.3209], yticks=[0., 50., 100., 150., 200.], yticklabels=['0', '50', '100', '150', '200'], ylim=(0., 200.))
plt.figure(1).ax_dict["c_MargScatter_Experts"].set(position=[0.7808, 0.6188, 0.1892, 0.3079], xlim=(0., 1.), ylim=(0., 1.))
plt.figure(1).ax_dict["c_MargScatter_Novices"].set(position=[0.7768, 0.1056, 0.1971, 0.3209], xlim=(0., 1.), yticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], yticklabels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], ylim=(0., 1.))
plt.figure(1).ax_dict["c_MargScatter_Novices"].yaxis.labelpad = 2.034640
plt.figure(1).ax_dict["c_MargScatter_Novices"].get_legend().set(visible=False)
#% end: automatic generated code from pylustrator

if CONFIG['ENABLE_PYLUSTRATOR']:
    plt.show()

save_axes_svgs(fig, FIGURES_DIR, 'marginal_reliability')
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'marginal_reliability_panel.pdf')

logger.info("✓ Panel: marginal reliability complete")
log_script_end(logger)
