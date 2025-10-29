"""
Pylustrator-driven layout for Neurosynth univariate results.

Creates all required panels using existing primitives but leaves the final
arrangement to pylustrator:

- Correlations (POS vs NEG) for All > Rest
- Correlation differences (POS − NEG) for All > Rest
- Correlations (POS vs NEG) for Check > No-Check
- Correlation differences (POS − NEG) for Check > No-Check
- Flat surface maps (left+right) for All > Rest
- Flat surface maps (left+right) for Check > No-Check

Usage:
    python 91_plot_neurosynth_univariate.py
Then arrange axes in the pylustrator window and save to inject layout code.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

# Import pylustrator BEFORE any Matplotlib figures are created
import pylustrator
pylustrator.start()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nilearn import image, surface, datasets

from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    set_axis_title,
    PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    save_axes_svgs,
    save_panel_svg,
)
from modules.plot_utils import (
    plot_correlations_on_ax,
    plot_differences_on_ax,
    embed_flat_pair_on_ax,
)
from common.io_utils import find_latest_results_directory
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.plotting import plot_flat_pair


## Helpers moved to modules.plot_utils


# =============================================================================
# Setup and load data
# =============================================================================

apply_nature_rc()

RESULTS_DIR = find_latest_results_directory(
    script_dir / 'results',
    pattern='*_neurosynth_univariate',
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True,
)
FIGURES_DIR = RESULTS_DIR / 'figures'

extra = {"RESULTS_DIR": str(RESULTS_DIR), "FIGURES_DIR": str(FIGURES_DIR)}
_, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='pylustrator_neurosynth_univariate.log',
)


# Load CSVs for both contrasts
stem_all = 'spmT_exp-gt-nonexp_all-gt-rest'
stem_check = 'spmT_exp-gt-nonexp_check-gt-nocheck'

def _load_triple(stem: str):
    pos = pd.read_csv(RESULTS_DIR / f"{stem}_term_corr_positive.csv")
    neg = pd.read_csv(RESULTS_DIR / f"{stem}_term_corr_negative.csv")
    diff = pd.read_csv(RESULTS_DIR / f"{stem}_term_corr_difference.csv")
    return pos, neg, diff

df_pos_all, df_neg_all, df_diff_all = _load_triple(stem_all)
df_pos_chk, df_neg_chk, df_diff_chk = _load_triple(stem_check)


# =============================================================================
# Create a single figure with independent axes (to arrange in pylustrator)
# =============================================================================

fig = plt.figure(1)

# Correlation bars
ax_A1 = plt.axes(); ax_A1.set_label('A1_Corr_All_gt_Rest')
plot_correlations_on_ax(ax_A1, df_pos_all, df_neg_all, title='Term correlations (All > Rest)')

ax_B1 = plt.axes(); ax_B1.set_label('B1_Corr_Check_gt_NoCheck')
plot_correlations_on_ax(ax_B1, df_pos_chk, df_neg_chk, title='Term correlations (Check > No-Check)')

# Differences
ax_A2 = plt.axes(); ax_A2.set_label('A2_Diff_All_gt_Rest')
plot_differences_on_ax(ax_A2, df_diff_all, title='ΔCorrelation (pos − neg) (All > Rest)')

ax_B2 = plt.axes(); ax_B2.set_label('B2_Diff_Check_gt_NoCheck')
plot_differences_on_ax(ax_B2, df_diff_chk, title='ΔCorrelation (pos − neg) (Check > No-Check)')

# Compute symmetric surface range shared across univariate maps
try:
    fsavg = datasets.fetch_surf_fsaverage()
    def _absmax_surface(zimg):
        texl = surface.vol_to_surf(zimg, fsavg.pial_left)
        texr = surface.vol_to_surf(zimg, fsavg.pial_right)
        return float(np.nanmax(np.abs(np.concatenate([texl, texr]))))
    vmax_univ = 0.0
    for stem in [stem_all, stem_check]:
        z_img = image.load_img(str(RESULTS_DIR / f"zmap_{stem}.nii.gz"))
        vmax_univ = max(vmax_univ, _absmax_surface(z_img))
    vmin_univ = -vmax_univ
except Exception:
    # Fallback: compute from volume values
    def _absmax_volume(zimg):
        arr = zimg.get_fdata()
        return float(np.nanmax(np.abs(arr)))
    vmax_univ = 0.0
    for stem in [stem_all, stem_check]:
        z_img = image.load_img(str(RESULTS_DIR / f"zmap_{stem}.nii.gz"))
        vmax_univ = max(vmax_univ, _absmax_volume(z_img))
    vmin_univ = -vmax_univ

# Flat surfaces (left+right pair embedded)
ax_C = plt.axes(); ax_C.set_label('C_Flat_All_gt_Rest')
embed_flat_pair_on_ax(
    ax_C,
    zmap_path=RESULTS_DIR / f"zmap_{stem_all}.nii.gz",
    title='All > Rest (flat surfaces)',
    tmp_stem=FIGURES_DIR / 'pylu_surface_all_rest',
    vmin=vmin_univ, vmax=vmax_univ,
)

ax_D = plt.axes(); ax_D.set_label('D_Flat_Check_gt_NoCheck')
embed_flat_pair_on_ax(
    ax_D,
    zmap_path=RESULTS_DIR / f"zmap_{stem_check}.nii.gz",
    title='Check > No-Check (flat surfaces)',
    tmp_stem=FIGURES_DIR / 'pylu_surface_check_nocheck',
    vmin=vmin_univ, vmax=vmax_univ,
)


# Provide axis dictionary for pylustrator convenience
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}


# Show pylustrator window; save to inject layout code
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].set(position=[0.07445, 0.6342, 0.2189, 0.2292], ylim=(-0.15, 0.19))
plt.figure(1).ax_dict["A1_Corr_All_gt_Rest"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].set(position=[0.34, 0.6342, 0.1211, 0.2292], ylim=(-0.15, 0.35))
plt.figure(1).ax_dict["A2_Diff_All_gt_Rest"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].set(position=[0.5356, 0.6342, 0.2189, 0.2292], ylim=(-0.15, 0.19))
plt.figure(1).ax_dict["B1_Corr_Check_gt_NoCheck"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].set(position=[0.7997, 0.6342, 0.1211, 0.2292], ylim=(-0.15, 0.35))
plt.figure(1).ax_dict["B2_Diff_Check_gt_NoCheck"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["C_Flat_All_gt_Rest"].set(position=[0.07452, 0.2735, 0.3865, 0.2924])
plt.figure(1).ax_dict["D_Flat_Check_gt_NoCheck"].set(position=[0.5357, 0.2735, 0.3865, 0.2924])
plt.figure(1).text(0.1839, 0.8676, 'All boards > Baseline', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.4005, 0.8676, 'All boards > Baseline', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.6450, 0.8676, 'Checkmate > Non-checkmate', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.8602, 0.8676, 'Checkmate > Non-checkmate', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[3].new
plt.figure(1).text(0.2677, 0.5269, 'Surface projection', transform=plt.figure(1).transFigure, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).texts[4].new
plt.figure(1).text(0.7289, 0.5269, 'Surface projection', transform=plt.figure(1).transFigure, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).text(0.2677, 0.5143, 'All boards > Baseline', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[6].new
plt.figure(1).text(0.7289, 0.5143, 'Checkmate > Non-checkmate', transform=plt.figure(1).transFigure, ha='center')  # id=plt.figure(1).texts[7].new
#% end: automatic generated code from pylustrator
plt.show()

# Save each axis separately first, then full panel
fig = plt.gcf()
save_axes_svgs(fig, FIGURES_DIR, 'neurosynth_univariate')
save_panel_svg(fig, FIGURES_DIR / 'panels' / 'neurosynth_univariate_panel.svg')

log_script_end(logger)
