"""
Pylustrator-driven layout for Neurosynth RSA results (searchlight Experts > Novices).

Creates all required panels using existing primitives but leaves the final
arrangement to pylustrator:

- For each pattern (Checkmate, Strategy, Visual Similarity):
  - Correlations (POS vs NEG) vs term maps
  - Correlation differences (POS − NEG)
  - Flat surface maps (left+right) embedded

Usage:
    python 92_plot_neurosynth_rsa.py
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
from nilearn import image

from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    set_axis_title,
    PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    save_axes_svgs,
    save_panel_svg,
    save_axes_pdfs,
    save_panel_pdf,
    compute_surface_symmetric_range,
)
from common.io_utils import find_latest_results_directory
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.plotting import plot_flat_pair
from modules.plot_utils import (
    plot_correlations_on_ax,
    plot_differences_on_ax,
    embed_figure_on_ax,
)


PATTERNS = [
    ('searchlight_checkmate', 'Checkmate | RSA searchlight'),
    ('searchlight_strategy', 'Strategy | RSA searchlight'),
    ('searchlight_visualSimilarity', 'Visual Similarity | RSA searchlight'),
]


## Helpers moved to modules.plot_utils


# =============================================================================
# Setup and load data
# =============================================================================

apply_nature_rc()

RESULTS_DIR = find_latest_results_directory(
    script_dir / 'results',
    pattern='*_neurosynth_rsa',
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
    log_name='pylustrator_neurosynth_rsa.log',
)


def _load_triple(stem: str):
    pos = pd.read_csv(RESULTS_DIR / f"{stem}_term_corr_positive.csv")
    neg = pd.read_csv(RESULTS_DIR / f"{stem}_term_corr_negative.csv")
    diff = pd.read_csv(RESULTS_DIR / f"{stem}_term_corr_difference.csv")
    return pos, neg, diff


# =============================================================================
# Create a single figure with independent axes (to arrange in pylustrator)
# =============================================================================

fig = plt.figure(1)

z_imgs = [image.load_img(str(RESULTS_DIR / f"zmap_{stem}.nii.gz")) for stem, _pretty in PATTERNS]
vmin_rsa, vmax_rsa = compute_surface_symmetric_range(z_imgs)

for idx, (stem, pretty) in enumerate(PATTERNS, start=1):
    try:
        df_pos, df_neg, df_diff = _load_triple(stem)
    except Exception as e:
        continue

    # Correlation bars
    ax_corr = plt.axes(); ax_corr.set_label(f'{idx}A_RSA_Corr_{stem}')
    plot_correlations_on_ax(ax_corr, df_pos, df_neg, title=f'{pretty}: Term correlations')

    # Differences
    ax_diff = plt.axes(); ax_diff.set_label(f'{idx}B_RSA_Diff_{stem}')
    plot_differences_on_ax(ax_diff, df_diff, title=f'{pretty}: ΔCorrelation (pos − neg)')

    # Flat surfaces (left+right pair embedded)
    ax_flat = plt.axes(); ax_flat.set_label(f'{idx}C_RSA_Flat_{stem}')
    z_img_rsa = image.load_img(str(RESULTS_DIR / f"zmap_{stem}.nii.gz"))
    surface_fig_rsa = plot_flat_pair(
        data=z_img_rsa,
        title='',
        threshold=0,
        output_file=None,
        show_hemi_labels=False,
        show_colorbar=False,
        vmin=vmin_rsa,
        vmax=vmax_rsa,
        show_directions=True,
    )
    embed_figure_on_ax(ax_flat, surface_fig_rsa, title=f'{pretty} (flat surfaces)')


# Provide axis dictionary for pylustrator convenience
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}


# Show pylustrator window; save to inject layout code
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].set(position=[0.4529, 0.1361, 0.2768, 0.21])
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].title.set(visible=False)
plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].text(0.4999, 1.0453, 'Checkmate', transform=plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["1A_RSA_Corr_searchlight_checkmate"].texts[0].new
plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].set(position=[0.7832, 0.1361, 0.129, 0.21])
plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].title.set(visible=False)
plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].text(0.5282, 1.0453, 'Checkmate', transform=plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["1B_RSA_Diff_searchlight_checkmate"].texts[0].new
plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].set(position=[0.005021, 0.09398, 0.3889, 0.2942])
plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].title.set(visible=False)
plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].text(0.5000, 0.8892, 'Checkmate', transform=plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["1C_RSA_Flat_searchlight_checkmate"].texts[1].new
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].set(position=[0.4529, 0.4473, 0.2768, 0.21])
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].title.set(visible=False)
plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].text(0.5000, 1.0454, 'Strategy', transform=plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["2A_RSA_Corr_searchlight_strategy"].texts[0].new
plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].set(position=[0.7832, 0.4473, 0.129, 0.21])
plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].title.set(visible=False)
plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].text(0.5284, 1.0454, 'Strategy', transform=plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["2B_RSA_Diff_searchlight_strategy"].texts[0].new
plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].set(position=[0.005021, 0.4052, 0.3889, 0.2943])
plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].title.set(visible=False)
plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].text(0.5000, 0.8891, 'Strategy', transform=plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["2C_RSA_Flat_searchlight_strategy"].texts[0].new
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].set(position=[0.4529, 0.7505, 0.2768, 0.21])
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].title.set(visible=False)
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].text(0.5000, 1.1234, 'Neurosynth-RSA correlations', transform=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].transAxes, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].texts[0].new
plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].text(0.5000, 1.0392, 'Visual similarity', transform=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["3A_RSA_Corr_searchlight_visualSimilarity"].texts[1].new
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].set(position=[0.7832, 0.7505, 0.129, 0.21])
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].spines[['right', 'top']].set_visible(False)
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].title.set(visible=False)
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].text(0.5284, 1.1234, 'Correlations differences', transform=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].transAxes, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].texts[0].new
plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].text(0.5284, 1.0392, 'Visual similarity', transform=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["3B_RSA_Diff_searchlight_visualSimilarity"].texts[1].new
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].set(position=[0.005021, 0.7084, 0.3889, 0.2943])
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].title.set(visible=False)
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].text(0.5000, 0.8847, 'Visual similarity', transform=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].texts[0].new
plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].text(0.6428, 0.8214, 'Right hemisphere', transform=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].transAxes, ha='center', color='#b2b2b2ff')  # id=plt.figure(1).ax_dict["3C_RSA_Flat_searchlight_visualSimilarity"].texts[1].new
plt.figure(1).text(0.1995, 0.9864, 'RSA Searchlight maps surface projection', transform=plt.figure(1).transFigure, ha='center', fontsize=7., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.1156, 0.9501, 'Left hemisphere', transform=plt.figure(1).transFigure, color='#b2b2b2ff')  # id=plt.figure(1).texts[1].new
#% end: automatic generated code from pylustrator
plt.show()

# Save each axis separately first, then full panel (SVG + PDF)
fig = plt.gcf()
save_axes_svgs(fig, FIGURES_DIR, 'neurosynth_rsa')
save_axes_pdfs(fig, FIGURES_DIR, 'neurosynth_rsa')
save_panel_svg(fig, FIGURES_DIR / 'panels' / 'neurosynth_rsa_panel.svg')
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'neurosynth_rsa_panel.pdf')
log_script_end(logger)
