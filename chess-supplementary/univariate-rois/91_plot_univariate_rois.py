#!/usr/bin/env python3
"""
Pylustrator-driven layout for Univariate ROI results (Glasser-180 Bilateral).

Creates flat surface panels showing ROI-level t-statistics (Experts vs Novices)
for each univariate contrast, arranged with pylustrator.

Usage:
    python 91_plot_univariate_rois.py
Then arrange axes in the pylustrator window and save to inject layout code.
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Enable repo root imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import matplotlib.pyplot as plt
from nibabel.freesurfer import io as fsio
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.neuro_utils import roi_values_to_surface_texture
from common.plotting import (
    apply_nature_rc,
    plot_flat_pair,
    embed_figure_on_ax,
    save_axes_svgs,
    save_panel_svg,
    save_axes_pdfs,
    save_panel_pdf,
)
from modules import UNIV_CONTRASTS


script_dir = Path(__file__).parent
results_base = script_dir / 'results'

results_dir = find_latest_results_directory(
    results_base,
    pattern='*_univariate_rois',
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True,
)

figures_dir = results_dir / 'figures'

extra = {"RESULTS_DIR": str(results_dir), "FIGURES_DIR": str(figures_dir)}
config, _, logger = setup_analysis_in_dir(
    results_dir,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='pylustrator_univariate_rois.log',
)

apply_nature_rc()

# Load analysis results
with open(results_dir / 'univ_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

# Load surface annotations and ROI metadata
labels_l, _, _ = fsio.read_annot(CONFIG['ROI_GLASSER_180_ANNOT_L'])
labels_r, _, _ = fsio.read_annot(CONFIG['ROI_GLASSER_180_ANNOT_R'])

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_180'])
roi_info_bilateral = roi_info[roi_info['roi_id'] <= 180].copy()

logger.info(f"Loaded surface annotations: L={len(np.unique(labels_l))} labels, R={len(np.unique(labels_r))} labels")
logger.info(f"ROI metadata: {len(roi_info_bilateral)} bilateral ROIs")

contrasts = list(UNIV_CONTRASTS.keys())

# Compute shared symmetric vmin/vmax across all contrasts
all_abs = []
for con_code in contrasts:
    if 'contrasts' in index and con_code in index['contrasts']:
        tvals = index['contrasts'][con_code]['welch_expert_vs_novice']['t_stat'].to_numpy()
        all_abs.append(np.abs(tvals[np.isfinite(tvals)]))
vmax_shared = float(np.max(all_abs)) if len(all_abs) else 1.0
vmin_shared = -vmax_shared
logger.info(f"Univariate shared symmetric color scale (t): vmin={vmin_shared:.3f}, vmax={vmax_shared:.3f}")


# =============================================================================
# Create a single figure with independent axes (to arrange in pylustrator)
# =============================================================================

fig = plt.figure(1)

for idx, con_code in enumerate(contrasts, start=1):
    if 'contrasts' not in index or con_code not in index['contrasts']:
        logger.warning(f"No stats for {con_code}; skipping")
        continue

    blocks = index['contrasts'][con_code]
    welch = blocks['welch_expert_vs_novice']

    bilateral_roi_ids = welch['ROI_Label'].to_numpy()
    tvals = welch['t_stat'].to_numpy()
    finite_mask = np.isfinite(tvals)

    # Map bilateral values to both L and R hemisphere surfaces
    roi_ids = bilateral_roi_ids

    # Create textures for both hemispheres
    tex_l_all = roi_values_to_surface_texture(labels_l, roi_ids, tvals, include_mask=finite_mask, default_value=0.0)
    tex_r_all = roi_values_to_surface_texture(labels_r, roi_ids, tvals, include_mask=finite_mask, default_value=0.0)

    # Create surface plot and embed in matplotlib axis (in-memory, no disk writes)
    ax = plt.axes()
    ax.set_label(f'{idx}_Univariate_{con_code}')

    surface_fig = plot_flat_pair(
        data={'left': tex_l_all, 'right': tex_r_all},
        title='',
        threshold=None,
        output_file=None,
        show_hemi_labels=False,
        show_colorbar=False,
        vmin=vmin_shared,
        vmax=vmax_shared,
        show_directions=True,
    )
    embed_figure_on_ax(ax, surface_fig, title=f'{UNIV_CONTRASTS[con_code]} (Experts > Novices)')

# Provide axis dictionary for pylustrator convenience
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}

# Show pylustrator window; save to inject layout code
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["1_Univariate_con_0001"].set(position=[0.05, 0.55, 0.9, 0.4])
plt.figure(1).ax_dict["2_Univariate_con_0002"].set(position=[0.05, 0.05, 0.9, 0.4])
#% end: automatic generated code from pylustrator
plt.show()

# Save each axis separately first, then full panel (SVG + PDF)
save_axes_svgs(fig, figures_dir, 'univariate_rois')
save_axes_pdfs(fig, figures_dir, 'univariate_rois')
save_panel_svg(fig, figures_dir / 'panels' / 'univariate_rois_panel.svg')
save_panel_pdf(fig, figures_dir / 'panels' / 'univariate_rois_panel.pdf')

log_script_end(logger)
