#!/usr/bin/env python3
"""
Pylustrator-driven layout for Neurosynth term maps visualization.

Creates glass brain and flat surface panels for each Neurosynth term association map,
arranged with pylustrator.

Usage:
    python 91_plot_neurosynth_terms.py
Then arrange axes in the pylustrator window and save to inject layout code.
"""

import sys
import os
from pathlib import Path

# Ensure repo root on sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
script_dir = Path(__file__).parent

# Import pylustrator BEFORE creating any Matplotlib figures
import pylustrator
pylustrator.start()

import matplotlib.pyplot as plt
from nilearn import image, plotting

from common import CONFIG
from common.plotting import (
    apply_nature_rc,
    plot_flat_pair,
    compute_surface_symmetric_range,
    embed_figure_on_ax,
    save_axes_svgs,
    save_panel_svg,
    save_axes_pdfs,
    save_panel_pdf,
    CMAP_BRAIN,
)
from common.logging_utils import setup_analysis, log_script_end


def _load_term_maps(term_dir: Path):
    """
    Load Neurosynth term maps from directory.

    Returns dict mapping term name -> NIfTI path.
    """
    out = {}
    for f in sorted(term_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name.endswith('.nii') or f.name.endswith('.nii.gz'):
            term = f.stem.replace('_', ' ').lower()
            # Remove numeric prefix if present (e.g., "01_term" -> "term")
            parts = term.split(' ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                term = parts[1]
            out[term] = f
    return out


apply_nature_rc()

# Set up analysis
config, out_dir, logger = setup_analysis(
    analysis_name="neurosynth_terms_panel",
    results_base=script_dir / "results",
    script_file=__file__,
)
figs_dir = out_dir / 'figures'
figs_dir.mkdir(parents=True, exist_ok=True)

term_dir = Path(CONFIG['NEUROSYNTH_TERMS_DIR'])
term_maps = _load_term_maps(term_dir)
order = CONFIG.get('NEUROSYNTH_TERM_ORDER') or list(term_maps.keys())
logger.info(f"Loaded {len(term_maps)} term maps from {term_dir}")

# Preload images and compute symmetric range for surfaces
zimgs = [image.load_img(str(term_maps[t])) for t in order if t in term_maps]
vmin, vmax = compute_surface_symmetric_range(zimgs)
logger.info(f"Neurosynth shared symmetric color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")


# =============================================================================
# Create a single figure with independent axes (to arrange in pylustrator)
# =============================================================================

fig = plt.figure(1)

for i, term in enumerate(order, start=1):
    if term not in term_maps:
        logger.warning(f"Term '{term}' not found in term maps; skipping")
        continue

    z_img = image.load_img(str(term_maps[term]))

    # Glass brain axis
    ax_glass = plt.axes()
    ax_glass.set_label(f'{i:02d}_Glass_{term.replace(" ", "_")}')
    glass_fig = plotting.plot_glass_brain(
        z_img,
        display_mode='lyrz',
        colorbar=False,
        cmap=CMAP_BRAIN,
        symmetric_cbar=True,
        plot_abs=False,
        threshold=1e-5,
        vmin=vmin,
        vmax=vmax,
    ).frame_axes.figure
    embed_figure_on_ax(ax_glass, glass_fig, title=f'{term.title()} — Glass Brain')

    # Flat surface axis
    ax_surface = plt.axes()
    ax_surface.set_label(f'{i:02d}_Surface_{term.replace(" ", "_")}')
    surface_fig = plot_flat_pair(
        data=z_img,
        title='',
        threshold=1e-5,
        output_file=None,
        show_hemi_labels=False,
        show_colorbar=False,
        vmin=vmin,
        vmax=vmax,
        show_directions=True,
    )
    embed_figure_on_ax(ax_surface, surface_fig, title=f'{term.title()} — Surface')

# Provide axis dictionary for pylustrator convenience
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}

# Show pylustrator window; save to inject layout code
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
# Add layout code here after arranging in pylustrator
#% end: automatic generated code from pylustrator
plt.show()

# Save each axis separately first, then full panel (SVG + PDF)
save_axes_svgs(fig, figs_dir, 'neurosynth_terms')
save_axes_pdfs(fig, figs_dir, 'neurosynth_terms')
save_panel_svg(fig, figs_dir / 'panels' / 'neurosynth_terms_panel.svg')
save_panel_pdf(fig, figs_dir / 'panels' / 'neurosynth_terms_panel.pdf')

log_script_end(logger)
