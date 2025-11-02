#!/usr/bin/env python3
"""
Generate Neurosynth Term Maps Supplementary Figure (Pylustrator)
================================================================

Creates publication-ready visualizations of Neurosynth term association maps
showing brain regions associated with specific cognitive terms. For each term,
generates both glass brain and flat surface visualizations with shared symmetric
color scale. Uses pylustrator for interactive layout arrangement.

Figures Produced
----------------

Neurosynth Terms Panel
- File: figures/panels/neurosynth_terms_panel.svg (and .pdf)
- Individual axes saved to figures/: neurosynth_terms_01_Glass_*.svg, neurosynth_terms_01_Surface_*.svg, etc.
- Content: For each Neurosynth term, shows two visualizations:
  1) Glass brain (multi-panel view: left, right, posterior, dorsal)
  2) Flat surface pair (left/right hemispheres)

Panel Layout:
- Pair of panels (glass brain + surface) for each term
- Terms ordered according to CONFIG['NEUROSYNTH_TERM_ORDER']
- Shared vmin/vmax across all terms for consistent color interpretation
- Colormap: CMAP_BRAIN (blue-white-red diverging)

Inputs
------
- Neurosynth term association maps (.nii.gz files) from CONFIG['NEUROSYNTH_TERMS_DIR']
- Files expected to be named: term_name.nii.gz or 01_term_name.nii.gz (with numeric prefix)
- Each file contains z-scores for association with that cognitive term

Dependencies
------------
- pylustrator (optional; import is guarded by CONFIG['ENABLE_PYLUSTRATOR'])
- nilearn (for loading NIfTI images and glass brain plotting)
- common.plotting primitives (apply_nature_rc, plot_flat_pair, save_axes_svgs)
- common.plotting.compute_surface_symmetric_range (for shared color scale)
- Strict I/O: fails if expected term maps are missing; no silent fallbacks

Usage
-----
python chess-supplementary/neurosynth-terms/91_plot_neurosynth_terms.py
"""

import sys
import os
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
script_dir = Path(__file__).parent

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import matplotlib.pyplot as plt
from nilearn import image, plotting
from common.plotting import (
    apply_nature_rc,
    plot_flat_pair,
    compute_ylim_range,
    embed_figure_on_ax,
    save_axes_svgs,
    save_panel_pdf,
    CMAP_BRAIN,
)
from common.neuro_utils import project_volume_to_surfaces
from common.logging_utils import setup_analysis, log_script_end

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chess-neurosynth')))
from modules.io_utils import load_term_maps, reorder_by_term

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()


# =============================================================================
# Configuration and Setup
# =============================================================================
# Initialize logging and create output directories for Neurosynth term figures

apply_nature_rc()  # Apply Nature journal style to all figures

# Set up analysis directory and logging
config, out_dir, logger = setup_analysis(
    analysis_name="neurosynth_terms_panel",
    results_base=script_dir / "results",
    script_file=__file__,
)
figs_dir = out_dir / 'figures'
figs_dir.mkdir(parents=True, exist_ok=True)

# Load Neurosynth term association maps from configured directory
term_dir = Path(CONFIG['NEUROSYNTH_TERMS_DIR'])
term_maps = load_term_maps(term_dir)  # Dict: term name -> file path

# Get term order from CONFIG or use alphabetical order
order = CONFIG.get('NEUROSYNTH_TERM_ORDER') or list(term_maps.keys())
logger.info(f"Loaded {len(term_maps)} term maps from {term_dir}")


# =============================================================================
# Compute Shared Color Scale
# =============================================================================
# Compute symmetric vmin/vmax across all term maps to ensure consistent
# color interpretation. Uses max absolute z-score projected to surface.

# Preload all term images (NIfTI volumes) and project to surfaces ONCE
zimgs = {}
surface_textures = {}
for t in order:
    zimg = image.load_img(str(term_maps[t]))
    zimgs[t] = zimg
    # Project each volume to surfaces once (DRY: no re-projection!)
    tex_l, tex_r = project_volume_to_surfaces(zimg)
    surface_textures[t] = (tex_l, tex_r)

# Compute symmetric range from ALL surface textures
# This ensures colorbar matches actual values shown on surface visualizations
all_textures = [tex for tex_pair in surface_textures.values() for tex in tex_pair]
vmin, vmax = compute_ylim_range(*all_textures, symmetric=True, padding_pct=0.0)
logger.info(f"Neurosynth shared symmetric color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")


# =============================================================================
# Figure: Neurosynth Term Maps (Independent Axes for Pylustrator)
# =============================================================================
# Creates two axes per term (glass brain + flat surface), arranged by pylustrator.
# Glass brain shows multi-panel view (left, right, posterior, dorsal).
# Flat surface shows bilateral hemisphere pair (left/right).

fig = plt.figure(1)

for i, term in enumerate(order, start=1):
    # Skip terms not found in term maps directory
    if term not in term_maps:
        logger.warning(f"Term '{term}' not found in term maps; skipping")
        continue

    # Load term association z-map
    z_img = image.load_img(str(term_maps[term]))

    # -----------------------------------------------------------------------------
    # Panel A: Glass Brain (Multi-panel view)
    # -----------------------------------------------------------------------------
    # Creates a glass brain visualization showing brain from multiple angles
    # Glass brain = semi-transparent overlay showing z-scores on anatomical template
    ax_glass = plt.axes()
    ax_glass.set_label(f'{i:02d}_Glass_{term.replace(" ", "_")}')

    # Create glass brain plot using nilearn
    # display_mode='lyrz' = left, right, posterior, dorsal views
    glass_fig = plotting.plot_glass_brain(
        z_img,                         # Neurosynth z-map
        display_mode='lyrz',           # Four-panel view (left, right, posterior, dorsal)
        colorbar=False,                # Don't show colorbar (shown separately)
        cmap=CMAP_BRAIN,               # Diverging colormap (blue-white-red)
        symmetric_cbar=True,           # Use symmetric color scale
        plot_abs=False,                # Don't plot absolute values
        threshold=1e-5,                # Minimal threshold (show all non-zero)
        vmin=vmin,                     # Shared color scale minimum
        vmax=vmax,                     # Shared color scale maximum
    ).frame_axes.figure

    # Embed glass brain figure in matplotlib axis with title
    embed_figure_on_ax(ax_glass, glass_fig, title=f'{term.title()} — Glass Brain')

    # -----------------------------------------------------------------------------
    # Panel B: Flat Surface Pair (Left/Right hemispheres)
    # -----------------------------------------------------------------------------
    # Creates bilateral flat surface visualization showing z-scores on cortical surface
    ax_surface = plt.axes()
    ax_surface.set_label(f'{i:02d}_Surface_{term.replace(" ", "_")}')

    # Get pre-computed textures for this term (no re-projection!)
    tex_l, tex_r = surface_textures[term]

    # Create flat surface plot using pre-computed textures (in-memory, no disk writes)
    surface_fig = plot_flat_pair(
        textures=(tex_l, tex_r),       # Pre-computed surface textures
        title='',                      # No title (added by embed_figure_on_ax)
        threshold=1e-5,                # Minimal threshold (show all non-zero)
        output_file=None,              # Don't save to disk (in-memory only)
        show_hemi_labels=False,        # Don't show L/R labels
        show_colorbar=False,           # Don't show colorbar (shown separately)
        vmin=vmin,                     # Shared color scale minimum
        vmax=vmax,                     # Shared color scale maximum
        show_directions=True,          # Show anterior/posterior labels
    )

    # Embed surface figure in matplotlib axis with title
    embed_figure_on_ax(ax_surface, surface_fig, title=f'{term.title()} — Surface')

# Provide axis dictionary for pylustrator convenience
# This allows pylustrator to reference axes by label
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}


# =============================================================================
# Pylustrator Auto-Generated Layout Code
# =============================================================================
# The code between "#% start:" and "#% end:" markers is automatically generated
# by pylustrator when you save the layout interactively. This code positions
# and styles each axis according to your manual adjustments in the GUI.
# DO NOT manually edit this section - it will be overwritten on next save.

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(14.000000/2.54, 12.490000/2.54, forward=True)
plt.figure(1).ax_dict["01_Glass_working_memory"].set(position=[0.01445, 0.7842, 0.2713, 0.1338])
plt.figure(1).ax_dict["01_Glass_working_memory"].set_position([0.015449, 0.784281, 0.290045, 0.133629])
plt.figure(1).ax_dict["01_Glass_working_memory"].text(0.4438, 0.9560, 'Term 1: Working Memory', transform=plt.figure(1).ax_dict["01_Glass_working_memory"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["01_Glass_working_memory"].texts[0].new
plt.figure(1).ax_dict["01_Glass_working_memory"].title.set(visible=False)
plt.figure(1).ax_dict["01_Surface_working_memory"].set(position=[0.01445, 0.6286, 0.2713, 0.1729])
plt.figure(1).ax_dict["01_Surface_working_memory"].set_position([0.015449, 0.628702, 0.290045, 0.172679])
plt.figure(1).ax_dict["01_Surface_working_memory"].title.set(visible=False)
plt.figure(1).ax_dict["02_Glass_navigation"].set(position=[0.3289, 0.7842, 0.2713, 0.1338])
plt.figure(1).ax_dict["02_Glass_navigation"].set_position([0.351612, 0.784281, 0.290045, 0.133629])
plt.figure(1).ax_dict["02_Glass_navigation"].text(0.5000, 0.9560, 'Term 2: Memory Retrieval', transform=plt.figure(1).ax_dict["02_Glass_navigation"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["02_Glass_navigation"].texts[0].new
plt.figure(1).ax_dict["02_Glass_navigation"].title.set(visible=False)
plt.figure(1).ax_dict["02_Surface_navigation"].set(position=[0.3289, 0.6286, 0.2713, 0.1729])
plt.figure(1).ax_dict["02_Surface_navigation"].set_position([0.351612, 0.628702, 0.290045, 0.172679])
plt.figure(1).ax_dict["02_Surface_navigation"].title.set(visible=False)
plt.figure(1).ax_dict["03_Glass_memory_retrieval"].set(position=[0.6433, 0.7842, 0.2713, 0.1338])
plt.figure(1).ax_dict["03_Glass_memory_retrieval"].set_position([0.687775, 0.784281, 0.290045, 0.133629])
plt.figure(1).ax_dict["03_Glass_memory_retrieval"].text(0.5000, 0.9560, 'Term 3: Navigation', transform=plt.figure(1).ax_dict["03_Glass_memory_retrieval"].transAxes, ha='center', fontsize=7., rotation=1.)  # id=plt.figure(1).ax_dict["03_Glass_memory_retrieval"].texts[0].new
plt.figure(1).ax_dict["03_Glass_memory_retrieval"].title.set(visible=False)
plt.figure(1).ax_dict["03_Surface_memory_retrieval"].set(position=[0.6433, 0.6286, 0.2713, 0.1729])
plt.figure(1).ax_dict["03_Surface_memory_retrieval"].set_position([0.687775, 0.628702, 0.290045, 0.172679])
plt.figure(1).ax_dict["03_Surface_memory_retrieval"].title.set(visible=False)
plt.figure(1).ax_dict["04_Glass_language_network"].set(position=[0.01445, 0.4654, 0.2713, 0.1338])
plt.figure(1).ax_dict["04_Glass_language_network"].set_position([0.015449, 0.465463, 0.290045, 0.133629])
plt.figure(1).ax_dict["04_Glass_language_network"].text(0.4481, 0.9457, 'Term 4: Language Network', transform=plt.figure(1).ax_dict["04_Glass_language_network"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["04_Glass_language_network"].texts[0].new
plt.figure(1).ax_dict["04_Glass_language_network"].title.set(visible=False)
plt.figure(1).ax_dict["04_Surface_language_network"].set(position=[0.01445, 0.3098, 0.2713, 0.1729])
plt.figure(1).ax_dict["04_Surface_language_network"].set_position([0.015449, 0.309884, 0.290045, 0.172679])
plt.figure(1).ax_dict["04_Surface_language_network"].title.set(visible=False)
plt.figure(1).ax_dict["05_Glass_object_recognition"].set(position=[0.6433, 0.4654, 0.2713, 0.1338])
plt.figure(1).ax_dict["05_Glass_object_recognition"].set_position([0.687775, 0.465463, 0.290045, 0.133629])
plt.figure(1).ax_dict["05_Glass_object_recognition"].text(0.5000, 0.9457, 'Term 6: Face Recognition', transform=plt.figure(1).ax_dict["05_Glass_object_recognition"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["05_Glass_object_recognition"].texts[0].new
plt.figure(1).ax_dict["05_Glass_object_recognition"].title.set(visible=False)
plt.figure(1).ax_dict["05_Surface_object_recognition"].set(position=[0.6433, 0.3098, 0.2713, 0.1729])
plt.figure(1).ax_dict["05_Surface_object_recognition"].set_position([0.687775, 0.309884, 0.290045, 0.172679])
plt.figure(1).ax_dict["05_Surface_object_recognition"].title.set(visible=False)
plt.figure(1).ax_dict["06_Glass_face_recognition"].set(position=[0.3289, 0.4654, 0.2713, 0.1338])
plt.figure(1).ax_dict["06_Glass_face_recognition"].set_position([0.351612, 0.465463, 0.290045, 0.133629])
plt.figure(1).ax_dict["06_Glass_face_recognition"].text(0.5000, 0.9457, 'Term 5: Object Recognition', transform=plt.figure(1).ax_dict["06_Glass_face_recognition"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["06_Glass_face_recognition"].texts[0].new
plt.figure(1).ax_dict["06_Glass_face_recognition"].title.set(visible=False)
plt.figure(1).ax_dict["06_Surface_face_recognition"].set(position=[0.3289, 0.3098, 0.2713, 0.1729])
plt.figure(1).ax_dict["06_Surface_face_recognition"].set_position([0.351612, 0.309884, 0.290045, 0.172679])
plt.figure(1).ax_dict["06_Surface_face_recognition"].title.set(visible=False)
plt.figure(1).ax_dict["07_Glass_early_visual"].set(position=[0.3289, 0.1585, 0.2713, 0.1338])
plt.figure(1).ax_dict["07_Glass_early_visual"].set_position([0.351612, 0.158622, 0.290045, 0.133629])
plt.figure(1).ax_dict["07_Glass_early_visual"].text(0.5000, 0.8939, 'Term 7: Early Visual', transform=plt.figure(1).ax_dict["07_Glass_early_visual"].transAxes, ha='center', fontsize=7.)  # id=plt.figure(1).ax_dict["07_Glass_early_visual"].texts[0].new
plt.figure(1).ax_dict["07_Glass_early_visual"].title.set(visible=False)
plt.figure(1).ax_dict["07_Surface_early_visual"].set(position=[0.3289, -0.001428, 0.2713, 0.1729])
plt.figure(1).ax_dict["07_Surface_early_visual"].set_position([0.351612, -0.001306, 0.290045, 0.172679])
plt.figure(1).ax_dict["07_Surface_early_visual"].title.set(visible=False)
plt.figure(1).text(0.3775, 0.9577, 'Neurosynth Term Maps', transform=plt.figure(1).transFigure, fontsize=7., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.403544, 0.957698])
#% end: automatic generated code from pylustrator

# Display figure in pylustrator GUI for interactive layout adjustment
plt.show()


# =============================================================================
# Save Figures (Individual Axes and Full Panel)
# =============================================================================
# After arranging axes in pylustrator, save both:
# 1. Individual axes as separate SVG/PDF files (for modular figure assembly)
# 2. Full panel as complete SVG/PDF file (for standalone use)

# Save individual axes (one file per axis, named by prefix + axis label)
save_axes_svgs(fig, figs_dir, 'neurosynth_terms')

# Save full panel (complete multi-axis figure)
save_panel_pdf(fig, figs_dir / 'panels' / 'neurosynth_terms_panel.pdf')

logger.info("✓ Panel: neurosynth term associations complete")

log_script_end(logger)
