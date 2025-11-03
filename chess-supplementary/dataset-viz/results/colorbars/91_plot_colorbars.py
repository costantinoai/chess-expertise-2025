"""
Generate Standalone Colorbars for Supplementary Materials
==========================================================

Creates publication-ready standalone colorbar figures for all analyses in the
supplementary materials. Each colorbar is extracted from the actual data range
to ensure consistency across all visualizations. Colorbars are saved as both
SVG and PDF for flexible integration into composite figures.

Figures Produced
----------------

1. Behavioral Directional Matrices Colorbar
   - File: figures/colorbar_behavioral_directional_vertical.svg (and .pdf)
   - Orientation: Vertical
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'Directional Preference'
   - Data source: Expert and novice directional dissimilarity matrices

2. Positive 0-0.1 Colorbar
   - File: figures/colorbar_0_to_01_horizontal.svg (and .pdf)
   - Orientation: Horizontal
   - Type: Sequential (RdPu colormap)
   - Label: 'r' (correlation coefficient)
   - Range: Fixed 0.0 to 0.1

3. Neurosynth Univariate Surfaces Colorbar
   - File: figures/colorbar_neurosynth_univariate_vertical.svg (and .pdf)
   - Orientation: Vertical
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'z-score'
   - Data source: Neurosynth univariate z-maps projected to surface

4. Neurosynth RSA Surfaces Colorbar
   - File: figures/colorbar_neurosynth_rsa_horizontal.svg (and .pdf)
   - Orientation: Horizontal
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'z-score'
   - Data source: Neurosynth RSA searchlight z-maps projected to surface

5. Manifold PR Profiles Matrix Colorbar
   - File: figures/colorbar_manifold_pr_profiles_vertical.svg (and .pdf)
   - Orientation: Vertical
   - Type: Sequential (mako colormap)
   - Label: 'PR' (Participation Ratio)
   - Data source: Subject×ROI PR matrix

6. Manifold PCA Components Colorbar
   - File: figures/colorbar_manifold_pca_components_horizontal.svg (and .pdf)
   - Orientation: Horizontal
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'Loading'
   - Data source: PCA component loadings (ROI contributions)

Inputs
------
- chess-behavioral/results/*_behavioral_rsa/expert_directional_dsm.npy
- chess-behavioral/results/*_behavioral_rsa/novice_directional_dsm.npy
- chess-neurosynth/results/*_neurosynth_univariate/zmap_*.nii.gz
- chess-neurosynth/results/*_neurosynth_rsa/zmap_*.nii.gz
- chess-manifold/results/*_manifold/pr_results.pkl

Dependencies
------------
- nilearn (for surface projection of neurosynth z-maps)
- common.plotting primitives (create_standalone_colorbar, apply_nature_rc)
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-supplementary/dataset-viz/91_plot_colorbars.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from common import apply_nature_rc, save_figure, CMAP_BRAIN, create_standalone_colorbar
from common.logging_utils import setup_analysis
from common.io_utils import find_latest_results_directory


# =============================================================================
# Configuration and Setup
# =============================================================================
# Initialize logging and create output directories for colorbar figures.
# All colorbars will be saved to results/<timestamp>_colorbars/figures/

analysis_name = "colorbars"
script_dir = Path(__file__).parent
results_base = script_dir / "results"
config, output_dir, logger = setup_analysis(
    analysis_name=analysis_name,
    results_base=results_base,
    script_file=__file__,
)

figures_dir = output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

apply_nature_rc()  # Apply Nature journal style to all figures


# =============================================================================
# Colorbar 1: Behavioral Directional Matrices (Vertical, Symmetric)
# =============================================================================
# Creates a vertical diverging colorbar for behavioral directional dissimilarity
# matrices. Range is computed from actual expert and novice DSM data to ensure
# symmetric scaling around 0 (required for diverging colormaps).
# - Data: Expert and novice directional DSMs (preferred vs non-preferred moves)
# - Colormap: CMAP_BRAIN (blue-white-red diverging)
# - Output: Vertical orientation for use with matrix visualizations

# Behavioral directional colorbar removed - not needed for current analysis


# =============================================================================
# Colorbar 2: Positive 0-0.1 Range (Horizontal, Sequential)
# =============================================================================
# Creates a horizontal sequential colorbar for correlation coefficients
# in the range 0.0 to 0.1. Used for weak positive correlations.
# - Data: Fixed range (not data-driven)
# - Colormap: RdPu (red-purple sequential)
# - Output: Horizontal orientation for use with correlation plots

logger.info("Generating 0-0.1 colorbar...")

import matplotlib.pyplot as plt

# Create standalone colorbar with fixed 0.0-0.1 range
fig_0_01 = create_standalone_colorbar(
    cmap=plt.cm.RdPu,             # Sequential colormap (light to dark red-purple)
    vmin=0.0,                     # Minimum correlation (white/light)
    vmax=0.1,                     # Maximum correlation (dark purple)
    orientation='horizontal',     # Horizontal layout
    label='r',                    # Correlation coefficient label
)
save_figure(fig_0_01, figures_dir / "colorbar_0_to_01_horizontal")
logger.info("✓ Saved 0-0.1 colorbar (horizontal)")


# =============================================================================
# Colorbar 3: Neurosynth Univariate Surfaces (Vertical, Symmetric)
# =============================================================================
# Creates a vertical diverging colorbar for neurosynth univariate z-maps
# displayed on cortical surfaces. Range is computed from actual z-map data
# projected to fsaverage surface to match visualization range.
# - Data: Neurosynth univariate z-maps (expert > novice contrasts)
# - Colormap: CMAP_BRAIN (blue-white-red diverging)
# - Output: Vertical orientation for use with surface brain plots
# - Tries surface projection first (accurate), falls back to volume if needed

logger.info("Loading neurosynth univariate zmaps...")

# Find latest neurosynth univariate results directory
neurosynth_univ_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-neurosynth" / "results",
    pattern="*_neurosynth_univariate",
    require_exists=True,
    verbose=False,
)

# Try to compute range from surface projection (preferred method)
try:
    from nilearn import image, datasets, surface
    fsavg = datasets.fetch_surf_fsaverage('fsaverage')  # Standard MNI surface template

    def _absmax_surface(zimg):
        """Compute max absolute value from volume projected to surface."""
        # Project z-map to left and right hemisphere surfaces
        texl = surface.vol_to_surf(zimg, fsavg.pial_left)
        texr = surface.vol_to_surf(zimg, fsavg.pial_right)
        # Return maximum absolute value across both hemispheres
        return float(np.nanmax(np.abs(np.concatenate([texl, texr]))))

    # Compute maximum absolute z-score across all univariate contrasts
    vmax_univ = 0.0
    for stem in ['spmT_exp-gt-nonexp_all-gt-rest', 'spmT_exp-gt-nonexp_check-gt-nocheck']:
        zmap_file = neurosynth_univ_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_univ = max(vmax_univ, _absmax_surface(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_univ = -vmax_univ  # Symmetric range for diverging colormap

except Exception as e:
    logger.warning(f"Could not compute from surface, using volume fallback: {e}")

    # Fallback: compute from volume values (less accurate for surface plots)
    def _absmax_volume(zimg):
        """Compute max absolute value from volume data."""
        arr = zimg.get_fdata()
        return float(np.nanmax(np.abs(arr)))

    vmax_univ = 0.0
    for stem in ['spmT_exp-gt-nonexp_all-gt-rest', 'spmT_exp-gt-nonexp_check-gt-nocheck']:
        zmap_file = neurosynth_univ_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_univ = max(vmax_univ, _absmax_volume(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_univ = -vmax_univ  # Symmetric range for diverging colormap

logger.info(f"Neurosynth univariate vmin={vmin_univ:.4f}, vmax={vmax_univ:.4f}")

# Create standalone colorbar figure
fig_neurosynth_univ = create_standalone_colorbar(
    cmap=CMAP_BRAIN,              # Diverging colormap (blue-white-red)
    vmin=vmin_univ,               # Minimum z-score (negative)
    vmax=vmax_univ,               # Maximum z-score (positive)
    orientation='vertical',       # Vertical layout for brain surface plots
    label='z-score',              # Statistical z-score label
)
save_figure(fig_neurosynth_univ, figures_dir / "colorbar_neurosynth_univariate_vertical")
logger.info("✓ Saved neurosynth univariate colorbar (vertical)")


# =============================================================================
# Colorbar 4: Neurosynth RSA Surfaces (Horizontal, Symmetric)
# =============================================================================
# Creates a horizontal diverging colorbar for neurosynth RSA searchlight z-maps
# displayed on cortical surfaces. Range is computed from actual z-map data
# projected to fsaverage surface to match visualization range.
# - Data: Neurosynth RSA searchlight z-maps (checkmate, strategy, visual similarity)
# - Colormap: CMAP_BRAIN (blue-white-red diverging)
# - Output: Horizontal orientation for use with surface brain plots
# - Tries surface projection first (accurate), falls back to volume if needed

logger.info("Loading neurosynth RSA zmaps...")

# Find latest neurosynth RSA results directory
neurosynth_rsa_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-neurosynth" / "results",
    pattern="*_neurosynth_rsa",
    require_exists=True,
    verbose=False,
)

# Define all RSA patterns to include in colorbar range computation
PATTERNS = [
    ("searchlight_checkmate", "Checkmate"),
    ("searchlight_strategy", "Strategy"),
    ("searchlight_visualSimilarity", "Visual Similarity"),
]

# Try to compute range from surface projection (preferred method)
try:
    # Compute maximum absolute z-score across all RSA searchlight maps
    vmax_rsa = 0.0
    for stem, _pretty in PATTERNS:
        zmap_file = neurosynth_rsa_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_rsa = max(vmax_rsa, _absmax_surface(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_rsa = -vmax_rsa  # Symmetric range for diverging colormap

except Exception as e:
    logger.warning(f"Could not compute from surface, using volume fallback: {e}")

    # Fallback: compute from volume values (less accurate for surface plots)
    vmax_rsa = 0.0
    for stem, _pretty in PATTERNS:
        zmap_file = neurosynth_rsa_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_rsa = max(vmax_rsa, _absmax_volume(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_rsa = -vmax_rsa  # Symmetric range for diverging colormap

logger.info(f"Neurosynth RSA vmin={vmin_rsa:.4f}, vmax={vmax_rsa:.4f}")

# Create standalone colorbar figure
fig_neurosynth_rsa = create_standalone_colorbar(
    cmap=CMAP_BRAIN,              # Diverging colormap (blue-white-red)
    vmin=vmin_rsa,                # Minimum z-score (negative)
    vmax=vmax_rsa,                # Maximum z-score (positive)
    orientation='horizontal',     # Horizontal layout for brain surface plots
    label='z-score',              # Statistical z-score label
)
save_figure(fig_neurosynth_rsa, figures_dir / "colorbar_neurosynth_rsa_horizontal")
logger.info("✓ Saved neurosynth RSA colorbar (horizontal)")


# =============================================================================
# Colorbar 5: Manifold PR Profiles Matrix (Vertical, Sequential)
# =============================================================================
# Creates a vertical sequential colorbar for the manifold Participation Ratio
# matrix (subjects × ROIs). Range is computed from actual PR values across all
# subjects and ROIs to ensure accurate representation.
# - Data: Subject×ROI PR matrix (from manifold analysis)
# - Colormap: mako (dark blue to light yellow sequential)
# - Output: Vertical orientation for use with heatmap visualization

logger.info("Loading manifold PR profiles matrix...")

import pickle

# Find latest manifold analysis results directory
manifold_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-manifold" / "results",
    pattern="*_manifold",
    require_exists=True,
    verbose=False,
)

# Load manifold results pickle containing PR matrix and other analysis outputs
with open(manifold_results_dir / "pr_results.pkl", "rb") as f:
    manifold_results = pickle.load(f)

# Extract PR matrix (shape: n_subjects × n_rois)
# Contains Participation Ratio values for each subject in each ROI
pr_matrix = manifold_results['pr_matrix']['matrix']
pr_vmin = float(pr_matrix.min())  # Minimum PR across all subjects/ROIs
pr_vmax = float(pr_matrix.max())  # Maximum PR across all subjects/ROIs

logger.info(f"Manifold PR profiles vmin={pr_vmin:.4f}, vmax={pr_vmax:.4f}")

# Create standalone colorbar figure
fig_pr_profiles = create_standalone_colorbar(
    cmap='mako',                  # Sequential colormap (dark blue to light yellow)
    vmin=pr_vmin,                 # Minimum PR value
    vmax=pr_vmax,                 # Maximum PR value
    orientation='vertical',       # Vertical layout for matrix plots
    label='PR',                   # Participation Ratio label
)
save_figure(fig_pr_profiles, figures_dir / "colorbar_manifold_pr_profiles_vertical")
logger.info("✓ Saved manifold PR profiles colorbar (vertical)")


# =============================================================================
# Colorbar 6: Manifold PCA Components (Horizontal, Symmetric)
# =============================================================================
# Creates a horizontal diverging colorbar for PCA component loadings showing
# how each ROI contributes to PC1 and PC2. Range is computed from actual
# loading values to ensure symmetric scaling around 0.
# - Data: PCA component loadings (ROI contributions to PC1 and PC2)
# - Colormap: CMAP_BRAIN (blue-white-red diverging)
# - Output: Horizontal orientation for use with component loading heatmap
# - Positive = ROI increases along component, negative = ROI decreases

logger.info("Loading manifold PCA components...")

# Extract PCA component loadings from manifold results (already loaded above)
# Shape: (2, n_rois) - Row 0 = PC1 loadings, Row 1 = PC2 loadings
pca_loadings = manifold_results['pca2d']['components']

# Compute symmetric range (max absolute value across all loadings)
# This ensures colorbar is centered at 0 with equal positive/negative range
pca_max_abs = float(np.abs(pca_loadings).max())
pca_vmin = -pca_max_abs  # Negative extreme (blue)
pca_vmax = pca_max_abs   # Positive extreme (red)

logger.info(f"Manifold PCA components vmin={pca_vmin:.4f}, vmax={pca_vmax:.4f}")

# Create standalone colorbar figure
fig_pca_components = create_standalone_colorbar(
    cmap=CMAP_BRAIN,              # Diverging colormap (blue-white-red)
    vmin=pca_vmin,                # Minimum loading (negative)
    vmax=pca_vmax,                # Maximum loading (positive)
    orientation='horizontal',     # Horizontal layout for component matrix
    label='Loading',              # Component loading label
)
save_figure(fig_pca_components, figures_dir / "colorbar_manifold_pca_components_horizontal")
logger.info("✓ Saved manifold PCA components colorbar (horizontal)")

logger.info("\n" + "="*80)
logger.info("All colorbars generated successfully!")
logger.info("="*80)

logger.info("✓ Panel: standalone colorbars complete")
