"""
Generate Dataset Visualization Supplementary Figures
====================================================

Creates publication-ready supplementary figures visualizing the chess dataset
and simulated neural manifold examples. Uses standardized plotting primitives
and module-level visualization utilities. All figures are saved as both SVG
and PDF for flexible integration into manuscripts.

Figures Produced
----------------

1. Chess Dataset Grid
   - File: figures/dataset_grid_bordered.svg (and .pdf)
   - Content: Complete grid of all chess board positions (5 columns)
   - Data source: Stimuli metadata (stimuli.tsv) and position images
   - Shows full experimental stimulus set with borders

2. Simulated 3D Manifold with Ribbon
   - File: figures/manifold_3d_ribbon.svg (and .pdf)
   - Content: 3D PCA visualization of simulated neural data with decision ribbon
   - Shows manifold structure with Participation Ratio (PR) annotation
   - Simulated data: 40 samples, 15 features, class separation = 3.0

3. Simulated Linear Separability (3D Hyperplane)
   - File: figures/separability_hyperplane_3d.svg (and .pdf)
   - Content: 3D visualization of SVM decision hyperplane
   - Shows linear decision boundary separating two classes
   - Same simulated data as manifold figure

4. Observations × Voxels Heatmap
   - File: figures/obs_voxel_heatmap.svg (and .pdf)
   - Content: Heatmap showing neural responses (observations × voxels)
   - Visualizes simulated ROI data structure (max 15 voxels displayed)

5. ROI Group Legends (4 variants)
   - Files: roi_group_legend_2row.svg, roi_group_legend_1row.svg
            roi_group_legend_2row_colorblind.svg, roi_group_legend_1row_colorblind.svg
   - Content: Standalone legends showing ROI group families with colors
   - Variants: 1-row vs 2-row layouts, normal vs colorblind-safe colors

Inputs
------
- stimuli.tsv (from CONFIG['STIMULI_FILE']) for chess board position metadata
- Chess board position images (same directory as stimuli.tsv)
- Simulated data generated in-script using modules.generate_simulation()

Dependencies
------------
- modules.dataset_viz utilities (create_chess_dataset_grid_bordered, etc.)
- modules.simulation utilities (generate_simulation, compute_participation_ratio, etc.)
- common.plotting primitives (apply_nature_rc, save_figure, create_roi_group_legend)

Usage
-----
python chess-supplementary/dataset-viz/91_plot_dataset_viz.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Add parent (repo root) to sys.path for 'common' and 'modules'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import CONFIG, apply_nature_rc, save_figure, create_roi_group_legend
from common.logging_utils import setup_analysis

# Import dataset visualization utilities from modules
from modules import (
    create_chess_dataset_grid_bordered,  # Chess position grid with borders
    generate_simulation,                  # Simulated neural data generator
    compute_participation_ratio,          # PR calculation for manifold
    plot_3d_manifold_surface,             # 3D PCA with ribbon visualization
    plot_decoding_hyperplane_3d,          # 3D SVM hyperplane visualization
    plot_obs_voxel_heatmap,               # Observations × voxels heatmap
)

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()


# =============================================================================
# Configuration and Setup
# =============================================================================
# Initialize logging and create output directories for supplementary figures.
# All figures will be saved to results/<timestamp>_dataset_viz/figures/

analysis_name = "dataset_viz"
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
# Figure 1: Chess Dataset Grid (Complete Stimulus Set)
# =============================================================================
# Creates a grid visualization showing all chess board positions used in the
# experiment. Positions are arranged in a 5-column grid with borders.
# - Data: Chess board position images from stimuli directory
# - Layout: 5 columns, rows determined by total number of positions
# - Visual: Each position shown with border for clarity

# Resolve images directory from stimuli.tsv location in CONFIG
# stimuli.tsv and images are expected to be in the same directory
stimuli_file = Path(CONFIG['STIMULI_FILE'])
images_dir = stimuli_file.parent if stimuli_file.exists() else None

# Create chess dataset grid with borders around each position
fig_grid = create_chess_dataset_grid_bordered(
    images_dir=images_dir,           # Directory containing position images
    n_cols=5,                        # Number of columns in grid
    title="Full Dataset",           # Figure title
)
save_figure(fig_grid, figures_dir / "dataset_grid_bordered")
logger.info("✓ Saved dataset grid figure")


# =============================================================================
# Figure 2: Simulated 3D Manifold with Ribbon
# =============================================================================
# Creates a 3D visualization of simulated neural manifold data using PCA.
# Shows how neural population activity forms a low-dimensional manifold
# structure that can be visualized as a curved surface/ribbon in 3D space.
# - Data: Simulated neural responses (40 samples, 15 features, 2 classes)
# - Visualization: PCA projection to 3D with manifold ribbon
# - Annotation: Participation Ratio (PR) indicating dimensionality

# Generate simulated neural data with controlled properties
X, y = generate_simulation(
    n_samples=40,        # Number of observations (trials/conditions)
    n_features=15,       # Number of features (voxels/neurons)
    class_sep=3.0,       # Class separation (higher = more separable)
    n_redundant=10,      # Number of redundant features (creates low-dim structure)
    random_seed=42       # Reproducible random seed
)

# Compute Participation Ratio (measure of manifold dimensionality)
# PR ≈ 1 means low-dimensional, PR ≈ n_features means high-dimensional
pr, _, _, _ = compute_participation_ratio(X)

# Create 3D manifold visualization with ribbon showing data structure
fig_manifold = plot_3d_manifold_surface(
    X, y,                              # Simulated data and class labels
    add_ribbon=False,                  # Ribbon style (False = surface only)
    title=f"Manifold + Ribbon (PR≈{pr:.1f})",  # Title with PR annotation
)
save_figure(fig_manifold, figures_dir / "manifold_3d_ribbon")
logger.info("✓ Saved 3D manifold + ribbon figure")


# =============================================================================
# Figure 3: Simulated Linear Separability (3D Hyperplane)
# =============================================================================
# Creates a 3D visualization showing how a linear classifier (SVM) separates
# two classes in neural space. The hyperplane represents the decision boundary.
# - Data: Same simulated data as manifold figure (X, y from above)
# - Visualization: 3D scatter plot with SVM decision hyperplane
# - Purpose: Illustrates linear decodability of class information

# Create 3D hyperplane visualization using same simulated data
fig_sep = plot_decoding_hyperplane_3d(
    X, y,                              # Simulated data and class labels
    title="Decoding: SVM Hyperplane",  # Figure title
)
save_figure(fig_sep, figures_dir / "separability_hyperplane_3d")
logger.info("✓ Saved 3D separability figure")


# =============================================================================
# Figure 4: Observations × Voxels Heatmap
# =============================================================================
# Creates a heatmap showing the structure of neural data matrix
# (observations × voxels). Useful for visualizing data organization.
# - Data: Same simulated data (X, y from above)
# - Layout: Rows = observations, Columns = voxels (max 15 displayed)
# - Purpose: Shows how neural data is organized in matrix form

# Create observations × voxels heatmap
fig_heatmap = plot_obs_voxel_heatmap(
    X, y,                    # Simulated data and class labels
    max_voxels=15,           # Maximum number of voxels to display
    title="ROI voxels",      # Figure title
)
save_figure(fig_heatmap, figures_dir / "obs_voxel_heatmap")
logger.info("✓ Saved obs × voxel heatmap figure")


# =============================================================================
# Figure 5: ROI Group Legends (Standalone, Multiple Variants)
# =============================================================================
# Creates standalone legend figures showing all ROI group families with their
# colors. Multiple variants are generated for different use cases:
# - Layout: 1-row (horizontal) vs 2-row (compact)
# - Colors: Normal vs colorblind-safe palettes
# These legends can be embedded in other figures or used standalone.

# -----------------------------------------------------------------------------
# Variant 1: 2-row layout, normal colors
# -----------------------------------------------------------------------------
fig_legend_2row = create_roi_group_legend(
    single_row=False,        # Use 2-row layout for compact display
    colorblind=False,        # Use standard color palette
)
save_figure(fig_legend_2row, figures_dir / "roi_group_legend_2row")
logger.info("✓ Saved ROI group legend (2 rows, normal colors)")

# -----------------------------------------------------------------------------
# Variant 2: 2-row layout, colorblind-safe colors
# -----------------------------------------------------------------------------
fig_legend_2row_cb = create_roi_group_legend(
    single_row=False,        # Use 2-row layout for compact display
    colorblind=True,         # Use colorblind-safe palette
)
save_figure(fig_legend_2row_cb, figures_dir / "roi_group_legend_2row_colorblind")
logger.info("✓ Saved ROI group legend (2 rows, colorblind colors)")

# -----------------------------------------------------------------------------
# Variant 3: 1-row layout, normal colors
# -----------------------------------------------------------------------------
fig_legend_1row = create_roi_group_legend(
    single_row=True,         # Use 1-row horizontal layout
    colorblind=False,        # Use standard color palette
)
save_figure(fig_legend_1row, figures_dir / "roi_group_legend_1row")
logger.info("✓ Saved ROI group legend (1 row, normal colors)")

# -----------------------------------------------------------------------------
# Variant 4: 1-row layout, colorblind-safe colors
# -----------------------------------------------------------------------------
fig_legend_1row_cb = create_roi_group_legend(
    single_row=True,         # Use 1-row horizontal layout
    colorblind=True,         # Use colorblind-safe palette
)
save_figure(fig_legend_1row_cb, figures_dir / "roi_group_legend_1row_colorblind")
logger.info("✓ Saved ROI group legend (1 row, colorblind colors)")

logger.info("✓ Panel: dataset visualization figures complete")
