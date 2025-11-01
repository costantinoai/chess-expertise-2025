"""
Generate supplementary dataset visualizations:
- Full chess board dataset grid (from stimuli metadata + images dir)
- Simulated 2D manifold (PCA) with decision boundary
- Simulated linear separability (decision score histograms)

Saves publication-ready figures using the centralized plotting style.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import CONFIG, apply_nature_rc, save_figure, CMAP_BRAIN, create_standalone_colorbar, create_roi_group_legend
from common.logging_utils import setup_analysis

# Import dataset visualization utilities from modules
from modules import (
    create_chess_dataset_grid_bordered,
    generate_simulation,
    compute_participation_ratio,
    plot_3d_manifold_surface,
    plot_decoding_hyperplane_3d,
    plot_obs_voxel_heatmap,
)


# =============================================================================
# Setup results directory and style
# =============================================================================

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

apply_nature_rc()


# =============================================================================
# 1) Full dataset grid
# =============================================================================

# Try to resolve images_dir from stimuli.tsv location
stimuli_file = Path(CONFIG['STIMULI_FILE'])
images_dir = stimuli_file.parent if stimuli_file.exists() else None
fig_grid = create_chess_dataset_grid_bordered(
    images_dir=images_dir,
    n_cols=5,
    title="Full Dataset",
)
save_figure(fig_grid, figures_dir / "dataset_grid_bordered")
logger.info("✓ Saved dataset grid figure")


# =============================================================================
# 2) Simulated 3D manifold (PCA) with ribbon
# =============================================================================

X, y = generate_simulation(n_samples=40, n_features=15, class_sep=3.0, n_redundant=10, random_seed=42)
pr, _, _, _ = compute_participation_ratio(X)

fig_manifold = plot_3d_manifold_surface(
    X, y,
    add_ribbon=False,
    title=f"Manifold + Ribbon (PR≈{pr:.1f})",
)
save_figure(fig_manifold, figures_dir / "manifold_3d_ribbon")
logger.info("✓ Saved 3D manifold + ribbon figure")


# =============================================================================
# 3) Simulated linear separability (3D hyperplane)
# =============================================================================

fig_sep = plot_decoding_hyperplane_3d(
    X, y,
    title="Decoding: SVM Hyperplane",
)
save_figure(fig_sep, figures_dir / "separability_hyperplane_3d")
logger.info("✓ Saved 3D separability figure")


# =============================================================================
# 4) Observations × Voxels heatmap
# =============================================================================

fig_heatmap = plot_obs_voxel_heatmap(
    X, y,
    max_voxels=15,
    title="ROI voxels",
)
save_figure(fig_heatmap, figures_dir / "obs_voxel_heatmap")
logger.info("✓ Saved obs × voxel heatmap figure")


# =============================================================================
# 5) ROI Group Legend (standalone) - Normal and Colorblind versions
# =============================================================================

# Normal colors, 2 rows
fig_legend_2row = create_roi_group_legend(
    single_row=False,
    colorblind=False,
)
save_figure(fig_legend_2row, figures_dir / "roi_group_legend_2row")
logger.info("✓ Saved ROI group legend (2 rows, normal colors)")

# Colorblind colors, 2 rows
fig_legend_2row_cb = create_roi_group_legend(
    single_row=False,
    colorblind=True,
)
save_figure(fig_legend_2row_cb, figures_dir / "roi_group_legend_2row_colorblind")
logger.info("✓ Saved ROI group legend (2 rows, colorblind colors)")

# Normal colors, 1 row
fig_legend_1row = create_roi_group_legend(
    single_row=True,
    colorblind=False,
)
save_figure(fig_legend_1row, figures_dir / "roi_group_legend_1row")
logger.info("✓ Saved ROI group legend (1 row, normal colors)")

# Colorblind colors, 1 row
fig_legend_1row_cb = create_roi_group_legend(
    single_row=True,
    colorblind=True,
)
save_figure(fig_legend_1row_cb, figures_dir / "roi_group_legend_1row_colorblind")
logger.info("✓ Saved ROI group legend (1 row, colorblind colors)")
