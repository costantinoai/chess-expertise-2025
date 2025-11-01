"""
Dataset visualization utilities module.

This module provides functions for creating chess dataset grids and
simulated manifold visualizations for supplementary materials.
"""

# Dataset grid utilities
from .dataset_grid_utils import (
    create_chess_dataset_grid,
    create_chess_dataset_grid_bordered,
)

# Simulation and manifold utilities
from .simulation_utils import (
    generate_simulation,
    compute_participation_ratio,
    plot_3d_manifold_surface,
    plot_decoding_hyperplane_3d,
    plot_obs_voxel_heatmap,
)

__all__ = [
    # Dataset grid functions
    'create_chess_dataset_grid',
    'create_chess_dataset_grid_bordered',
    # Simulation functions
    'generate_simulation',
    'compute_participation_ratio',
    'plot_3d_manifold_surface',
    'plot_decoding_hyperplane_3d',
    'plot_obs_voxel_heatmap',
]
