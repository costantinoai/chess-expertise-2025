"""
Utilities for simulated manifold and linear separability visualizations.

Generates synthetic binary classification datasets with controllable class separation
and redundancy, then produces:
- 3D PCA manifold plots with optional ribbon curves
- SVM hyperplane visualizations
- Observation × voxel heatmaps
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.interpolate import griddata, splprep, splev
from matplotlib.colors import LinearSegmentedColormap

from common import (
    PLOT_PARAMS,
    apply_nature_rc,
    figure_size,
    save_figure,
    COLORS_CHECKMATE_NONCHECKMATE,
    CMAP_BRAIN,
    plot_matrix_on_ax,
)


def generate_simulation(
    n_samples: int = 40,
    n_features: int = 24,
    class_sep: float = 2.0,
    n_redundant: int = 6,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary classification dataset for manifold visualization.

    Uses sklearn's make_classification to create a dataset with controlled
    class separation and feature redundancy.

    Parameters
    ----------
    n_samples : int, default=40
        Number of samples (observations).
    n_features : int, default=24
        Total number of features (voxels).
    class_sep : float, default=2.0
        Class separation factor (larger = easier separation).
    n_redundant : int, default=6
        Number of redundant features (linear combinations of informative ones).
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix, normalized to [0, 1] range.
    y : ndarray, shape (n_samples,)
        Binary class labels (1 or 0).
    """
    # Ensure feature counts are valid
    n_informative = min(max(2, n_features // 2), n_features)
    n_redundant = max(0, min(n_redundant, n_features - n_informative))
    n_repeated = max(0, n_features - n_informative - n_redundant)

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_clusters_per_class=1,
        n_classes=2,
        class_sep=class_sep,
        shuffle=False,
        random_state=random_seed,
    )

    # Normalize to [0, 1] range for visualization consistency
    X = X.astype(float)
    X_min, X_max = X.min(), X.max()
    if X_max > X_min:
        X = (X - X_min) / (X_max - X_min)

    return X, y.astype(int)


def compute_participation_ratio(
    X: np.ndarray
) -> Tuple[float, PCA, np.ndarray, np.ndarray]:
    """
    Compute participation ratio (PR) from PCA eigenvalues.

    The participation ratio quantifies the effective dimensionality:
    PR = (sum of eigenvalues)^2 / (sum of squared eigenvalues)

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    pr : float
        Participation ratio.
    pca : PCA
        Fitted PCA object.
    eigvals : ndarray
        PCA eigenvalues (explained variances).
    explained_ratio : ndarray
        Variance explained ratio per component.
    """
    pca = PCA()
    pca.fit(X)
    eigvals = pca.explained_variance_

    # Calculate PR: (sum λ)^2 / sum(λ^2)
    pr = (
        float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))
        if eigvals.size
        else float('nan')
    )

    return pr, pca, eigvals, pca.explained_variance_ratio_


def _pca_embed_3d(X: np.ndarray) -> np.ndarray:
    """
    Project data onto first 3 principal components.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    ndarray, shape (n_samples, 3)
        3D PCA embedding.
    """
    pca = PCA()
    emb = pca.fit_transform(X)
    return emb[:, :3]


def plot_3d_manifold_surface(
    X: np.ndarray,
    y: np.ndarray,
    add_ribbon: bool = True,
    title: Optional[str] = "3D Manifold",
    output_path: Optional[Path] = None,
    params: dict | None = None,
) -> plt.Figure:
    """
    Render a 3D surface fitted to the first three PCA dimensions.

    The surface is interpolated over the PC1-PC2 plane with PC3 as height,
    colored by the brain colormap. Points are colored by class using the
    checkmate/non-checkmate palette. Optionally adds a ribbon curve along PC1.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary class labels (1 = checkmate, 0 = non-checkmate).
    add_ribbon : bool, default=True
        Whether to add a spline ribbon curve along PC1.
    title : str, optional
        Plot title.
    output_path : Path, optional
        If provided, save figure to this path.
    params : dict, optional
        PLOT_PARAMS override.

    Returns
    -------
    plt.Figure
        The created 3D figure.
    """
    if params is None:
        params = PLOT_PARAMS
    apply_nature_rc(params)

    # Project to 3D PCA space
    embedding_3d = _pca_embed_3d(X)

    # Create dense grid over PC1-PC2 plane
    grid_x, grid_y = np.mgrid[
        embedding_3d[:, 0].min():embedding_3d[:, 0].max():40j,
        embedding_3d[:, 1].min():embedding_3d[:, 1].max():40j,
    ]

    # Interpolate PC3 values onto grid (cubic for smoothness)
    grid_z = griddata(
        (embedding_3d[:, 0], embedding_3d[:, 1]),
        embedding_3d[:, 2],
        (grid_x, grid_y),
        method='cubic'
    )

    # Normalize PC3 for surface coloring with brain colormap
    norm = (grid_z - np.nanmin(grid_z)) / (np.nanmax(grid_z) - np.nanmin(grid_z))
    surface_colors = CMAP_BRAIN(norm)

    # Create figure with 3D axes
    fig_w, fig_h = figure_size(columns=1, height_mm=80)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection='3d')

    # Plot semi-transparent interpolated surface
    ax.plot_surface(
        grid_x, grid_y, grid_z,
        facecolors=surface_colors,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.2
    )

    # Optional ribbon: spline curve along sorted PC1
    if add_ribbon:
        sorted_idx = np.argsort(embedding_3d[:, 0])
        x_sorted = embedding_3d[sorted_idx, 0]
        y_sorted = embedding_3d[sorted_idx, 1]
        z_sorted = embedding_3d[sorted_idx, 2]

        # Fit B-spline through sorted points
        tck, _ = splprep([x_sorted, y_sorted, z_sorted], s=3)
        ribbon = splev(np.linspace(0, 1, 100), tck)

        # Draw ribbon as dashed black line
        ax.plot(
            ribbon[0], ribbon[1], ribbon[2],
            color='black',
            linewidth=2,
            linestyle='--'
        )

    # Scatter actual data points, colored by class
    point_colors = np.where(
        y == 1,
        COLORS_CHECKMATE_NONCHECKMATE['checkmate'],
        COLORS_CHECKMATE_NONCHECKMATE['non_checkmate']
    )
    ax.scatter(
        embedding_3d[:, 0],
        embedding_3d[:, 1],
        embedding_3d[:, 2],
        c=point_colors,
        s=60,
        edgecolor='k',
        linewidth=0.5
    )

    # Style: clean axes (no ticks, transparent panes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)

    if title:
        ax.set_title(
            title,
            fontsize=params['font_size_title'],
            fontweight='bold',
            pad=params.get('title_pad', 10.0)
        )

    if output_path is not None:
        save_figure(fig, Path(output_path))

    return fig


def plot_decoding_hyperplane_3d(
    X: np.ndarray,
    y: np.ndarray,
    title: Optional[str] = "Decoding: SVM Hyperplane",
    output_path: Optional[Path] = None,
    params: dict | None = None,
) -> plt.Figure:
    """
    Train a linear SVM on the first three PCA dimensions and visualize the decision plane.

    The hyperplane is computed from the SVM coefficients:
    w·x + b = 0  =>  z = -(w[0]*x + w[1]*y + b) / w[2]

    Points are colored by class using the checkmate/non-checkmate palette.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary class labels (1 = checkmate, 0 = non-checkmate).
    title : str, optional
        Plot title.
    output_path : Path, optional
        If provided, save figure to this path.
    params : dict, optional
        PLOT_PARAMS override.

    Returns
    -------
    plt.Figure
        The created 3D figure with hyperplane and points.
    """
    if params is None:
        params = PLOT_PARAMS
    apply_nature_rc(params)

    # Project to 3D PCA space
    emb3d = _pca_embed_3d(X)

    # Train linear SVM
    clf = SVC(kernel='linear')
    clf.fit(emb3d, y)
    w = clf.coef_[0]  # Normal vector to hyperplane
    b = clf.intercept_[0]  # Bias

    # Create grid over PC1-PC2 plane
    x_range = np.linspace(emb3d[:, 0].min(), emb3d[:, 0].max(), 10)
    y_range = np.linspace(emb3d[:, 1].min(), emb3d[:, 1].max(), 10)
    xx, yy = np.meshgrid(x_range, y_range)

    # Compute hyperplane height: z = -(w[0]*x + w[1]*y + b) / w[2]
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]

    # Create 3D figure
    fig_w, fig_h = figure_size(columns=1, height_mm=80)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection='3d')

    # Plot decision plane (semi-transparent blue)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightblue', edgecolor=None)

    # Scatter data points colored by class
    point_colors = np.where(
        y == 1,
        COLORS_CHECKMATE_NONCHECKMATE['checkmate'],
        COLORS_CHECKMATE_NONCHECKMATE['non_checkmate']
    )
    ax.scatter(
        emb3d[:, 0],
        emb3d[:, 1],
        emb3d[:, 2],
        c=point_colors,
        s=60,
        edgecolor='k',
        linewidth=0.5
    )

    # Style: clean axes (no ticks, transparent panes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)

    if title:
        ax.set_title(
            title,
            fontsize=params['font_size_title'],
            fontweight='bold',
            pad=params.get('title_pad', 10.0)
        )

    if output_path is not None:
        save_figure(fig, Path(output_path))

    return fig


def plot_obs_voxel_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    max_voxels: int = 20,
    title: Optional[str] = "Data Matrix: Observations × Features",
    output_path: Optional[Path] = None,
    params: dict | None = None,
) -> plt.Figure:
    """
    Plot observation × voxel (feature) heatmap with class-based diverging colormap.

    Observations are sorted by class and displayed as rows. Feature values
    are encoded with a diverging colormap: checkmate class (positive) in
    green and non-checkmate class (negative) in red.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix, normalized to [0, 1] range.
    y : ndarray, shape (n_samples,)
        Binary class labels (1 = checkmate, 0 = non-checkmate).
    max_voxels : int, default=20
        Maximum number of features (voxels) to display.
    title : str, optional
        Plot title.
    output_path : Path, optional
        If provided, save figure to this path.
    params : dict, optional
        PLOT_PARAMS override.

    Returns
    -------
    plt.Figure
        The generated heatmap figure.
    """
    if params is None:
        params = PLOT_PARAMS
    apply_nature_rc(params)

    # Limit features to max_voxels for readability
    X_display = X[:, :max_voxels]
    n_obs, n_feat = X_display.shape

    # Sort observations by class for visual clarity
    sort_idx = np.argsort(y)
    X_sorted = X_display[sort_idx]
    y_sorted = y[sort_idx]

    # Create signed matrix: positive for checkmate, negative for non-checkmate
    # This allows us to use a diverging colormap
    heatmap_matrix = np.where(y_sorted[:, None] == 1, X_sorted, -X_sorted)

    # Diverging colormap: non-checkmate (red) → white → checkmate (green)
    cmap_diverging = LinearSegmentedColormap.from_list(
        'obs_voxel_class',
        [
            COLORS_CHECKMATE_NONCHECKMATE['non_checkmate'],
            'white',
            COLORS_CHECKMATE_NONCHECKMATE['checkmate'],
        ]
    )

    # Create tall figure for observations as rows
    fig_w, fig_h = figure_size(columns=1, height_mm=120)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Plot heatmap using centralized utility
    plot_matrix_on_ax(
        ax=ax,
        matrix=heatmap_matrix,
        cmap=cmap_diverging,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        show_colorbar=False,
        params=params,
    )

    # Style axes
    ax.set_xlabel('Voxels', fontsize=params['font_size_label'])
    ax.set_ylabel('Stimuli', fontsize=params['font_size_label'])
    ax.tick_params(labelsize=params['font_size_tick'])

    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Gray spines
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(params['plot_linewidth'])

    if title:
        ax.set_title(
            title,
            fontsize=params['font_size_title'],
            fontweight='bold',
            pad=params.get('title_pad', 10.0)
        )

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, Path(output_path))

    return fig
