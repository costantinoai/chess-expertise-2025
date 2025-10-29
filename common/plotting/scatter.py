#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D embedding and scatter plot functions for Nature-compliant figures.

Provides:
- plot_2d_embedding(): Standalone 2D embedding figure
- plot_2d_embedding_on_ax(): 2D embedding on existing axes (for panels)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict


# =============================================================================
# 2D Embedding Plotting
# =============================================================================

def plot_2d_embedding_on_ax(
    ax: plt.Axes,
    coords: np.ndarray,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    point_colors: Optional[List[str]] = None,
    point_alphas: Optional[List[float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    fill: Optional[Dict] = None,
    params: dict = None,
    hide_tick_marks: bool = False
) -> None:
    """
    Plot 2D embedding on existing axes (PRIMARY function for panels).

    Style matches behavioral MDS plots: hidden ticks, all spines visible,
    centralized title/subtitle. Optionally adds axis labels and a background
    fill (e.g., decision boundary) via contourf.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    coords : np.ndarray, shape (n_samples, 2)
        Precomputed 2D coordinates to plot.
    title : str, optional
        Plot title (bold, 7pt)
    subtitle : str, optional
        Plot subtitle (normal weight, 6pt)
    point_colors : list of str, optional
        Color for each point. If None, uses a default single color.
    point_alphas : list of float, optional
        Alpha transparency for each point. If None, uses 1.0 for all.
    xlim, ylim : Tuple[float, float], optional
        Axis limits. NEW parameters for shared ranges across panels.
    x_label, y_label : str, optional
        Axis labels to display. Ticks remain hidden for style consistency.
    fill : dict, optional
        Optional background fill, typically a decision boundary. Expected keys:
          - 'xx': meshgrid X array
          - 'yy': meshgrid Y array
          - 'Z':  2D array of values/classes for contourf
          - 'colors': list of colors for levels (optional)
          - 'alpha': float alpha for fill (optional, default 0.15)
          - 'levels': list/tuple of contour levels (optional)
    params : dict, optional
        PLOT_PARAMS override

    Examples
    --------
    >>> # Shared axis limits across panels
    >>> xlim = (-2, 2)
    >>> ylim = (-2, 2)
    >>> plot_2d_embedding_on_ax(axes['A'], expert_coords, xlim=xlim, ylim=ylim)
    >>> plot_2d_embedding_on_ax(axes['B'], novice_coords, xlim=xlim, ylim=ylim)
    """
    from .style import PLOT_PARAMS
    from .helpers import set_axis_title
    from .helpers import style_spines, hide_ticks
    if params is None:
        params = PLOT_PARAMS

    # Optional background fill (e.g., classifier decision boundary)
    if fill is not None:
        xx = fill.get('xx')
        yy = fill.get('yy')
        Z = fill.get('Z')
        if xx is not None and yy is not None and Z is not None:
            colors = fill.get('colors')
            alpha = fill.get('alpha', 0.15)
            levels = fill.get('levels', [0, 0.5, 1])
            ax.contourf(xx, yy, Z, alpha=alpha, levels=levels, colors=colors)

    # Point styling
    n = coords.shape[0]
    if point_colors is None:
        point_colors = ['#333333'] * n
    if point_alphas is None:
        point_alphas = [1.0] * n

    # Plot points with larger marker size for visibility
    for (x, y), c, a in zip(coords, point_colors, point_alphas):
        ax.scatter(x, y, color=c, marker='o', alpha=a,
                   s=params['marker_size'],
                   edgecolors='black', linewidths=params['plot_linewidth'])

    # Apply axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Axis labels (ticks remain hidden per style)
    if x_label or y_label:
        from .helpers import label_axes
        label_axes(ax, xlabel=x_label, ylabel=y_label, params=params)

    # Set title if provided
    if title or subtitle:
        set_axis_title(ax, title, subtitle=subtitle, params=params)

    # Apply consistent styling (keep all 4 spines for embeddings)
    style_spines(ax, visible_spines=['left', 'right', 'top', 'bottom'], params=params)

    # Hide tick labels but keep axes (AFTER style_spines to avoid being overridden)
    hide_ticks(ax, hide_x=True, hide_y=True)

    # Hide tick marks if requested
    if hide_tick_marks:
        ax.tick_params(axis='both', which='both', length=0)


def plot_2d_embedding(
    coords: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    point_colors: Optional[List[str]] = None,
    point_alphas: Optional[List[float]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    fill: Optional[Dict] = None,
    params: dict = None
) -> plt.Figure:
    """
    Plot a 2D embedding scatter as standalone figure using precomputed coordinates.

    Style matches behavioral MDS plots: hidden ticks, all spines visible,
    centralized title/subtitle. Optionally adds axis labels and a background
    fill (e.g., decision boundary) via contourf.

    Parameters
    ----------
    coords : np.ndarray, shape (n_samples, 2)
        Precomputed 2D coordinates to plot.
    title : str
        Plot title (bold).
    subtitle : str, optional
        Plot subtitle (normal weight).
    output_path : Path, optional
        Path to save the figure (saved if provided).
    point_colors : list of str, optional
        Color for each point. If None, uses a default single color.
    point_alphas : list of float, optional
        Alpha transparency for each point. If None, uses 1.0 for all.
    x_label, y_label : str, optional
        Axis labels to display. Ticks remain hidden for style consistency.
    fill : dict, optional
        Optional background fill, typically a decision boundary. Expected keys:
          - 'xx': meshgrid X array
          - 'yy': meshgrid Y array
          - 'Z':  2D array of values/classes for contourf
          - 'colors': list of colors for levels (optional)
          - 'alpha': float alpha for fill (optional, default 0.15)
          - 'levels': list/tuple of contour levels (optional)
    params : dict, optional
        Centralized plotting parameters.

    Returns
    -------
    plt.Figure
        The created figure.

    Examples
    --------
    >>> coords = np.random.randn(20, 2)
    >>> colors = ['red'] * 10 + ['blue'] * 10
    >>> fig = plot_2d_embedding(coords, "MDS Projection", point_colors=colors,
    ...                          output_path=Path("embedding.pdf"))
    """
    from .style import PLOT_PARAMS, apply_nature_rc, figure_size
    if params is None:
        params = PLOT_PARAMS

    # Create figure
    figsize = figure_size(columns=2, height_mm=100)
    apply_nature_rc(params)
    fig, ax = plt.subplots(figsize=figsize, facecolor=params['facecolor'])

    # Plot using core function
    plot_2d_embedding_on_ax(
        ax=ax,
        coords=coords,
        title=title,
        subtitle=subtitle,
        point_colors=point_colors,
        point_alphas=point_alphas,
        x_label=x_label,
        y_label=y_label,
        fill=fill,
        params=params
    )

    # Save if path provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                   facecolor=params['facecolor'], dpi=params['dpi'])
        plt.close(fig)

    return fig
