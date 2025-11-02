#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDM/heatmap plotting functions for Nature-compliant figures.

Provides:
- plot_rdm(): Standalone RDM figure
- plot_rdm_on_ax(): RDM on existing axes (for panels)
- add_rdm_category_bars(): Add colored category bars to RDM axes
- add_roi_color_legend(): Add ROI color legend
- plot_matrix_on_ax(): Generic matrix heatmap on existing axes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional


# =============================================================================
# RDM Category Bars
# =============================================================================

def add_rdm_category_bars(
    ax,
    colors: List[str],
    alphas: Optional[List[float]] = None,
    axis: str = 'both',
    params: dict = None
):
    """
    Add colored category bars along RDM axes.

    This function adds colored rectangle patches outside the RDM heatmap to indicate
    stimulus categories (e.g., checkmate status, strategy).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object containing the RDM heatmap
    colors : list of str
        Color for each item (e.g., stimulus)
    alphas : list of float, optional
        Alpha transparency for each item (default: 1.0 for all)
    axis : str, default='both'
        Which axes to add bars to: 'x', 'y', or 'both'
    params : dict, optional
        PLOT_PARAMS override

    Notes
    -----
    - Uses axis transforms (get_xaxis_transform, get_yaxis_transform) to position
      bars outside the plot area
    - Bars are placed just outside axis (offset set by params['rdm_category_bar_offset'])
    - Automatically groups consecutive items with same color+alpha
    - Uses clip_on=False to allow bars to extend beyond axis limits

    Example
    -------
    >>> colors = ['red']*10 + ['blue']*10
    >>> alphas = [0.5]*5 + [1.0]*5 + [0.5]*5 + [1.0]*5
    >>> add_rdm_category_bars(ax, colors, alphas, axis='both')
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    if alphas is None:
        alphas = [1.0] * len(colors)

    n_items = len(colors)

    # Group consecutive items with same color and alpha
    groups = []
    current_group = {'color': colors[0], 'alpha': alphas[0], 'start': 0, 'end': 1}

    for i in range(1, n_items):
        if colors[i] == current_group['color'] and alphas[i] == current_group['alpha']:
            current_group['end'] = i + 1
        else:
            groups.append(current_group)
            current_group = {'color': colors[i], 'alpha': alphas[i], 'start': i, 'end': i + 1}
    groups.append(current_group)

    # Get thickness and offset from params
    thickness = params.get('rdm_category_bar_thickness', 0.035)
    offset = params.get('rdm_category_bar_offset', -0.06)

    # Add bars along x-axis (bottom of plot)
    if axis in ['x', 'both']:
        for group in groups:
            rect_x = plt.Rectangle(
                (group['start'], offset),  # Position below x-axis
                group['end'] - group['start'],  # Width
                thickness,  # Height (thickness)
                facecolor=group['color'],
                alpha=group['alpha'],
                edgecolor='none',
                clip_on=False,
                transform=ax.get_xaxis_transform()  # Critical: use axis transform
            )
            ax.add_patch(rect_x)

    # Add bars along y-axis (left of plot)
    if axis in ['y', 'both']:
        for group in groups:
            rect_y = plt.Rectangle(
                (offset, group['start']),  # Position left of y-axis
                thickness,  # Width (thickness)
                group['end'] - group['start'],  # Height
                facecolor=group['color'],
                alpha=group['alpha'],
                edgecolor='none',
                clip_on=False,
                transform=ax.get_yaxis_transform()  # Critical: use axis transform
            )
            ax.add_patch(rect_y)


def add_roi_color_legend(
    ax,
    roi_info_df: pd.DataFrame,
    ncol: int = 4,
    loc: str = 'upper center',
    bbox_to_anchor: Tuple[float, float] = (0.5, -0.35),
    params: dict = None
):
    """
    Add a horizontal legend showing ROI group colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add legend to
    roi_info_df : pd.DataFrame
        DataFrame with columns: 'color', 'pretty_name' (or 'region_name')
    ncol : int, default=4
        Number of legend columns
    loc : str, default='upper center'
        Legend location
    bbox_to_anchor : tuple, default=(0.5, -0.35)
        Bbox anchor position (for placing below plot)
    params : dict, optional
        Plotting parameters

    Notes
    -----
    - Extracts unique color-name pairs from roi_info_df
    - Creates patch-based legend with ROI group colors
    - Useful for showing ROI category information in bar plots

    Example
    -------
    >>> add_roi_color_legend(ax, roi_info, ncol=4)
    """
    from .style import PLOT_PARAMS
    from matplotlib.patches import Patch
    if params is None:
        params = PLOT_PARAMS

    # Get unique color/name combinations
    name_col = 'pretty_name' if 'pretty_name' in roi_info_df.columns else 'region_name'
    unique_pairs = roi_info_df[['color', name_col]].drop_duplicates(subset='color')

    # Create legend handles
    handles = [Patch(facecolor=row['color'], edgecolor='black', label=row[name_col])
               for _, row in unique_pairs.iterrows()]

    # Add legend
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor,
             ncol=ncol, frameon=False,
             fontsize=params['font_size_legend'],
             title='ROI Groups', title_fontsize=params['font_size_legend'])


# =============================================================================
# RDM Plotting
# =============================================================================

def plot_rdm_on_ax(
    ax: plt.Axes,
    rdm: np.ndarray,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colors: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    show_colorbar: bool = False,
    colorbar_label: str = "Dissimilarity",
    params: dict = None
) -> None:
    """
    Plot RDM heatmap on existing axes (PRIMARY function for panels).

    CRITICAL: ALL RDMs use CMAP_BRAIN with center=0 (no exceptions).

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    rdm : np.ndarray
        RDM matrix to plot (n_items × n_items)
    title : str, optional
        Plot title (bold, 7pt)
    subtitle : str, optional
        Plot subtitle (normal weight, 6pt)
    vmin, vmax : float, optional
        Color scale limits. NEW parameter for shared color range across panels.
        If not provided, computed as symmetric around 0 based on this RDM.
    colors : list of str, optional
        Color for each item (for category bars on axes)
    alphas : list of float, optional
        Alpha transparency for each item (for category bars)
    show_colorbar : bool, default=False
        Whether to show vertical colorbar. Default False for panels
        (show only on last panel).
    colorbar_label : str, default="Dissimilarity"
        Label for colorbar
    params : dict, optional
        PLOT_PARAMS override

    Notes
    -----
    - Uses CMAP_BRAIN colormap (only colormap for RDMs)
    - center=0 (RDM 0 at center of colormap)
    - vmin/vmax symmetric around 0
    - If colors provided, shows colored category bars on left and bottom
    - For consistent coloring across multiple RDMs in panels,
      compute global vmin/vmax and pass to all plot_rdm_on_ax calls

    Examples
    --------
    >>> # Shared color range across panels
    >>> vmin, vmax = compute_symmetric_range(expert_rdm, novice_rdm, diff_rdm)
    >>> plot_rdm_on_ax(axes['A'], expert_rdm, vmin=vmin, vmax=vmax, colors=colors)
    >>> plot_rdm_on_ax(axes['B'], novice_rdm, vmin=vmin, vmax=vmax, colors=colors,
    ...                show_colorbar=True)  # Show colorbar on last panel
    """
    from .style import PLOT_PARAMS
    from .colors import CMAP_BRAIN
    from .helpers import set_axis_title
    from .helpers import style_spines, hide_ticks
    if params is None:
        params = PLOT_PARAMS

    # Compute vmin/vmax symmetric around 0 (for center=0)
    if vmin is None or vmax is None:
        max_abs = np.max(np.abs(rdm))
        vmin, vmax = -max_abs, max_abs

    # Plot heatmap with BRAIN CMAP (ONLY colormap for RDMs)
    heatmap = sns.heatmap(
        rdm,
        annot=False,
        cmap=CMAP_BRAIN,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar=show_colorbar,
        cbar_kws={
            'label': colorbar_label,
            'shrink': 0.8,
            'aspect': 20
        } if show_colorbar else None,
        ax=ax,
        square=True
    )
    ax.set_aspect('equal')

    # Style the colorbar if shown
    if show_colorbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=params['font_size_tick'])
        cbar.set_label(colorbar_label, fontsize=params['font_size_label'])

    # Add colored category bars if colors provided
    if colors is not None:
        add_rdm_category_bars(ax, colors, alphas, axis='both', params=params)

    # Set title if provided
    if title or subtitle:
        set_axis_title(ax, title, subtitle=subtitle, params=params)

    # Hide tick labels
    hide_ticks(ax, hide_x=True, hide_y=True)

    # Apply consistent styling (keep all 4 spines for heatmaps)
    style_spines(ax, visible_spines=['left', 'right', 'top', 'bottom'], params=params)


def plot_matrix_on_ax(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    cmap: str = 'mako',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    show_colorbar: bool = False,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    square: bool = False,
    params: dict = None
) -> None:
    """
    Plot a generic matrix heatmap on an existing axes with Nature-compliant styling.

    This is a general-purpose counterpart to plot_rdm_on_ax for non-RDM matrices
    (e.g., PR subject×ROI matrix, PCA loadings). It centralizes seaborn heatmap
    usage and spine/tick/title handling to keep panel scripts DRY.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    matrix : np.ndarray
        2D array to visualize
    title, subtitle : str, optional
        Axis title and subtitle (subtitle normal weight)
    cmap : str, default='mako'
        Colormap name
    vmin, vmax : float, optional
        Color scale limits. If not provided, inferred from data
    center : float, optional
        Center value for diverging colormaps (e.g., 0 for loadings)
    show_colorbar : bool, default=False
        Whether to show a colorbar
    xticklabels, yticklabels : list of str, optional
        Tick labels to display. If None, no labels are shown
    square : bool, default=False
        If True, set square aspect
    params : dict, optional
        PLOT_PARAMS override

    Notes
    -----
    - Uses centralized set_axis_title, style_spines
    - Hides tick marks by default; pass labels to show them
    """
    from .style import PLOT_PARAMS
    from .helpers import set_axis_title
    from .helpers import style_spines, hide_ticks
    if params is None:
        params = PLOT_PARAMS

    hm = sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar=show_colorbar,
        xticklabels=bool(xticklabels),
        yticklabels=bool(yticklabels),
        linewidths=0,
        linecolor='none',
        square=square,
    )

    # Apply spine styling first (this may apply MaxNLocator and change ticks)
    style_spines(ax, visible_spines=['left', 'right', 'top', 'bottom'], params=params)

    # Now set tick positions and labels deterministically so they persist
    n_rows, n_cols = matrix.shape

    if xticklabels is not None:
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(xticklabels, rotation=30, ha='right', fontsize=params['font_size_tick'])
    else:
        hide_ticks(ax, hide_x=True, hide_y=False)

    if yticklabels is not None:
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(
            yticklabels,
            fontsize=params['font_size_tick'],
            rotation=0,          # <-- make them horizontal
            va='center',         # vertical alignment (middle)
            ha='right'           # horizontal alignment (optional: 'center' or 'right')
        )

    else:
        hide_ticks(ax, hide_x=False, hide_y=True)

    if title or subtitle:
        set_axis_title(ax, title=title, subtitle=subtitle, params=params)


def plot_rdm(
    rdm: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    colors: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    show_colorbar: bool = True,
    colorbar_label: str = "Dissimilarity",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    params: dict = None
) -> plt.Figure:
    """
    Plot RDM as standalone figure with colored category bars and colorbar.

    CRITICAL: ALL RDMs use CMAP_BRAIN with center=0 (no exceptions).

    Parameters
    ----------
    rdm : np.ndarray
        RDM matrix to plot (n_items × n_items)
    title : str
        Plot title (bolded)
    subtitle : str, optional
        Plot subtitle (normal weight)
    output_path : Path, optional
        Path to save figure (if None, figure is not saved)
    colors : list of str, optional
        Color for each item (for category bars on axes)
    alphas : list of float, optional
        Alpha transparency for each item (for category bars)
    show_colorbar : bool, default=True
        Whether to show vertical colorbar on right side
    colorbar_label : str, default="Dissimilarity"
        Label for colorbar
    vmin, vmax : float, optional
        Color scale limits. If not provided, computed as symmetric around 0
        based on max absolute value in this RDM. For consistent coloring across
        multiple RDMs, compute global max and pass explicitly.
    params : dict, optional
        Plotting parameters dictionary

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object

    Notes
    -----
    - Uses CMAP_BRAIN colormap (only colormap for RDMs)
    - center=0 (RDM 0 at center of colormap)
    - vmin/vmax symmetric around 0
    - If colors provided, shows colored category bars on left and bottom
    - Colorbar on right (default, can be disabled with show_colorbar=False)

    Example
    -------
    >>> from common.plotting import plot_rdm, CMAP_BRAIN
    >>> rdm = np.random.randn(20, 20)
    >>> colors = ['red'] * 10 + ['blue'] * 10
    >>> fig = plot_rdm(rdm, "Example RDM", colors=colors, output_path=Path("rdm.pdf"))
    """
    from .style import PLOT_PARAMS, apply_nature_rc, figure_size
    if params is None:
        params = PLOT_PARAMS

    # Create figure
    figsize = figure_size(columns=2, height_mm=100)
    apply_nature_rc(params)
    fig, ax = plt.subplots(figsize=figsize, facecolor=params['facecolor'])

    # Plot using core function
    plot_rdm_on_ax(
        ax=ax,
        rdm=rdm,
        title=title,
        subtitle=subtitle,
        vmin=vmin,
        vmax=vmax,
        colors=colors,
        alphas=alphas,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label,
        params=params
    )

    # Save if path provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                   facecolor=params['facecolor'], dpi=params['dpi'])
        plt.close(fig)

    return fig
