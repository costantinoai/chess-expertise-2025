#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-panel figure composition for Nature-compliant figures.

Provides:
- make_panel_grid(): Create lettered panel layouts
- set_axis_title(): Set title and subtitle above plot
- add_panel_label(): Add lowercase panel letters (internal helper)
"""

import matplotlib.pyplot as plt
from typing import Union, List, Dict, Optional, Tuple, Literal


# =============================================================================
# Multi-Panel Grid Creation
# =============================================================================

def make_panel_grid(
    layout: Union[str, List[List[str]]],
    columns: Literal[1, 2] = 2,
    height_mm: Optional[float] = None,
    width_ratios: Optional[List[float]] = None,
    height_ratios: Optional[List[float]] = None,
    hspace_mm: float = 5.0,
    wspace_mm: float = 5.0,
    add_panel_labels: bool = True,
    params: dict = None
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Create Nature-compliant multi-panel figure.

    Parameters
    ----------
    layout : str or list
        Panel arrangement. Examples:
        - "AB\\nCD" → 2×2 grid (4 panels)
        - "A\\nB\\nC" → 3×1 vertical (3 panels)
        - [['A', 'A'], ['B', 'C']] → A spans top row, B and C below
    columns : 1 or 2, default=2
        Figure width (89mm or 183mm)
    height_mm : float, optional
        Total figure height (max 170mm)
    width_ratios, height_ratios : list, optional
        Relative panel sizes
    hspace_mm, wspace_mm : float, default=5.0
        Panel spacing in millimeters
    add_panel_labels : bool, default=True
        Add lowercase letters ("a", "b", "c"...) in top-left corner
    params : dict, optional
        PLOT_PARAMS override

    Returns
    -------
    fig : plt.Figure
        Figure at final print size
    axes_map : dict
        {panel_key: Axes} - Access as axes_map['A'], axes_map['B'], etc.

    Examples
    --------
    >>> # Two-panel vertical
    >>> fig, axes = make_panel_grid("A\\nB", columns=2, height_ratios=[1, 1.5])
    >>> plot_grouped_bars_on_ax(axes['A'], ...)
    >>> plot_grouped_bars_on_ax(axes['B'], ...)

    >>> # 2×2 grid with spanning panels
    >>> fig, axes = make_panel_grid([['A', 'A'], ['B', 'C']], columns=2)

    Notes
    -----
    - Panel labels ("a", "b", "c"...) are bold, 8pt, lowercase, NO period
    - All text in plots uses final figure scale (6pt = 6pt printed)
    - Titles/subtitles set via set_axis_title() after plotting
    """
    from .style import PLOT_PARAMS, figure_size, _MM
    if params is None:
        params = PLOT_PARAMS

    # Compute figure size
    figsize = figure_size(columns=columns, height_mm=height_mm)

    # Convert mm spacing to inches
    hspace_in = hspace_mm * _MM
    wspace_in = wspace_mm * _MM

    # Create figure with subplot_mosaic
    fig, axes_dict = plt.subplot_mosaic(
        layout,
        figsize=figsize,
        gridspec_kw={
            'width_ratios': width_ratios,
            'height_ratios': height_ratios,
            'hspace': hspace_in / figsize[1],  # Convert to figure fraction
            'wspace': wspace_in / figsize[0],
        },
        constrained_layout=True,
    )

    # Add panel labels if requested
    if add_panel_labels:
        for label, ax in axes_dict.items():
            if isinstance(label, str) and label.strip():
                add_panel_label(ax, label, params=params)

    return fig, axes_dict


# =============================================================================
# Panel Labels
# =============================================================================

def add_panel_label(
    ax: plt.Axes,
    label: str,
    params: dict = None
) -> None:
    """
    Add panel label ("a", "b", etc.) to top-left corner.

    Internal function called by make_panel_grid().

    Parameters
    ----------
    ax : plt.Axes
        Axes to label
    label : str
        Panel letter (e.g., "A", "B", "C")
    params : dict, optional
        PLOT_PARAMS override

    Notes
    -----
    - Format: "a" (lowercase, bold, NO period)
    - Font size: 8pt (only exception to 5-7pt rule)
    - Position: Top-left corner, outside plot area
    - Offset controlled by params['panel_label_offset_mm']
    - Nature style: lowercase letters without periods
    """
    from .style import PLOT_PARAMS, _MM
    if params is None:
        params = PLOT_PARAMS

    # Format label: "a" (lowercase, no period)
    label_text = label.lower()

    # Get offset from params (in mm)
    offset_x_mm, offset_y_mm = params['panel_label_offset_mm']

    # Convert mm to figure fraction
    fig = ax.get_figure()
    figsize = fig.get_size_inches()
    offset_x_frac = (offset_x_mm * _MM) / figsize[0]
    offset_y_frac = (offset_y_mm * _MM) / figsize[1]

    # Add text in axes coordinates
    ax.text(
        offset_x_frac, 1.0 + offset_y_frac,
        label_text,
        transform=ax.transAxes,
        fontsize=params['font_size_panel_label'],
        fontweight='bold',
        ha='left', va='bottom'
    )


# =============================================================================
# Axis Titles
# =============================================================================

def set_axis_title(
    ax: plt.Axes,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    params: dict = None
) -> None:
    """
    Set title and subtitle above plot area.

    Parameters
    ----------
    ax : plt.Axes
        Axes to add title to
    title : str, optional
        Main title (bold, 7pt)
    subtitle : str, optional
        Subtitle below title (normal weight, 6pt)
    params : dict, optional
        PLOT_PARAMS override

    Notes
    -----
    - Title is ALWAYS bold
    - Subtitle is ALWAYS normal weight
    - Position: centered above plot area
    - Spacing controlled by params['title_pad']

    Examples
    --------
    >>> set_axis_title(ax, title="Representational Similarity Analysis",
    ...                subtitle="All ps FDR corrected")
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    title_size = params['font_size_title']
    subtitle_size = params['font_size_label']  # 6pt for subtitles
    pad_pts = params.get('title_pad', 10.0)

    def _escape_mathtext(s: str) -> str:
        return (
            s.replace("\\", r"\\")
             .replace("_", r"\_")
             .replace("%", r"\%")
        )

    if title:
        title = _escape_mathtext(title)
    if subtitle:
        subtitle = _escape_mathtext(subtitle)

    # Simple case: title only
    if title and not subtitle:
        ax.set_title(f"$\\mathbf{{{title}}}$", fontsize=title_size, pad=pad_pts)
        return

    # Subtitle only (rare)
    if subtitle and not title:
        ax.set_title(subtitle, fontsize=subtitle_size, fontweight='normal', pad=pad_pts)
        return

    # Both title and subtitle: manual layout
    ax.set_title("")  # Clear default title

    fig = ax.get_figure()
    bbox_axes_in_fig = ax.get_position()
    fig_h_in = fig.get_figheight()
    axes_h_in = bbox_axes_in_fig.height * fig_h_in
    pts_to_axes = (1.0/72.0) / axes_h_in

    # Title baseline position
    title_y = 1.0 + pad_pts * pts_to_axes

    # Add title (bold)
    ax.text(
        0.5, title_y, f"$\\mathbf{{{title}}}$",
        transform=ax.transAxes,
        fontsize=title_size,
        ha='center', va='bottom'
    )

    # Subtitle below title (normal weight)
    # Simplified spacing: 1.0 × title font size
    subtitle_offset_pts = title_size * 1.0
    subtitle_y = title_y - subtitle_offset_pts * pts_to_axes

    ax.text(
        0.5, subtitle_y, subtitle,
        transform=ax.transAxes,
        fontsize=subtitle_size,
        fontweight='normal',
        ha='center', va='bottom'
    )
