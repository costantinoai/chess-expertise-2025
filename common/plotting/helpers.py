#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper utilities for Nature-compliant plotting.

Provides:
- compute_symmetric_range(): Symmetric vmin/vmax for RDMs
- compute_ylim_range(): Shared y-axis limits for bar plots
- format_axis_commas(): Thousands separators (1,000)
- label_axes(): Format axis labels (capitalize, no periods)
- style_spines(): Apply spine styling with MaxNLocator
- hide_ticks(): Hide tick labels
- save_figure(): Centralized figure saving with validation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal
from matplotlib.ticker import FuncFormatter, MaxNLocator


# =============================================================================
# Range Computation
# =============================================================================

def compute_symmetric_range(*arrays: np.ndarray) -> Tuple[float, float]:
    """
    Compute symmetric vmin/vmax for multiple arrays (for RDMs/heatmaps).

    Ensures color scale is centered at 0 with symmetric limits.

    Parameters
    ----------
    *arrays : np.ndarray
        One or more arrays to compute range from

    Returns
    -------
    vmin, vmax : float, float
        Symmetric limits (-max_abs, max_abs)

    Examples
    --------
    >>> # Shared color range for multiple RDMs
    >>> vmin, vmax = compute_symmetric_range(expert_rdm, novice_rdm, diff_rdm)
    >>> plot_rdm_on_ax(ax1, expert_rdm, vmin=vmin, vmax=vmax)
    >>> plot_rdm_on_ax(ax2, novice_rdm, vmin=vmin, vmax=vmax)
    """
    max_abs = max(np.abs(arr).max() for arr in arrays if arr is not None)
    return (-max_abs, max_abs)


def compute_ylim_range(
    *value_lists: List[float],
    padding_pct: float = 0.1
) -> Tuple[float, float]:
    """
    Compute shared ylim for multiple bar plots.

    Parameters
    ----------
    *value_lists : List[float]
        One or more lists of values to compute range from
    padding_pct : float, default=0.1
        Padding as fraction of range (0.1 = 10% padding)

    Returns
    -------
    ylim : (ymin, ymax)

    Examples
    --------
    >>> # Shared y-axis for expert/novice/difference panels
    >>> ylim = compute_ylim_range(expert_vals, novice_vals, diff_vals)
    >>> plot_grouped_bars_on_ax(ax1, ..., ylim=ylim)
    >>> plot_grouped_bars_on_ax(ax2, ..., ylim=ylim)
    """
    all_vals = [v for lst in value_lists for v in lst if v is not None]
    if not all_vals:
        return (0, 1)

    ymin = min(all_vals)
    ymax = max(all_vals)
    y_range = ymax - ymin

    if y_range == 0:
        # All values identical
        return (ymin - 0.1, ymax + 0.1)

    padding = y_range * padding_pct
    return (ymin - padding, ymax + padding)


# =============================================================================
# Axis Formatting
# =============================================================================

def format_axis_commas(ax, axis: Literal['x', 'y'] = 'y', decimals: int = 0):
    """
    Apply thousands separators to axis tick labels (1,000 not 1000).

    Parameters
    ----------
    ax : plt.Axes
        Axes to format
    axis : 'x' or 'y', default='y'
        Which axis to format
    decimals : int, default=0
        Number of decimal places

    Examples
    --------
    >>> format_axis_commas(ax, axis='y', decimals=2)
    """
    formatter = FuncFormatter(lambda x, p: f'{x:,.{decimals}f}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def label_axes(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    params: dict = None
):
    """
    Set axis labels with Nature formatting.

    - Capitalizes first letter
    - Removes trailing periods
    - Uses font size from params

    Parameters
    ----------
    ax : plt.Axes
        Axes to label
    xlabel, ylabel : str, optional
        Axis labels
    params : dict, optional
        PLOT_PARAMS override

    Examples
    --------
    >>> label_axes(ax, xlabel="correlation (r)", ylabel="count")
    # Result: "Correlation (r)" and "Count"
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    def _format_label(text: str) -> str:
        if not text:
            return text
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        return text.rstrip('.')

    if xlabel:
        ax.set_xlabel(_format_label(xlabel), fontsize=params['font_size_label'])
    if ylabel:
        ax.set_ylabel(_format_label(ylabel), fontsize=params['font_size_label'])


# =============================================================================
# Spine and Tick Styling
# =============================================================================

def style_spines(
    ax,
    visible_spines: List[str] = ['left', 'bottom'],
    params: dict = None
):
    """
    Apply spine styling and tick locator.

    Parameters
    ----------
    ax : plt.Axes
        Axes to style
    visible_spines : list of str, default=['left', 'bottom']
        Which spines to show
    params : dict, optional
        PLOT_PARAMS override

    Notes
    -----
    - Also applies MaxNLocator to limit tick counts (max 6 intervals)
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    linewidth = params['spine_linewidth']

    for spine_loc in ['left', 'right', 'top', 'bottom']:
        if spine_loc in visible_spines:
            ax.spines[spine_loc].set_visible(True)
            ax.spines[spine_loc].set_linewidth(linewidth)
            ax.spines[spine_loc].set_edgecolor('black')
        else:
            ax.spines[spine_loc].set_visible(False)

    # Apply MaxNLocator for legible tick counts at small fonts
    tick_max_nbins = params.get('tick_max_nbins', 6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_max_nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_max_nbins))


def hide_ticks(ax, hide_x: bool = True, hide_y: bool = True):
    """
    Hide tick labels.

    Parameters
    ----------
    ax : plt.Axes
        Axes to modify
    hide_x, hide_y : bool, default=True
        Whether to hide x/y tick labels
    """
    if hide_x:
        ax.set_xticks([])
    if hide_y:
        ax.set_yticks([])


# =============================================================================
# Figure Saving
# =============================================================================

def save_figure(
    fig: plt.Figure,
    path_stem: Union[str, Path],
    width: Literal['single', 'double'] = 'double',
    height_mm: Optional[float] = None,
    dpi: int = 450,
    formats: Tuple[str, ...] = ('pdf', 'svg'),
    include_png: bool = False,
    params: dict = None
) -> List[Path]:
    """
    Save figure with Nature-compliant settings.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    path_stem : str or Path
        Output path without extension (e.g., 'output_dir/figure1')
    width : 'single' or 'double', default='double'
        Figure width (89mm or 183mm)
    height_mm : float, optional
        Expected height (for validation)
    dpi : int, default=450
        DPI for raster elements (Nature: ≥300)
    formats : tuple, default=('pdf', 'svg')
        Output formats (vector preferred)
    include_png : bool, default=False
        Also save PNG for drafts/previews
    params : dict, optional
        PLOT_PARAMS override

    Returns
    -------
    list of Path
        Saved file paths

    Notes
    -----
    - Validates dimensions (max 170mm height)
    - Ensures white facecolor, no transparency
    - Saves PDF first (Nature preferred), then SVG
    - Optional PNG for quick preview

    Examples
    --------
    >>> save_figure(fig, output_dir / 'analysis', width='double')
    [PosixPath('output_dir/analysis.pdf'), PosixPath('output_dir/analysis.svg')]
    """
    from .style import PLOT_PARAMS, _MM, _NATURE_MAX_HEIGHT_MM
    if params is None:
        params = PLOT_PARAMS

    path_stem = Path(path_stem)

    # Validate dimensions
    figsize = fig.get_size_inches()
    if height_mm and (figsize[1] / _MM) > _NATURE_MAX_HEIGHT_MM:
        import warnings
        warnings.warn(
            f"Figure height {figsize[1]/_MM:.1f}mm exceeds Nature max (170mm)"
        )

    # Save with consistent settings
    save_kwargs = {
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': params['facecolor'],
        'dpi': dpi,
    }

    saved_files = []

    # PDF (Nature preferred)
    if 'pdf' in formats:
        pdf_path = path_stem.with_suffix('.pdf')
        fig.savefig(pdf_path, format='pdf', **save_kwargs)
        saved_files.append(pdf_path)

    # SVG (editable vector)
    if 'svg' in formats:
        svg_path = path_stem.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', **save_kwargs)
        saved_files.append(svg_path)

    # PNG (optional draft/preview)
    if include_png or 'png' in formats:
        png_path = path_stem.with_suffix('.png')
        fig.savefig(png_path, format='png', **save_kwargs)
        saved_files.append(png_path)

    return saved_files
