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

    import re

    def _sanitize(text: str) -> str:
        # Remove math delimiters and simple LaTeX styling commands
        t = text.replace('$', '')
        t = re.sub(r"\\it\{([^}]*)\}", r"\1", t)
        t = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
        t = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", t)
        return t

    def _format_label(text: str) -> str:
        if not text:
            return text
        text = _sanitize(text)
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

def format_roi_labels_and_colors(
    welch_df,
    roi_info,
    alpha: float = 0.05
) -> Tuple[List[str], List[str], List[str]]:
    """
    Format ROI labels and get matching colors (EXACTLY matching PR ttest plot).

    Merges welch dataframe with roi_info to get pretty names and colors.
    This is the standard helper for all bar plots, ensuring consistency.

    Parameters
    ----------
    welch_df : pd.DataFrame
        Statistics dataframe with columns: ROI_Label, ROI_Name, p_val_fdr, etc.
    roi_info : pd.DataFrame
        ROI metadata with columns: roi_id, roi_name, pretty_name, color
        (from load_roi_metadata)
    alpha : float, default=0.05
        Significance threshold

    Returns
    -------
    formatted_names : list of str
        Pretty ROI names with newlines removed (for display)
    roi_colors : list of str
        Hex colors matching ROI order
    label_colors : list of str
        Colors for x-tick labels (ROI color if sig, grey if not sig)

    Examples
    --------
    >>> # Standard usage in bar plot panels (EXACTLY like PR plot)
    >>> welch = group_stats['rsa_corr']['checkmate']['welch_expert_vs_novice']
    >>> roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
    >>> names, colors, label_colors = format_roi_labels_and_colors(welch, roi_info)
    >>> plot_grouped_bars_on_ax(ax, x, vals, cis, group1_color=colors)
    >>> ax.set_xticklabels(names, rotation=30, ha='right')
    >>> for ticklabel, color in zip(ax.get_xticklabels(), label_colors):
    ...     ticklabel.set_color(color)
    """
    import pandas as pd

    # Merge with roi_info to get pretty names and colors (EXACTLY like PR plot)
    # welch has ROI_Label (numeric), roi_info has roi_id (numeric)
    merged = welch_df.merge(
        roi_info[['roi_id', 'pretty_name', 'color']],
        left_on='ROI_Label', right_on='roi_id', how='left'
    )

    # Extract pretty names and format (remove newlines)
    formatted_names = merged['pretty_name'].tolist()
    formatted_names = [name.replace("\\n", " ") for name in formatted_names]

    # Extract colors
    roi_colors = merged['color'].tolist()

    # Determine label colors based on significance (EXACTLY like PR plot bottom panel)
    pvals = merged['p_val_fdr'].values
    label_colors = [
        color if pval < alpha else '#999999'
        for pval, color in zip(pvals, roi_colors)
    ]

    return formatted_names, roi_colors, label_colors


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


# =============================================================================
# Export utilities (DRY): save arranged axes and panel SVGs
# =============================================================================

def sanitize_label_to_filename(label: str) -> str:
    """Sanitize an axes label to a safe filename token."""
    import re
    if not label:
        return 'axis'
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(label))


def save_axes_svgs(fig, out_dir: Path | str, prefix: str,
                   expand_xy: tuple = (1.02, 1.08),
                   dpi: int = 450) -> List[Path]:
    """
    Save each axes in a Matplotlib figure as a separate SVG using tight bboxes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure whose axes will be exported
    out_dir : Path | str
        Output directory to place individual axis SVGs
    prefix : str
        Filename prefix for exported SVGs
    expand_xy : (float, float)
        Expansion factors for bbox width and height to avoid clipping titles

    Returns
    -------
    list[Path]
        Paths of saved SVG files
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    saved: List[Path] = []
    for idx, ax in enumerate(fig.axes, start=1):
        label = getattr(ax, 'get_label', lambda: f'ax{idx}')()
        safe = sanitize_label_to_filename(label or f'ax{idx}')
        out_path = out_dir / f"{prefix}__{safe}.svg"
        bbox = ax.get_tightbbox(renderer).expanded(expand_xy[0], expand_xy[1])
        bbox_in = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out_path, format='svg', bbox_inches=bbox_in, dpi=dpi)
        saved.append(out_path)
    return saved


def save_panel_svg(fig, output_file: Path | str, dpi: int = 450) -> Path:
    """Save full arranged panel as SVG with tight bbox and return path."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format='svg', bbox_inches='tight', dpi=dpi)
    return output_file


# =============================================================================
# Titles
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
        Main title (bold)
    subtitle : str, optional
        Subtitle below title (normal weight)
    params : dict, optional
        PLOT_PARAMS override
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    import re

    def _sanitize_text(s: Optional[str]) -> Optional[str]:
        if not s:
            return s
        s = s.replace('$', '')
        s = re.sub(r"\\it\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
        s = s.replace('  ', ' ')
        return s

    title = _sanitize_text(title)
    subtitle = _sanitize_text(subtitle)

    title_size = params['font_size_title']
    subtitle_size = params['font_size_label']
    pad_pts = params.get('title_pad', 10.0)

    if title and not subtitle:
        ax.set_title(title, fontsize=title_size, fontweight='bold', pad=pad_pts)
        return

    if subtitle and not title:
        ax.set_title(subtitle, fontsize=subtitle_size, fontweight='normal', pad=pad_pts)
        return

    ax.set_title("")

    fig = ax.get_figure()
    bbox_axes_in_fig = ax.get_position()
    fig_h_in = fig.get_figheight()
    axes_h_in = bbox_axes_in_fig.height * fig_h_in
    pts_to_axes = (1.0/72.0) / axes_h_in

    title_y = 1.0 + pad_pts * pts_to_axes
    ax.text(
        0.5, title_y, title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight='bold',
        ha='center', va='bottom'
    )

    subtitle_offset_pts = title_size * 0.9
    subtitle_y = title_y - subtitle_offset_pts * pts_to_axes
    ax.text(
        0.5, subtitle_y, subtitle,
        transform=ax.transAxes,
        fontsize=subtitle_size,
        fontweight='normal',
        ha='center', va='bottom'
    )
