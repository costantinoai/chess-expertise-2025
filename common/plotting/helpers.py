#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper utilities for Nature-compliant plotting.

Provides:
- compute_ylim_range(): Universal range computation for bar plots, RDMs, brain maps
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

def compute_ylim_range(
    *value_lists,
    symmetric: bool = False,
    zero_anchor: bool = False,
    padding_pct: float = 0.1,
    round_decimals: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute shared range (ylim/vmin/vmax) for visualizations.

    Universal range computation for bar plots, RDMs, brain maps, and surface plots.
    Handles lists and numpy arrays, ignoring non-finite values.

    Use Cases
    ---------
    **Bar plots** (default behavior):
        Uses simple (min, max) with 10% padding for natural spacing.

        >>> # Shared y-axis across expert/novice/difference panels
        >>> ylim = compute_ylim_range(expert_vals, novice_vals, diff_vals)
        >>> plot_grouped_bars_on_ax(ax1, ..., ylim=ylim)
        >>> plot_grouped_bars_on_ax(ax2, ..., ylim=ylim)

    **RDMs and brain maps** (symmetric around 0):
        Centers color scale at 0 using maximum absolute value.

        >>> # Symmetric range for RDM triplet (expert/novice/difference)
        >>> vmin, vmax = compute_ylim_range(
        ...     expert_rdm, novice_rdm, diff_rdm,
        ...     symmetric=True,
        ...     padding_pct=0.0
        ... )
        >>> plot_rdm_on_ax(ax1, expert_rdm, vmin=vmin, vmax=vmax)

        >>> # Symmetric range for NIfTI brain volumes
        >>> vmin, vmax = compute_ylim_range(
        ...     *[img.get_fdata() for img in nifti_volumes],
        ...     symmetric=True,
        ...     padding_pct=0.0
        ... )
        >>> plot_flat_pair(data=volume, vmin=vmin, vmax=vmax)

    **Zero-anchored surface maps** (positive or negative only):
        Anchors one end at 0, extends to extrema on the other side.

        >>> # Pial surface with only positive Δr values
        >>> vmin, vmax = compute_ylim_range(
        ...     delta_r_sig_values,
        ...     symmetric=False,
        ...     zero_anchor=True,
        ...     padding_pct=0.0,
        ...     round_decimals=1
        ... )
        >>> # Returns (0.0, max_positive) rounded to 1 decimal

    Parameters
    ----------
    *value_lists : list[float] or np.ndarray
        One or more value collections (lists, arrays, or flattened volumes).
        Non-finite values (NaN, Inf) are automatically ignored.
    symmetric : bool, default=False
        If True, center range at 0 using max absolute value: (-max_abs, max_abs).
        Use for: RDMs, correlation differences, brain activation maps.
    zero_anchor : bool, default=False
        If True (requires symmetric=False), anchor one end at 0.
        - Only positive values → (0.0, max)
        - Only negative values → (min, 0.0)
        - Both signs → (min, max)  [includes 0 within range]
        Use for: pial surface maps with only positive or only negative deltas.
    padding_pct : float, default=0.1
        Fractional padding added to range (0.1 = 10%).
        - Use 0.1 for bar plots (natural spacing)
        - Use 0.0 for brain/RDM visualizations (exact data range)
    round_decimals : int, optional
        If provided, round vmin and vmax to this many decimal places.
        Useful for clean colorbar labels (e.g., round_decimals=1 → 0.1, 0.2, ...).

    Returns
    -------
    (vmin, vmax) : tuple[float, float]
        Display range for ylim, vmin/vmax, or other range parameters.

    Notes
    -----
    - Backwards compatible: default parameters match original bar plot behavior
    - Handles mixed input types: lists, numpy arrays, flattened volumes
    - Robust to NaN/Inf: non-finite values are filtered before computation
    - Zero-division safe: identical values get small default padding

    See Also
    --------
    plot_grouped_bars_on_ax : Bar plotting with ylim parameter
    plot_rdm_on_ax : RDM plotting with vmin/vmax parameters
    plot_flat_pair : Surface plotting with vmin/vmax parameters
    """
    # Collect all finite values from inputs (handles lists and arrays)
    vals: list[float] = []
    for item in value_lists:
        if item is None:
            continue
        # Handle both lists and numpy arrays
        arr = np.asarray(item)
        if arr.ndim == 0:  # Scalar
            if np.isfinite(arr):
                vals.append(float(arr))
        else:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                vals.extend(finite.tolist())

    # Handle empty input
    if not vals:
        return (0.0, 1.0) if zero_anchor else (-1.0, 1.0)

    vmin_data = float(np.min(vals))
    vmax_data = float(np.max(vals))

    # Compute range based on mode
    if symmetric:
        # Symmetric around 0: (-max_abs, max_abs)
        m = max(abs(vmin_data), abs(vmax_data))
        vmin, vmax = -m, m
    elif zero_anchor:
        # Anchor one end at 0
        has_pos = vmax_data > 0
        has_neg = vmin_data < 0
        if has_pos and not has_neg:
            # Only positive → (0, max)
            vmin, vmax = 0.0, vmax_data
        elif has_neg and not has_pos:
            # Only negative → (min, 0)
            vmin, vmax = vmin_data, 0.0
        else:
            # Both signs → (min, max) which includes 0
            vmin, vmax = vmin_data, vmax_data
    else:
        # Simple min/max
        vmin, vmax = vmin_data, vmax_data

    # Apply padding (only if range is non-zero)
    if padding_pct and vmax > vmin:
        span = vmax - vmin
        if span == 0:
            # All values identical: add small default padding
            vmin -= 0.1
            vmax += 0.1
        else:
            pad = span * float(padding_pct)
            vmin -= pad
            vmax += pad

    # Optional rounding (for clean colorbar labels)
    if round_decimals is not None:
        vmin = round(vmin, int(round_decimals))
        vmax = round(vmax, int(round_decimals))

    return float(vmin), float(vmax)


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
    - Also applies MaxNLocator to limit Y-axis tick counts (max 6 intervals)
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

    # Apply MaxNLocator for legible tick counts (Y-axis only)
    # DRY policy: limit applies centrally to Y labels; X labels should show all
    # categories when explicitly provided by callers (e.g., bar plots).
    tick_max_nbins = params.get('tick_max_nbins', 6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_max_nbins))


def hide_ticks(ax, hide_x: bool = True, hide_y: bool = True):
    """
    Fully hide tick marks and labels on the given axes.
    """
    if hide_x:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if hide_y:
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)



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


# =============================================================================
# Regression overlay helper (DRY)
# =============================================================================

def draw_regression_line(
    ax,
    xvals,
    slope: float,
    intercept: float,
    r: float | None = None,
    p: float | None = None,
    *,
    params: dict | None = None,
    color: str = 'black',
    annotate: bool = True,
    text_xy: tuple = (0.98, 0.02),
):
    """
    Draw a simple regression line and optional r/p annotation on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    xvals : array-like
        X values used to bound the line segment (min..max).
    slope, intercept : float
        Regression line parameters (y = intercept + slope * x).
    r, p : float, optional
        Correlation and p-value to annotate in the lower-right corner.
    params : dict, optional
        PLOT_PARAMS override.
    color : str, default 'black'
        Line color.
    annotate : bool, default True
        Whether to add the text annotation when r and p are provided.
    text_xy : (float, float), default (0.98, 0.02)
        Axes-relative coordinates for the annotation text.
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    import numpy as _np
    xvals = _np.asarray(xvals)
    if xvals.size == 0:
        return

    x0, x1 = float(_np.min(xvals)), float(_np.max(xvals))
    y0 = intercept + slope * x0
    y1 = intercept + slope * x1

    ax.plot(
        [x0, x1], [y0, y1],
        color=color,
        linewidth=params['plot_linewidth'],
        alpha=params.get('line_alpha', 0.5),
        zorder=1,
    )

    if annotate and (r is not None) and (p is not None):
        ax.text(
            text_xy[0], text_xy[1], f"r={r:.2f}, p={p:.3f}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=params['font_size_tick'], color='#666666'
        )


def save_panel_pdf(fig, output_file: Path | str, dpi: int = 450) -> Path:
    """Save full arranged panel as PDF with tight bbox and return path."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format='pdf', bbox_inches='tight', dpi=dpi)
    return output_file

def save_axes_pngs(fig, out_dir: Path | str, prefix: str,
                   expand_xy: tuple = (1.02, 1.08),
                   dpi: int = 450) -> List[Path]:
    """
    Save each axes in a Matplotlib figure as a separate PNG using tight bboxes.

    Mirrors save_axes_svgs but writes PNG files for quick previews or raster submissions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    saved: List[Path] = []
    for idx, ax in enumerate(fig.axes, start=1):
        label = getattr(ax, 'get_label', lambda: f'ax{idx}')()
        safe = sanitize_label_to_filename(label or f'ax{idx}')
        out_path = out_dir / f"{prefix}__{safe}.png"
        bbox = ax.get_tightbbox(renderer).expanded(expand_xy[0], expand_xy[1])
        bbox_in = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out_path, format='png', bbox_inches=bbox_in, dpi=dpi)
        saved.append(out_path)
    return saved


def save_panel_png(fig, output_file: Path | str, dpi: int = 450) -> Path:
    """Save full arranged panel as PNG with tight bbox and return path."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format='png', bbox_inches='tight', dpi=dpi)
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

    # Increase spacing between title and subtitle (configurable via params)
    gap_factor = params.get('title_subtitle_gap_factor', 1.2)
    subtitle_offset_pts = title_size * gap_factor
    subtitle_y = title_y - subtitle_offset_pts * pts_to_axes
    ax.text(
        0.5, subtitle_y, subtitle,
        transform=ax.transAxes,
        fontsize=subtitle_size,
        fontweight='normal',
        ha='center', va='bottom'
    )


def create_standalone_colorbar(
    cmap,
    vmin: float = 0.0,
    vmax: float = 1.0,
    orientation: str = 'horizontal',
    label: Optional[str] = None,
    output_path: Optional[Path] = None,
    params: dict | None = None,
    tick_position: Optional[str] = None,
) -> plt.Figure:
    """
    Create a standalone colorbar figure with 3 ticks (vmin, center, vmax).

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to use.
    vmin : float, default=0.0
        Minimum value for colorbar.
    vmax : float, default=1.0
        Maximum value for colorbar.
    orientation : str, default='horizontal'
        Orientation: 'horizontal' or 'vertical'.
    label : str, optional
        Label for the colorbar.
    output_path : Path, optional
        If provided, save figure to this path.
    params : dict, optional
        Plotting parameters.
    tick_position : str, optional
        Position of ticks and labels. For vertical: 'left' or 'right' (default: 'right').
        For horizontal: 'top' or 'bottom' (default: 'bottom').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure containing only the colorbar.

    Notes
    -----
    - Colorbars have consistent dimensions (3.0 × 0.5 inches) regardless of orientation
    - 3 ticks are always shown: vmin, center (midpoint), and vmax
    - Tick labels are automatically formatted based on value magnitude
    """
    from .style import PLOT_PARAMS, apply_nature_rc

    if params is None:
        params = PLOT_PARAMS
    apply_nature_rc(params)

    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    # Create figure with exactly matching dimensions (just rotated)
    if orientation == 'horizontal':
        fig_w = 3.0  # inches
        fig_h = 0.5  # inches
    else:  # vertical
        fig_w = 0.5  # inches (must match horizontal height exactly)
        fig_h = 3.0  # inches

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Centered axis with identical physical thickness for both orientations
    bar_fraction = 0.18  # fraction of the short dimension occupied by the colorbar (thinner)
    margin_along_axis = 0.06  # padding along the long axis for labels/ticks (reduced)
    margin_per_side = (1.0 - bar_fraction) / 2.0

    if orientation == 'horizontal':
        ax = fig.add_axes([
            margin_along_axis,
            margin_per_side,
            1.0 - 2.0 * margin_along_axis,
            bar_fraction,
        ])
    else:
        ax = fig.add_axes([
            margin_per_side,
            margin_along_axis,
            bar_fraction,
            1.0 - 2.0 * margin_along_axis,
        ])
    ax.set_facecolor('none')

    # Create colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        cax=ax,
        orientation=orientation,
    )

    # Set 3 ticks: vmin, center, vmax
    center = (vmin + vmax) / 2.0
    ticks = [vmin, center, vmax]
    cbar.set_ticks(ticks)

    # Determine if we should use integer formatting (if vmin and vmax are whole numbers)
    def is_whole_number(val):
        return abs(val - round(val)) < 1e-10

    use_int_format = is_whole_number(vmin) and is_whole_number(vmax)

    # Format tick labels with appropriate precision
    def format_tick(val):
        if use_int_format and is_whole_number(val):
            return f"{int(round(val))}"
        elif abs(val) < 0.01 and val != 0:
            return f"{val:.2e}"
        elif abs(val) >= 1000:
            return f"{val:.0f}"
        elif abs(val) >= 10:
            return f"{val:.1f}"
        else:
            return f"{val:.2f}"

    cbar.set_ticklabels([format_tick(t) for t in ticks])

    # Position ticks and labels
    if tick_position:
        if orientation == 'vertical':
            if tick_position == 'left':
                cbar.ax.yaxis.set_ticks_position('left')
                cbar.ax.yaxis.set_label_position('left')
            elif tick_position == 'right':
                cbar.ax.yaxis.set_ticks_position('right')
                cbar.ax.yaxis.set_label_position('right')
        elif orientation == 'horizontal':
            if tick_position == 'top':
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
            elif tick_position == 'bottom':
                cbar.ax.xaxis.set_ticks_position('bottom')
                cbar.ax.xaxis.set_label_position('bottom')

    # Style - use larger fonts for better readability (2x standard size)
    cbar.ax.tick_params(labelsize=params['font_size_label'] * 2)  # Use 2x label size for ticks

    if label:
        if orientation == 'horizontal':
            cbar.set_label(label, fontsize=params['font_size_title'] * 2)  # Use 2x title size for label
        else:
            if tick_position == 'left':
                cbar.set_label(label, fontsize=params['font_size_title'] * 2, rotation=90, labelpad=5)
            else:
                cbar.set_label(label, fontsize=params['font_size_title'] * 2, rotation=270, labelpad=12)

    # Set colorbar outline width
    cbar.outline.set_linewidth(params['plot_linewidth'])

    # Note: tight_layout() causes issues with colorbar-only figures, so we skip it

    if output_path is not None:
        save_figure(fig, Path(output_path))
    return fig
