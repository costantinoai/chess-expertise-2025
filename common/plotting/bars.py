#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar plot functions for Nature-compliant figures.

Provides:
- plot_grouped_bars_with_ci(): Standalone bar figure (automatic sizing)
- plot_grouped_bars_on_ax(): Bar plotting on existing axes (for panels)
- Helper functions: _convert_ci_to_yerr, _should_use_hatching, _add_significance_stars
 - plot_counts_on_ax(): Simple count bar plot on existing axes (for panels)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Union

from ..formatters import significance_stars


# =============================================================================
# Helper Functions
# =============================================================================

def _should_use_hatching(group1_color: Union[str, List[str]],
                        group2_color: Union[str, List[str]]) -> bool:
    """
    Determine if hatching should be used to distinguish groups.

    Hatching is needed when groups share the same colors (e.g., ROI-specific colors),
    but NOT when groups have distinct colors (e.g., expert blue vs novice vermillion).

    Parameters
    ----------
    group1_color : str or List[str]
        Color(s) for group 1
    group2_color : str or List[str]
        Color(s) for group 2

    Returns
    -------
    bool
        True if groups share the same colors (hatching needed),
        False if groups have different colors (hatching not needed)

    Examples
    --------
    >>> _should_use_hatching('#0072B2', '#D55E00')  # Different colors
    False
    >>> _should_use_hatching(['#1f77b4', '#ff7f0e'], ['#1f77b4', '#ff7f0e'])  # Same ROI colors
    True
    """
    # Both strings: compare directly
    if isinstance(group1_color, str) and isinstance(group2_color, str):
        return group1_color == group2_color

    # Both lists: check if identical
    if isinstance(group1_color, list) and isinstance(group2_color, list):
        return group1_color == group2_color

    # One string, one list: different color schemes
    return False


def _convert_ci_to_yerr(ci_list: List[Tuple[float, float]],
                        values: List[float]) -> np.ndarray:
    """Convert (lower, upper) CIs to matplotlib error bar format."""
    lower_err = [val - ci[0] for val, ci in zip(values, ci_list)]
    upper_err = [ci[1] - val for val, ci in zip(values, ci_list)]
    return np.array([lower_err, upper_err])


def _calculate_offset_from_range(ax, offset_pct: float = 0.02) -> float:
    """
    Calculate offset as percentage of y-axis range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to get y-limits from
    offset_pct : float, default=0.02
        Percentage of y-range (0.02 = 2%)

    Returns
    -------
    float
        Offset in data units
    """
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    return y_range * offset_pct


def _calculate_auto_ylim(
    values: List[float],
    yerr: np.ndarray,
    pvals: Optional[List[float]] = None,
    group2_values: Optional[List[float]] = None,
    group2_yerr: Optional[np.ndarray] = None,
    group2_pvals: Optional[List[float]] = None,
    comparison_pvals: Optional[List[float]] = None,
    params: dict = None,
    padding_pct: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate automatic y-limits based on data range including error bars and significance stars.

    Parameters
    ----------
    values : List[float]
        Group 1 bar values
    yerr : np.ndarray
        Group 1 error bars (2 x n array: [lower_errors, upper_errors])
    pvals : List[float], optional
        Group 1 p-values for significance stars
    group2_values : List[float], optional
        Group 2 bar values (for grouped plots)
    group2_yerr : np.ndarray, optional
        Group 2 error bars
    group2_pvals : List[float], optional
        Group 2 p-values
    comparison_pvals : List[float], optional
        Comparison p-values between groups
    params : dict, optional
        Plotting parameters
    padding_pct : float, default=0.05
        Padding percentage (5%)

    Returns
    -------
    Tuple[float, float]
        (ymin, ymax) for y-axis limits
    """
    if params is None:
        from .style import PLOT_PARAMS
        params = PLOT_PARAMS

    # Find data range including error bars
    min_vals = []
    max_vals = []

    # Group 1
    for val, err in zip(values, yerr.T):
        min_vals.append(val - err[0])  # val - lower_error
        max_vals.append(val + err[1])  # val + upper_error

    # Group 2 (if present)
    if group2_values is not None and group2_yerr is not None:
        for val, err in zip(group2_values, group2_yerr.T):
            min_vals.append(val - err[0])
            max_vals.append(val + err[1])

    data_min = min(min_vals)
    data_max = max(max_vals)
    data_range = data_max - data_min

    # Add space for significance stars if present
    offset_pct = params.get('significance_offset_pct', 0.02)
    has_stars = (
        (pvals is not None and any(p < 0.05 for p in pvals if p is not None)) or
        (group2_pvals is not None and any(p < 0.05 for p in group2_pvals if p is not None)) or
        (comparison_pvals is not None and any(p < 0.05 for p in comparison_pvals if p is not None))
    )

    if has_stars:
        # Add extra space for stars (offset + line + star height)
        # For comparison mode: offset * 2.0 (line) + offset (star) = 3.0× total
        # For single mode positive: offset (star) = 1.0×
        # For single mode negative: offset * 3.5 (star) = 3.5×
        star_space = data_range * offset_pct * 4.5  # Conservative estimate for all cases
        data_max += star_space
        if data_min < 0:  # Also add space below if we have negative bars
            data_min -= star_space

    # Add padding
    total_range = data_max - data_min
    padding = total_range * padding_pct

    ymin = data_min - padding
    ymax = data_max + padding

    return ymin, ymax


def _add_significance_stars(
    ax, x_positions: np.ndarray,
    values: List[float],
    yerr: np.ndarray,
    pvals: List[float],
    params: dict,
    comparison_mode: bool = False,
    group2_values: Optional[List[float]] = None,
    group2_yerr: Optional[np.ndarray] = None,
    bar_width: Optional[float] = None
):
    """
    Add significance stars above/below bars (unified function).

    Handles both single-bar significance and between-group comparisons.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_positions : np.ndarray
        X positions of bars (or bar groups for comparison mode)
    values : List[float]
        Bar values (group1 in comparison mode)
    yerr : np.ndarray
        Error bar values (2 x n array: [lower_errors, upper_errors])
        Group1 errors in comparison mode
    pvals : List[float]
        P-values for significance testing
    params : dict
        Plotting parameters
    comparison_mode : bool, default=False
        If True, compares two groups with connecting line.
        If False, adds stars above individual bars.
    group2_values : List[float], optional
        Group 2 bar values (required if comparison_mode=True)
    group2_yerr : np.ndarray, optional
        Group 2 error values (required if comparison_mode=True)
    bar_width : float, optional
        Width of bars (required if comparison_mode=True for line positioning)

    Notes
    -----
    Single-bar mode:
    - For positive bars: stars appear above error bar top
    - For negative bars: stars appear below error bar bottom

    Comparison mode:
    - Line positioned at offset above max(group1, group2) error bar tops
    - Star positioned above the line
    - Line spans both bars (uses bar_width)
    """
    # Calculate dynamic offset (from centralized params, default 2% of y-range)
    offset_pct = params.get('significance_offset_pct', 0.02)
    y_offset = _calculate_offset_from_range(ax, offset_pct=offset_pct)

    if comparison_mode:
        # Between-group comparison mode
        if group2_values is None or group2_yerr is None or bar_width is None:
            raise ValueError(
                "comparison_mode=True requires group2_values, group2_yerr, and bar_width"
            )

        for i, pval in enumerate(pvals):
            stars = significance_stars(pval)
            if stars:
                # Find max error bar top between the two groups
                max_top = max(
                    values[i] + yerr[1, i],  # group1
                    group2_values[i] + group2_yerr[1, i]  # group2
                )

                # Line center positioned with larger offset to avoid touching error bars
                line_y = max_top + y_offset * 2.0  # 2× offset for clear separation

                # Draw connecting line first
                ax.plot(
                    [x_positions[i] - bar_width/2, x_positions[i] + bar_width/2],
                    [line_y, line_y],
                    color='black', linewidth=1, zorder=3
                )

                # Star is above the line, centered vertically on proper distance
                star_y = line_y + y_offset

                # Draw star (use center alignment for better visual balance with line)
                ax.text(x_positions[i], star_y, stars,
                       ha='center', va='center',
                       fontsize=params['font_size_annotation'],
                       fontweight='bold')
    else:
        # Single-bar mode (above/below individual bars)
        for x, val, err, pval in zip(x_positions, values, yerr.T, pvals):
            stars = significance_stars(pval)
            if stars:
                # Determine reference point and direction based on bar value
                if val >= 0:
                    # Positive bar: star above with va='bottom'
                    ref_y = val + err[1]  # Top of error bar
                    star_y = ref_y + y_offset
                    va = 'bottom'
                else:
                    # Negative bar: star below with va='top' (text extends downward)
                    ref_y = val - err[0]  # Bottom of error bar
                    star_y = ref_y - y_offset * 3.5  # 3.5× offset for clear separation below
                    va = 'top'

                # Draw star
                ax.text(x, star_y, stars,
                       ha='center', va=va,
                       fontsize=params['font_size_annotation'],
                       fontweight='bold')


# =============================================================================
# Bar Plotting Functions
# =============================================================================

def plot_grouped_bars_on_ax(
    ax, x_positions: np.ndarray,
    group1_values: List[float],
    group1_cis: List[Tuple[float, float]],
    group1_color: Union[str, List[str]],
    group2_values: Optional[List[float]] = None,
    group2_cis: Optional[List[Tuple[float, float]]] = None,
    group2_color: Optional[Union[str, List[str]]] = None,
    group1_label: str = "Group 1",
    group2_label: str = "Group 2",
    group1_pvals: Optional[List[float]] = None,
    group2_pvals: Optional[List[float]] = None,
    comparison_pvals: Optional[List[float]] = None,
    ylim: Optional[Tuple[float, float]] = None,  # NEW: shared y-range
    bar_width_multiplier: float = 1.0,
    params: dict = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Plot bars on an existing axes (for multi-panel figures).

    Handles both single-group and grouped (two-group) bar plots.
    This is the core plotting function containing all bar plotting logic.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_positions : np.ndarray
        X positions for bars
    group1_values : List[float]
        Bar heights for group 1 (or only group if single-group)
    group1_cis : List[Tuple[float, float]]
        Confidence intervals (lower, upper) for group 1
    group1_color : str or List[str]
        Colors for group 1 bars. Can be:
        - str: Single color for all bars
        - List[str]: One color per position (length must match x_positions)
    group2_values, group2_cis, group2_color : optional
        Group 2 data. If None, creates single-group plot.
    group1_label, group2_label : str
        Labels for legend
    group1_pvals, group2_pvals : List[float], optional
        P-values for within-group significance (stars above each group's bars)
    comparison_pvals : List[float], optional
        P-values for between-group comparisons (stars with connecting lines)
        Only used in grouped mode.
    ylim : Tuple[float, float], optional
        Y-axis limits (ymin, ymax). NEW parameter for shared ranges across panels.
    bar_width_multiplier : float, default=1.0
        Multiplier for bar width. Use 2.0 for single-group plots to make bars wider.
    params : dict, optional
        Plotting parameters

    Returns
    -------
    np.ndarray or tuple
        Single-group: returns group1_yerr
        Grouped: returns (group1_yerr, group2_yerr)

    Notes
    -----
    Hatching behavior in grouped mode:
    - Group 1 bars are always solid
    - Group 2 bars are hatched ONLY when both groups share the same colors
    - When groups have different colors, no hatching is applied
    - This ensures visual distinction without redundant hatching

    Examples
    --------
    # Single-group (difference plot)
    >>> plot_grouped_bars_on_ax(
    ...     ax, x_pos, diff_vals, diff_cis, roi_colors,
    ...     group1_pvals=pvals, bar_width_multiplier=2.0
    ... )

    # Two-group comparison
    >>> plot_grouped_bars_on_ax(
    ...     ax, x_pos, exp_vals, exp_cis, roi_colors,
    ...     group2_values=nov_vals, group2_cis=nov_cis, group2_color=roi_colors,
    ...     comparison_pvals=pvals, ylim=(0, 1)
    ... )
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    bw = params['target_bar_width_mm'] * bar_width_multiplier  # Will be converted to data units
    n_items = len(x_positions)
    is_grouped = group2_values is not None

    # Validate color inputs
    if isinstance(group1_color, list) and len(group1_color) != n_items:
        raise ValueError(
            f"group1_color list length ({len(group1_color)}) must match "
            f"number of x_positions ({n_items})"
        )
    if is_grouped and isinstance(group2_color, list) and len(group2_color) != n_items:
        raise ValueError(
            f"group2_color list length ({len(group2_color)}) must match "
            f"number of x_positions ({n_items})"
        )

    # Convert CIs to yerr format
    g1_yerr = _convert_ci_to_yerr(group1_cis, group1_values)

    # Bar width in data coordinates (EXACTLY matching main branch)
    # Main branch: bw = params['bar_width'] = 0.5
    # Offset = ±bw/2, so bars are side-by-side with no gap
    bw = 0.35 * bar_width_multiplier  # Slightly smaller than main (0.5) for Nature style

    if is_grouped:
        # === GROUPED MODE: Two groups side-by-side ===
        g2_yerr = _convert_ci_to_yerr(group2_cis, group2_values)

        # Determine if hatching is needed (only when colors are the same)
        use_hatch = _should_use_hatching(group1_color, group2_color)

        # Group 1 bars (solid) - EXACTLY matching main branch
        ax.bar(x_positions - bw/2, group1_values,
               width=bw, color=group1_color,
               edgecolor=params['bar_edgecolor'],
               linewidth=params['bar_linewidth'],
               alpha=params['bar_alpha'],
               label=group1_label)

        # Group 1 error bars - EXACTLY matching main branch
        ax.errorbar(x_positions - bw/2, group1_values, yerr=g1_yerr,
                    fmt='none', ecolor='black',
                    elinewidth=params['errorbar_linewidth'],
                    capsize=params['errorbar_capsize'], zorder=2)

        # Group 2 bars (hatched only if colors are the same) - EXACTLY matching main branch
        ax.bar(x_positions + bw/2, group2_values,
               width=bw, color=group2_color,
               edgecolor=params['bar_edgecolor'],
               linewidth=params['bar_linewidth'],
               alpha=params['bar_alpha'],
               hatch=params['bar_hatch_novice'] if use_hatch else None,
               label=group2_label)

        # Group 2 error bars - EXACTLY matching main branch
        ax.errorbar(x_positions + bw/2, group2_values, yerr=g2_yerr,
                    fmt='none', ecolor='black',
                    elinewidth=params['errorbar_linewidth'],
                    capsize=params['errorbar_capsize'], zorder=2)

        # Within-group significance stars - EXACTLY matching main branch
        if group1_pvals is not None:
            _add_significance_stars(
                ax, x_positions - bw/2, group1_values, g1_yerr, group1_pvals, params
            )
        if group2_pvals is not None:
            _add_significance_stars(
                ax, x_positions + bw/2, group2_values, g2_yerr, group2_pvals, params
            )

        # Between-group comparison stars - EXACTLY matching main branch
        if comparison_pvals is not None:
            _add_significance_stars(
                ax, x_positions, group1_values, g1_yerr, comparison_pvals, params,
                comparison_mode=True,
                group2_values=group2_values,
                group2_yerr=g2_yerr,
                bar_width=bw
            )

        # Apply ylim (auto-calculate if not provided)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            auto_ymin, auto_ymax = _calculate_auto_ylim(
                group1_values, g1_yerr,
                pvals=group1_pvals,
                group2_values=group2_values,
                group2_yerr=g2_yerr,
                group2_pvals=group2_pvals,
                comparison_pvals=comparison_pvals,
                params=params
            )
            ax.set_ylim(auto_ymin, auto_ymax)

        # Add zero reference line (always, for all bar plots)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.75, alpha=0.6, zorder=1)

        return g1_yerr, g2_yerr

    else:
        # === SINGLE-GROUP MODE: One set of bars ===
        # Bars (centered, wider if multiplier > 1) - EXACTLY matching main branch
        ax.bar(x_positions, group1_values,
               width=bw, color=group1_color,
               edgecolor=params['bar_edgecolor'],
               linewidth=params['bar_linewidth'],
               alpha=params['bar_alpha'],
               label=group1_label)

        # Error bars - EXACTLY matching main branch
        ax.errorbar(x_positions, group1_values, yerr=g1_yerr,
                    fmt='none', ecolor='black',
                    elinewidth=params['errorbar_linewidth'],
                    capsize=params['errorbar_capsize'], zorder=2)

        # Significance stars - EXACTLY matching main branch
        if group1_pvals is not None:
            _add_significance_stars(
                ax, x_positions, group1_values, g1_yerr, group1_pvals, params
            )

        # Apply ylim (auto-calculate if not provided)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            auto_ymin, auto_ymax = _calculate_auto_ylim(
                group1_values, g1_yerr,
                pvals=group1_pvals,
                params=params
            )
            ax.set_ylim(auto_ymin, auto_ymax)

        # Add zero reference line (always, for all bar plots)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.75, alpha=0.6, zorder=1)

        return g1_yerr


def plot_grouped_bars_with_ci(
    group1_values: List[float],
    group1_cis: List[Tuple[float, float]],
    x_labels: List[str],
    group2_values: Optional[List[float]] = None,
    group2_cis: Optional[List[Tuple[float, float]]] = None,
    group1_pvals: Optional[List[float]] = None,
    group2_pvals: Optional[List[float]] = None,
    comparison_pvals: Optional[List[float]] = None,
    group1_label: str = "Group 1",
    group2_label: str = "Group 2",
    group1_color: Optional[Union[str, List[str]]] = None,
    group2_color: Optional[Union[str, List[str]]] = None,
    x_label_colors: Optional[List[str]] = None,
    ylabel: str = "Value",
    title: str = "Comparison",
    subtitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    ylim: Optional[Tuple[float, float]] = None,
    height_mm: Optional[float] = None,
    add_zero_line: bool = False,
    legend_loc: str = "upper left",
    show_legend: bool = True,
    params: dict = None,
    return_ax: bool = False
) -> Union[plt.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Unified bar chart with error bars and significance stars.

    **Automatic sizing**: Figure width computed automatically for fixed 8mm bar widths.

    Handles both single-group and grouped (two-group comparison) bar plots.
    This is a convenience wrapper that creates a complete figure with title, legend, and saves to file.

    For multi-panel layouts, use plot_grouped_bars_on_ax directly on existing axes.

    Parameters
    ----------
    group1_values : List[float]
        Bar heights for group 1 (or only group if single-group plot)
    group1_cis : List[Tuple[float, float]]
        Confidence intervals (lower, upper) for group 1
    x_labels : List[str]
        Labels for x-axis categories
    group2_values, group2_cis : List[float], List[Tuple], optional
        Group 2 data. If None, creates single-group plot.
    group1_pvals, group2_pvals : List[float], optional
        P-values for within-group significance (stars above each group's bars)
    comparison_pvals : List[float], optional
        P-values for between-group comparisons (stars with connecting line above both bars)
        Only used when group2_values is provided.
    group1_label, group2_label : str
        Labels for legend (only used if show_legend=True)
    group1_color, group2_color : str or List[str], optional
        Colors for bars. Can be:
        - None: Uses default color ('#999999' for single-group, COLORS_EXPERT_NOVICE for grouped)
        - str: Single color applied to all bars in that group
        - List[str]: One color per category (length must match x_labels)

        For per-item colors (e.g., ROI colors), pass the same list to both:
        >>> plot_grouped_bars_with_ci(..., group1_color=roi_colors, group2_color=roi_colors)
    x_label_colors : List[str], optional
        Colors for x-axis tick labels (e.g., from format_roi_labels_and_colors).
        If None, labels use default black color.
    ylabel : str
        Y-axis label
    title, subtitle : str
        Plot title and subtitle
    output_path : Path, optional
        If provided, saves figure and closes it
    ylim : tuple, optional
        Y-axis limits (min, max)
    height_mm : float, optional
        Figure height in millimeters (default: 100mm)
    add_zero_line : bool, default=False
        Whether to add horizontal line at y=0
    legend_loc : str, default='upper left'
        Legend location
    show_legend : bool, default=True
        Whether to show legend. For single-group plots, often set to False.
    params : dict, optional
        Plotting parameters
    return_ax : bool, default=False
        If True, returns (fig, ax) instead of just fig

    Returns
    -------
    plt.Figure or (plt.Figure, plt.Axes)
        Figure object (and axes if return_ax=True)

    Notes
    -----
    **NEW: Automatic figure sizing**
    - Figure width computed automatically based on number of categories
    - Bar width is FIXED at 8mm (physical measurement)
    - Single-group: each bar = 8mm
    - Grouped: each bar = 4mm (pair = 8mm)
    - No need to specify figsize - it's inferred from data!

    **Color and hatching**:
    - When using per-item colors, both groups at same x-position use the same color
    - Group 1 bars are always solid
    - Group 2 bars are hatched ONLY when groups share the same colors (e.g., ROI colors)
    - Group 2 bars are NOT hatched when groups have different colors (e.g., expert blue vs novice vermillion)
    - Automatically handles single vs per-item color modes
    - For single-group plots (group2_values=None), only plots group1 bars

    Examples
    --------
    # Single-group plot
    >>> plot_grouped_bars_with_ci(
    ...     group1_values=diffs, group1_cis=cis,
    ...     x_labels=['Term1', 'Term2'], group1_pvals=pvals,
    ...     group1_color='blue', show_legend=False
    ... )
    # → Figure size computed automatically from len(diffs)

    # Two-group comparison
    >>> plot_grouped_bars_with_ci(
    ...     group1_values=expert_vals, group1_cis=expert_cis,
    ...     group2_values=novice_vals, group2_cis=novice_cis,
    ...     x_labels=['Visual', 'Strategy', 'Checkmate'],
    ...     group1_color='#0072B2', group2_color='#D55E00'
    ... )
    # → Figure size computed automatically from len(expert_vals)
    """
    from .style import PLOT_PARAMS, apply_nature_rc, auto_bar_figure_size
    from .colors import COLORS_EXPERT_NOVICE
    from .helpers import set_axis_title
    from .helpers import style_spines
    if params is None:
        params = PLOT_PARAMS

    # Determine if single-group or grouped plot
    is_grouped = group2_values is not None
    n_categories = len(group1_values)

    # AUTOMATIC SIZE INFERENCE
    figsize, bar_width_data = auto_bar_figure_size(
        n_categories=n_categories,
        is_grouped=is_grouped,
        height_mm=height_mm
    )

    # Set default colors if not provided
    if group1_color is None:
        group1_color = COLORS_EXPERT_NOVICE['expert'] if is_grouped else '#999999'
    if group2_color is None and is_grouped:
        group2_color = COLORS_EXPERT_NOVICE['novice']

    # Create figure and axis
    apply_nature_rc(params)
    fig, ax = plt.subplots(figsize=figsize, facecolor=params['facecolor'])

    x_positions = np.arange(n_categories)

    # Plot using core function
    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x_positions,
        group1_values=group1_values,
        group1_cis=group1_cis,
        group1_color=group1_color,
        group2_values=group2_values,
        group2_cis=group2_cis,
        group2_color=group2_color,
        group1_label=group1_label,
        group2_label=group2_label,
        group1_pvals=group1_pvals,
        group2_pvals=group2_pvals,
        comparison_pvals=comparison_pvals,
        ylim=ylim,
        params=params
    )

    # Add labels and styling (figure-level)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)  # Ensure all bars are visible
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')

    # Color x-tick labels if colors provided (e.g., from format_roi_labels_and_colors)
    if x_label_colors is not None:
        for ticklabel, color in zip(ax.get_xticklabels(), x_label_colors):
            ticklabel.set_color(color)

    # Use centralized label formatter (sanitizes mathtext and enforces casing)
    from .helpers import label_axes
    label_axes(ax, ylabel=ylabel, params=params)

    if add_zero_line:
        ax.axhline(0, color='black', linestyle='--',
                  linewidth=params['plot_linewidth'], alpha=0.3, zorder=1)

    set_axis_title(ax, title, subtitle=subtitle, params=params)

    if show_legend:
        ax.legend(loc=legend_loc, ncol=2, frameon=False,
                  fontsize=params['font_size_legend'])

    sns.despine(trim=False)
    style_spines(ax, visible_spines=['left', 'bottom'], params=params)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                   facecolor=params['facecolor'], dpi=params['dpi'])
        plt.close(fig)

    if return_ax:
        return fig, ax
    return fig


def plot_counts_on_ax(
    ax,
    x_values,
    counts,
    colors=None,
    alphas=None,
    xlabel: str = 'Stimulus ID',
    ylabel: str = 'Selection count',
    title: str = None,
    subtitle: str = None,
    legend: Optional[List[Tuple[str, str, float]]] = None,
    params: dict = None,
):
    """
    Plot a simple count bar chart on an existing axes with optional per-bar colors/alphas.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_values : array-like
        Category identifiers for x-axis
    counts : array-like
        Counts per category
    colors : list of str, optional
        Bar facecolors. If None, uses a neutral color
    alphas : list of float, optional
        Bar alpha values per bar. If None, uses 1.0
    xlabel, ylabel : str
        Axis labels (passed through label formatter)
    title, subtitle : str, optional
        Title and subtitle (subtitle normal weight)
    legend : list of tuples (label, color, alpha), optional
        Legend items to display (e.g., [('Checkmate', '#CCBB44', .7), ...])
    params : dict, optional
        PLOT_PARAMS override
    """
    from .style import PLOT_PARAMS
    from .helpers import label_axes, style_spines
    from .helpers import set_axis_title
    if params is None:
        params = PLOT_PARAMS

    x_idx = np.arange(len(x_values))
    if colors is None:
        colors = ['#999999'] * len(x_values)
    if alphas is None:
        alphas = [1.0] * len(x_values)

    # Draw bars
    for i, (x, h, c, a) in enumerate(zip(x_idx, counts, colors, alphas)):
        ax.bar(x, h, color=c, alpha=a, edgecolor='black', linewidth=params.get('bar_linewidth', 0.5))

    # Labels and styling
    label_axes(ax, xlabel=xlabel, ylabel=ylabel, params=params)
    if title or subtitle:
        set_axis_title(ax, title=title, subtitle=subtitle, params=params)

    # X tick handling: hide tick labels by default (dense), caller can override
    ax.set_xticks(x_idx)
    ax.set_xticklabels([])

    style_spines(ax, visible_spines=['left', 'bottom'], params=params)

    # Optional legend
    if legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=col, edgecolor='black', label=lbl, alpha=alpha) for (lbl, col, alpha) in legend]
        ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=params['font_size_legend'])

    # Zero baseline gridline not required; counts are positive
