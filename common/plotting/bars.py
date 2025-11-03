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
                    color='black', linewidth=params['comparison_linewidth'], zorder=3
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
    group1_cis: Optional[List[Tuple[float, float]]] = None,
    group1_color: Union[str, List[str]] = '#999999',
    group2_values: Optional[List[float]] = None,
    group2_cis: Optional[List[Tuple[float, float]]] = None,
    group2_color: Optional[Union[str, List[str]]] = None,
    group1_label: Union[str, List[str]] = "Group 1",
    group2_label: str = "Group 2",
    group1_pvals: Optional[List[float]] = None,
    group2_pvals: Optional[List[float]] = None,
    comparison_pvals: Optional[List[float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    bar_width_multiplier: float = 1.0,
    show_errorbars: bool = True,
    add_value_labels: bool = False,
    value_label_format: str = '.2f',
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xtick_labels: Optional[List[str]] = None,
    x_label_colors: Optional[List[str]] = None,
    x_tick_rotation: int = 30,
    x_tick_align: str = 'right',
    hide_xticklabels: bool = False,
    show_legend: Optional[bool] = None,
    legend_loc: str = 'upper right',
    visible_spines: Optional[List[str]] = None,
    params: dict = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Plot grouped or single bar charts on an existing Axes.

    Supports:
    - Standard grouped bar plots (two groups side-by-side)
    - Single-group plots with per-bar colors and auto-generated
      color legends using `_apply_color_legend_from_bars`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw the bars on.
    x_positions : np.ndarray
        X positions for bars.
    group1_values : list of float
        Heights for the first group (or the only group if single).
    group1_color : str or list of str
        Bar colors for group 1. Can be one color or a list per bar.
    group1_label : str or list of str
        Legend label(s). If list, must correspond to unique colors.
    show_legend : bool, optional
        Whether to show a legend (auto-suppressed if one was drawn manually).
    params : dict
        Style parameters including bar width, colors, fonts, etc.

    Returns
    -------
    np.ndarray or tuple
        Error bars (yerr arrays) for group 1 or (group1, group2) if grouped.

    Notes
    -----
    - If `group1_color` and `group1_label` are lists, the function will
      call `_apply_color_legend_from_bars()` to generate a legend with
      exact styling taken from the actual bars.
    - If `show_legend=True` but the internal color legend is created,
      `_apply_dry_formatting()` is instructed *not* to rebuild it.
    """
    from .style import PLOT_PARAMS
    if params is None:
        params = PLOT_PARAMS

    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    n_items = len(x_positions)
    is_grouped = group2_values is not None

    # === Validate color input consistency ===
    if isinstance(group1_color, list) and len(group1_color) != n_items:
        raise ValueError(
            f"group1_color list length ({len(group1_color)}) must match number of x_positions ({n_items})"
        )
    if is_grouped and isinstance(group2_color, list) and len(group2_color) != n_items:
        raise ValueError(
            f"group2_color list length ({len(group2_color)}) must match number of x_positions ({n_items})"
        )

    # === Convert confidence intervals to yerr arrays ===
    if group1_cis is None:
        group1_cis = [(v, v) for v in group1_values]
    g1_yerr = _convert_ci_to_yerr(group1_cis, group1_values)

    # === Set a "Nature-style" narrow bar width ===
    bw = 0.35 * bar_width_multiplier

    # ---------------------------------------------------------------------
    # GROUPED MODE: Two sets of bars (side-by-side)
    # ---------------------------------------------------------------------
    if is_grouped:
        if group2_cis is None:
            group2_cis = [(v, v) for v in group2_values]
        g2_yerr = _convert_ci_to_yerr(group2_cis, group2_values)

        use_hatch = _should_use_hatching(group1_color, group2_color)

        # Draw both bar groups
        bars1 = ax.bar(
            x_positions - bw/2, group1_values,
            width=bw, color=group1_color,
            edgecolor=params['bar_edgecolor'],
            linewidth=params['bar_linewidth'],
            alpha=params['bar_alpha'],
            label=group1_label if isinstance(group1_label, str) else "Group 1",
        )
        bars2 = ax.bar(
            x_positions + bw/2, group2_values,
            width=bw, color=group2_color,
            edgecolor=params['bar_edgecolor'],
            linewidth=params['bar_linewidth'],
            alpha=params['bar_alpha'],
            hatch=params['bar_hatch_novice'] if use_hatch else None,
            label=group2_label,
        )

        # Optional: add error bars for both groups
        if show_errorbars:
            ax.errorbar(x_positions - bw/2, group1_values, yerr=g1_yerr,
                        fmt='none', ecolor='black',
                        elinewidth=params['errorbar_linewidth'],
                        capsize=params['errorbar_capsize'], zorder=2)
            ax.errorbar(x_positions + bw/2, group2_values, yerr=g2_yerr,
                        fmt='none', ecolor='black',
                        elinewidth=params['errorbar_linewidth'],
                        capsize=params['errorbar_capsize'], zorder=2)

        # Optional numeric labels on bars
        if add_value_labels:
            for bar, val in zip(bars1, group1_values):
                if np.isfinite(val):
                    h = bar.get_height()
                    off = 0.04 if h >= 0 else -0.04
                    va = 'bottom' if h >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2, h + off,
                            f'{val:{value_label_format}}', ha='center', va=va,
                            fontsize=params['font_size_tick'] - 1)
            for bar, val in zip(bars2, group2_values):
                if np.isfinite(val):
                    h = bar.get_height()
                    off = 0.04 if h >= 0 else -0.04
                    va = 'bottom' if h >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2, h + off,
                            f'{val:{value_label_format}}', ha='center', va=va,
                            fontsize=params['font_size_tick'] - 1)

        # Significance annotations
        if group1_pvals is not None:
            _add_significance_stars(ax, x_positions - bw/2, group1_values, g1_yerr, group1_pvals, params)
        if group2_pvals is not None:
            _add_significance_stars(ax, x_positions + bw/2, group2_values, g2_yerr, group2_pvals, params)
        if comparison_pvals is not None:
            _add_significance_stars(ax, x_positions, group1_values, g1_yerr,
                                    comparison_pvals, params, comparison_mode=True,
                                    group2_values=group2_values, group2_yerr=g2_yerr, bar_width=bw)

        # Axis limits and zero line
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            auto_ymin, auto_ymax = _calculate_auto_ylim(
                group1_values, g1_yerr, pvals=group1_pvals,
                group2_values=group2_values, group2_yerr=g2_yerr,
                group2_pvals=group2_pvals, comparison_pvals=comparison_pvals,
                params=params)
            ax.set_ylim(auto_ymin, auto_ymax)
        # Baseline (zero) - DRY: use centralized alpha and linewidth
        ax.axhline(
            0,
            color='gray',
            linestyle=':',
            linewidth=params['reference_line_width'],  # Was 0.75
            alpha=params['reference_line_alpha'],      # Was 0.6
            zorder=1
        )

        # Unified DRY formatting (spines, titles, ticks, etc.)
        _apply_dry_formatting(
            ax, x_positions,
            y_label=y_label, title=title, subtitle=subtitle,
            xtick_labels=xtick_labels, x_label_colors=x_label_colors,
            x_tick_rotation=x_tick_rotation, x_tick_align=x_tick_align,
            hide_xticklabels=hide_xticklabels,
            show_legend=show_legend, legend_loc=legend_loc,
            visible_spines=visible_spines, params=params,
        )
        return g1_yerr, g2_yerr

    # ---------------------------------------------------------------------
    # SINGLE-GROUP MODE: One set of bars (with optional per-color legend)
    # ---------------------------------------------------------------------
    else:
        bars = ax.bar(
            x_positions, group1_values,
            width=bw, color=group1_color,
            edgecolor=params['bar_edgecolor'],
            linewidth=params['bar_linewidth'],
            alpha=params['bar_alpha'],
            label=group1_label if isinstance(group1_label, str) else None,
        )

        # Optional error bars
        if show_errorbars:
            ax.errorbar(x_positions, group1_values, yerr=g1_yerr,
                        fmt='none', ecolor='black',
                        elinewidth=params['errorbar_linewidth'],
                        capsize=params['errorbar_capsize'], zorder=2)

        # --- NEW FEATURE: Per-color legend from real bars ---
        legend_was_drawn = False
        if isinstance(group1_color, list) and isinstance(group1_label, list):
            legend_was_drawn = _apply_color_legend_from_bars(
                ax=ax,
                bars=bars,
                per_bar_colors=group1_color,
                per_color_labels=group1_label,
                legend_loc=legend_loc,
                params=params,
            )

        # Optional numeric labels on bars
        if add_value_labels:
            for bar, val in zip(bars, group1_values):
                if np.isfinite(val):
                    h = bar.get_height()
                    off = 0.04 if h >= 0 else -0.04
                    va = 'bottom' if h >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2, h + off,
                            f'{val:{value_label_format}}',
                            ha='center', va=va,
                            fontsize=params['font_size_tick'] - 1)

        # Significance stars
        if group1_pvals is not None:
            _add_significance_stars(ax, x_positions, group1_values, g1_yerr, group1_pvals, params)

        # Auto or fixed Y limits
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            auto_ymin, auto_ymax = _calculate_auto_ylim(
                group1_values, g1_yerr, pvals=group1_pvals, params=params
            )
            ax.set_ylim(auto_ymin, auto_ymax)

        # Baseline (zero) - DRY: use centralized alpha and linewidth
        ax.axhline(
            0,
            color='gray',
            linestyle=':',
            linewidth=params['reference_line_width'],  # Was 0.75
            alpha=params['reference_line_alpha'],      # Was 0.6
            zorder=1
        )

        # --- Final formatting ---
        # If legend was already drawn (using bar artists), suppress rebuilding it.
        _apply_dry_formatting(
            ax, x_positions,
            y_label=y_label, title=title, subtitle=subtitle,
            xtick_labels=xtick_labels, x_label_colors=x_label_colors,
            x_tick_rotation=x_tick_rotation, x_tick_align=x_tick_align,
            hide_xticklabels=hide_xticklabels,
            show_legend=(False if legend_was_drawn else show_legend),
            legend_loc=legend_loc,
            visible_spines=visible_spines, params=params,
        )

        return g1_yerr

def _apply_color_legend_from_bars(
    ax,
    bars,
    per_bar_colors: List[str],
    per_color_labels: List[str],
    legend_loc: str,
    params: dict,
) -> bool:
    """
    Internal utility to build a legend from the *actual bar rectangles*
    when each bar has its own color, but we only want one legend entry
    per unique color (e.g., "Exp > Nov", "Nov > Exp").

    This ensures the legend exactly matches the bar style — including
    alpha, edgecolor, linewidth, and hatch — because the handles are the
    original bar artists, not generic proxy patches.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to which the legend will be added.
    bars : list of matplotlib.patches.Rectangle
        Bar artists returned by `ax.bar(...)`.
    per_bar_colors : list of str
        List of facecolors used for each bar (same length as `bars`).
    per_color_labels : list of str
        List of legend labels corresponding to *unique colors*.
        Must have the same length as the number of unique colors in
        `per_bar_colors` (asserted).
    legend_loc : str
        Matplotlib legend location string (e.g., 'upper right').
    params : dict
        Plotting parameter dictionary. Should include font sizes and
        any custom legend style options, e.g.:
        - 'font_size_legend'
        - 'legend_frameon'

    Returns
    -------
    bool
        True if a legend was successfully created; False if skipped
        (e.g., due to invalid input types).

    Notes
    -----
    - The function preserves the first appearance order of unique colors.
    - Only the *first* bar for each color keeps its label;
      all subsequent bars are hidden from the legend (`"_nolegend_"`).
    - If the provided label count doesn’t match the number of unique
      colors, a ValueError is raised.
    """
    # === Validate input types ===
    if not isinstance(per_bar_colors, list) or not isinstance(per_color_labels, list):
        # Only activate this logic when both colors and labels are lists.
        return False
    if len(per_bar_colors) != len(bars):
        raise ValueError(
            f"per_bar_colors length ({len(per_bar_colors)}) must match number of bars ({len(bars)})."
        )

    # === Determine unique colors and the first bar for each color ===
    unique_colors: List[str] = []
    first_rect_for_color = {}
    for rect, color in zip(bars, per_bar_colors):
        if color not in first_rect_for_color:
            first_rect_for_color[color] = rect
            unique_colors.append(color)

    # === Validate label count ===
    if len(per_color_labels) != len(unique_colors):
        raise ValueError(
            f"Number of legend labels ({len(per_color_labels)}) must equal number of unique colors ({len(unique_colors)})."
        )

    # === Assign labels to bars ===
    color_to_label = {c: lab for c, lab in zip(unique_colors, per_color_labels)}
    seen = set()
    for rect, color in zip(bars, per_bar_colors):
        if color not in seen:
            # Assign the legend label only to the first occurrence
            rect.set_label(color_to_label[color])
            seen.add(color)
        else:
            # Hide subsequent bars of the same color from the legend
            rect.set_label("_nolegend_")

    # === Build and draw the legend using real bar rectangles ===
    handles = [first_rect_for_color[c] for c in unique_colors]
    labels = [color_to_label[c] for c in unique_colors]
    ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        frameon=params.get("legend_frameon", False),
        fontsize=params.get("font_size_legend", None),
    )

    return True

def _apply_dry_formatting(
    ax,
    x_positions,
    *,
    y_label: Optional[str],
    title: Optional[str],
    subtitle: Optional[str],
    xtick_labels: Optional[List[str]],
    x_label_colors: Optional[List[str]],
    x_tick_rotation: int,
    x_tick_align: str,
    hide_xticklabels: bool,
    show_legend: Optional[bool],
    legend_loc: str,
    visible_spines: Optional[List[str]],
    params: dict,
):
    """Apply labeling, axis styling, and optional legend in a DRY fashion."""
    from .helpers import label_axes, set_axis_title, style_spines

    # X ticks and labels
    if hide_xticklabels:
        ax.set_xlim(-0.5, len(x_positions) - 0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([])
    elif xtick_labels is not None:
        ax.set_xlim(-0.5, len(x_positions) - 0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xtick_labels, rotation=x_tick_rotation, ha=x_tick_align,
                           fontsize=params['font_size_tick'])
        if x_label_colors is not None:
            for ticklabel, color in zip(ax.get_xticklabels(), x_label_colors):
                ticklabel.set_color(color)

    # Y label
    if y_label is not None:
        label_axes(ax, ylabel=y_label, params=params)

    # Title/subtitle
    if title is not None or subtitle is not None:
        set_axis_title(ax, title=title or '', subtitle=subtitle or '', params=params)

    # Legend
    if show_legend is True:
        ax.legend(loc=legend_loc, ncol=2, frameon=False,
                  fontsize=params['font_size_legend'])

    # Spines
    if visible_spines is not None:
        style_spines(ax, visible_spines=visible_spines, params=params)


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
        # Comparison reference line - DRY: use centralized alpha
        ax.axhline(
            0,
            color='black',
            linestyle='--',
            linewidth=params['plot_linewidth'],
            alpha=params['comparison_line_alpha'],  # Was 0.3
            zorder=1
        )

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
