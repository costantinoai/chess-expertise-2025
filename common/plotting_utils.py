"""
Centralized plotting utilities for all analyses.

Design Principles:
- NO MAGIC NUMBERS: All sizes computed from base values
- Single clean dictionary (PLOT_PARAMS)
- Simple, no complicated fallbacks or logic
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Optional, List, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap
from .constants import CONFIG
from .formatters import significance_stars


# ============================================================================
# Colormaps for Plotting
# ============================================================================

def _make_brain_cmap():
    """
    Create brain colormap for RDMs and brain maps.

    This is the ONLY colormap for ALL RDMs (no exceptions).
    RDM value 0 should always be at the center of this colormap.

    Colormap structure:
    - Negative values: Cyan/Teal gradient
    - Center (0): RdPu(0) color
    - Positive values: RdPu gradient (pink to dark purple)

    Returns
    -------
    LinearSegmentedColormap
        Custom brain colormap

    Notes
    -----
    - Always use with center=0 when plotting RDMs
    - vmin/vmax should be symmetric around 0

    Example
    -------
    >>> from common.plotting_utils import CMAP_BRAIN
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(rdm, cmap=CMAP_BRAIN, center=0)
    """
    center = plt.cm.RdPu(0)[:3]  # Get RGB of RdPu at 0

    # Negative range: cyan/teal to center color
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)

    # Positive range: RdPu gradient
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]

    # Combine negative and positive
    colors = np.vstack((neg, pos))

    return LinearSegmentedColormap.from_list("brain_rdm", colors)

# Brain colormap - ONLY colormap for RDMs
CMAP_BRAIN = _make_brain_cmap()

# ============================================================================
# Color Palettes
# ============================================================================

# 1. Checkmate vs Non-Checkmate (for behavioral visualizations)
# Used for stimulus visualizations in behavioral RDMs, MDS, etc.
COLORS_CHECKMATE_NONCHECKMATE = {
    'checkmate': "#B96F25",
    'non_checkmate': "#305B7F",
}

# 2. Expert vs Novice
COLORS_EXPERT_NOVICE = {
    'expert': "#337538",  
    'novice': "#7e2954", 
}

# ============================================================================
# Base Plotting Parameters
# ============================================================================

# Base values - change these to scale everything
_BASE_FONT_SIZE = 20
_BASE_LINE_WIDTH = 1.5

# All plotting parameters - computed from base values (NO MAGIC NUMBERS)
PLOT_PARAMS = {
    # Base values
    'base_font_size': _BASE_FONT_SIZE,
    'base_line_width': _BASE_LINE_WIDTH,

    # Font sizes (base * ratio)
    'font_size_title': _BASE_FONT_SIZE * 1.6,
    'font_size_subtitle': _BASE_FONT_SIZE * 1.5,
    'font_size_label': _BASE_FONT_SIZE * 1.3,
    'font_size_tick': _BASE_FONT_SIZE * 1.3,
    'font_size_legend': _BASE_FONT_SIZE * 1.3,

    # Line widths (base * ratio)
    'spine_linewidth': _BASE_LINE_WIDTH * 1.0,
    'axes_linewidth': _BASE_LINE_WIDTH * 0.33,
    'errorbar_linewidth': _BASE_LINE_WIDTH * 0.8,
    'bar_linewidth': _BASE_LINE_WIDTH * 0.67,
    'plot_linewidth': _BASE_LINE_WIDTH * 0.67,

    # Tick sizes (spine_linewidth * ratio)
    'tick_major_size': _BASE_LINE_WIDTH * 1.0 * 1.67,
    'tick_minor_size': _BASE_LINE_WIDTH * 1.0 * 1.33,
    'tick_major_width': _BASE_LINE_WIDTH * 1.0,
    'tick_minor_width': _BASE_LINE_WIDTH * 1.0,

    # Marker size
    'marker_size': _BASE_FONT_SIZE * 0.15,

    # Figure sizes (inches)
    'figure_sizes': {
        'large': (10, 8),
        'single_panel': (1.75, 1.75),
        'two_panel': (3.5, 1.75),
        'four_panel': (3.5, 3.5),
        'full_width': (7.0, 3.5),
    },

    # Figure properties
    'facecolor': 'white',
    'dpi': 300,
    'format': 'pdf',

    # Bar parameters
    'bar_width': 0.5,
    'bar_alpha': 1,
    'bar_edgecolor': 'black',
    'bar_hatch_novice': '//',

    # Error bar parameters
    'errorbar_capsize': 0,

    # Title spacing
    'title_pad': 25,

    # Font
    # 'font_family': 'Helvetica Neue LT Std',
    'font_family': 'Arial',

    # RDM category bar offset (outside axis; negative for below/left)
    'rdm_category_bar_offset': -0.06,
}

__all__ = [
    'CMAP_BRAIN',
    'COLORS_EXPERT_NOVICE',
    'COLORS_CHECKMATE_NONCHECKMATE',
    'PLOT_PARAMS',
    'figure_style',
    'style_spines',
    'hide_ticks',
    'set_axis_title',
    'add_rdm_category_bars',
    'add_roi_color_legend',
    'plot_grouped_bars_with_ci',  # Standalone figure with single/grouped bars
    'plot_grouped_bars_on_ax',    # Multi-panel: grouped bars
    'compute_stimulus_palette',
    'plot_rdm',
    'plot_2d_embedding',
    'select_roi_labels_for_plot',
]

# Model display order and labels are provided via common.constants


def figure_style(params: dict = PLOT_PARAMS):
    """
    Apply plotting parameters to matplotlib/seaborn.

    Parameters
    ----------
    params : dict
        Plotting parameters dictionary
    """
    sns.set(
        style="ticks",
        context="paper",
        font=params['font_family'],
        rc={
            "font.size": params['base_font_size'],
            "font.family": "sans-serif",
            "font.sans-serif": [params['font_family'], 'Helvetica', 'Arial'],
            "figure.titlesize": params['font_size_title'],
            "axes.titlesize": params['font_size_title'],
            "axes.labelsize": params['font_size_label'],
            "axes.linewidth": params['axes_linewidth'],
            "lines.linewidth": params['plot_linewidth'],
            "lines.markersize": params['marker_size'],
            "xtick.labelsize": params['font_size_tick'],
            "ytick.labelsize": params['font_size_tick'],
            "savefig.transparent": True,
            "xtick.major.size": params['tick_major_size'],
            "ytick.major.size": params['tick_major_size'],
            "xtick.major.width": params['tick_major_width'],
            "ytick.major.width": params['tick_major_width'],
            "xtick.minor.size": params['tick_minor_size'],
            "ytick.minor.size": params['tick_minor_size'],
            "xtick.minor.width": params['tick_minor_width'],
            "ytick.minor.width": params['tick_minor_width'],
            'legend.fontsize': params['font_size_legend'],
            'legend.title_fontsize': params['font_size_legend'],
            'legend.frameon': False,
        }
    )

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def style_spines(ax, visible_spines: List[str] = ['left', 'bottom'],
                 params: dict = PLOT_PARAMS):
    """Apply spine styling."""
    linewidth = params['spine_linewidth']
    for spine_loc in ['left', 'right', 'top', 'bottom']:
        if spine_loc in visible_spines:
            ax.spines[spine_loc].set_visible(True)
            ax.spines[spine_loc].set_linewidth(linewidth)
            ax.spines[spine_loc].set_edgecolor('black')
        else:
            ax.spines[spine_loc].set_visible(False)


def hide_ticks(ax, hide_x: bool = True, hide_y: bool = True):
    """Hide tick labels."""
    if hide_x:
        ax.set_xticks([])
    if hide_y:
        ax.set_yticks([])


def set_axis_title(ax, title: str, subtitle: Optional[str] = None,
                   params: dict = PLOT_PARAMS):
    """Set axis title with optional subtitle."""
    title_size = params['font_size_title']
    subtitle_size = params['font_size_subtitle']
    pad_pts = params['title_pad']  # in points

    def _escape_mathtext(s: str) -> str:
        return (
            s.replace("\\", r"\\")
             .replace("_", r"\_")
             .replace("%", r"\%")
        )

    title = _escape_mathtext(title)
    subtitle = _escape_mathtext(subtitle) if subtitle is not None else None

    if subtitle is None:
        full_title = f"$\\mathbf{{{title}}}$"
        ax.set_title(full_title, fontsize=title_size, pad=pad_pts)
        return

    # manual 2-line layout
    ax.set_title("")  # clear default title

    fig = ax.get_figure()

    # axes height in inches
    bbox_axes_in_fig = ax.get_position()
    fig_h_in = fig.get_figheight()
    axes_h_in = bbox_axes_in_fig.height * fig_h_in

    # points -> axes coord
    pts_to_axes = (1.0/72.0) / axes_h_in

    # y position of the title baseline
    title_y = 1.0 + pad_pts * pts_to_axes

    ax.text(
        0.5, title_y, title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight='bold',
        ha='center', va='bottom'
    )

    # --- tighter subtitle placement ---
    #
    # How far below the title should the subtitle sit?
    # Instead of 1.0 * title_size, use a fraction.
    line_height_factor = 0.75  # <-- tune this
    line_gap_extra_pts = 0.0  # <-- your "line gap pts"; set to 0 for no extra spacing

    subtitle_offset_pts = title_size * line_height_factor + line_gap_extra_pts

    subtitle_y = title_y - subtitle_offset_pts * pts_to_axes

    ax.text(
        0.5, subtitle_y, subtitle,
        transform=ax.transAxes,
        fontsize=subtitle_size,
        fontweight='normal',
        ha='center', va='bottom'
    )




def select_roi_labels_for_plot(
    welch_df,
    default_names: List[str],
    default_colors: List[str],
    fallback_color: str = '#4C78A8'
):
    """
    Choose ROI labels and colors for plotting based on Welch results.

    If the Welch DataFrame contains 'ROI_Name' and its length differs from
    default_names, use the Welch ROI list with uniform fallback color.
    Otherwise use defaults.
    """
    try:
        has_roi_col = hasattr(welch_df, 'columns') and ('ROI_Name' in welch_df.columns)
    except Exception:
        has_roi_col = False
    if has_roi_col and len(welch_df) != len(default_names):
        names = welch_df['ROI_Name'].tolist()
        colors = [fallback_color] * len(names)
        return names, colors
    return default_names, default_colors


def add_rdm_category_bars(ax, colors: List[str], alphas: Optional[List[float]] = None,
                          axis: str = 'both', thickness: float = 0.035):
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
    thickness : float, default=0.035
        Thickness of bars as fraction of axis length

    Notes
    -----
    - Uses axis transforms (get_xaxis_transform, get_yaxis_transform) to position
      bars outside the plot area
    - Bars are placed just outside axis (offset set by PLOT_PARAMS['rdm_category_bar_offset'])
    - Automatically groups consecutive items with same color+alpha
    - Uses clip_on=False to allow bars to extend beyond axis limits

    Example
    -------
    >>> colors = ['red']*10 + ['blue']*10
    >>> alphas = [0.5]*5 + [1.0]*5 + [0.5]*5 + [1.0]*5
    >>> add_rdm_category_bars(ax, colors, alphas, axis='both')
    """
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

    # Add bars along x-axis (bottom of plot)
    offset = PLOT_PARAMS.get('rdm_category_bar_offset', -0.06)
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
    ylabel: str = "Value",
    title: str = "Comparison",
    subtitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    add_zero_line: bool = False,
    legend_loc: str = "upper left",
    show_legend: bool = True,
    params: dict = PLOT_PARAMS,
    return_ax: bool = False
) -> Union[plt.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Unified bar chart with error bars and significance stars.

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
    ylabel : str
        Y-axis label
    title, subtitle : str
        Plot title and subtitle
    output_path : Path, optional
        If provided, saves figure and closes it
    figsize : tuple, optional
        Figure size (defaults to PLOT_PARAMS['figure_sizes']['large'])
    ylim : tuple, optional
        Y-axis limits (min, max)
    add_zero_line : bool, default=False
        Whether to add horizontal line at y=0
    legend_loc : str, default='upper left'
        Legend location
    show_legend : bool, default=True
        Whether to show legend. For single-group plots, often set to False.
    params : dict
        Plotting parameters
    return_ax : bool, default=False
        If True, returns (fig, ax) instead of just fig

    Returns
    -------
    plt.Figure or (plt.Figure, plt.Axes)
        Figure object (and axes if return_ax=True)

    Notes
    -----
    - When using per-item colors, both groups at same x-position use the same color
    - Group 1 bars are always solid
    - Group 2 bars are hatched ONLY when groups share the same colors (e.g., ROI colors)
    - Group 2 bars are NOT hatched when groups have different colors (e.g., expert green vs novice red)
    - Automatically handles single vs per-item color modes
    - For single-group plots (group2_values=None), only plots group1 bars

    Examples
    --------
    # Single-group plot (replaces plot_bars_with_ci)
    >>> plot_grouped_bars_with_ci(
    ...     group1_values=diffs, group1_cis=cis,
    ...     x_labels=['Term1', 'Term2'], group1_pvals=pvals,
    ...     group1_color='blue', show_legend=False
    ... )

    # Two-group comparison (behavioral analysis)
    >>> plot_grouped_bars_with_ci(
    ...     group1_values=expert_vals, group1_cis=expert_cis,
    ...     group2_values=novice_vals, group2_cis=novice_cis,
    ...     x_labels=['Visual', 'Strategy', 'Checkmate'],
    ...     group1_color='green', group2_color='red'
    ... )

    # Per-item colors (ROI analysis)
    >>> roi_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    >>> plot_grouped_bars_with_ci(
    ...     group1_values=expert_vals, group1_cis=expert_cis,
    ...     group2_values=novice_vals, group2_cis=novice_cis,
    ...     x_labels=['V1', 'MT', 'IPS'],
    ...     group1_color=roi_colors, group2_color=roi_colors
    ... )
    """
    if figsize is None:
        figsize = params['figure_sizes']['large']

    # Determine if single-group or grouped plot
    is_grouped = group2_values is not None

    # Set default colors if not provided
    if group1_color is None:
        group1_color = COLORS_EXPERT_NOVICE['expert'] if is_grouped else '#999999'
    if group2_color is None and is_grouped:
        group2_color = COLORS_EXPERT_NOVICE['novice']

    # Create figure and axis
    figure_style(params)
    fig, ax = plt.subplots(figsize=figsize, facecolor=params['facecolor'])

    x_positions = np.arange(len(x_labels))

    if is_grouped:
        # Use core grouped plotting function
        plot_grouped_bars_on_ax(
            ax=ax,
            x_positions=x_positions,
            group1_values=group1_values,
            group2_values=group2_values,
            group1_cis=group1_cis,
            group2_cis=group2_cis,
            group1_label=group1_label,
            group2_label=group2_label,
            group1_color=group1_color,
            group2_color=group2_color,
            group1_pvals=group1_pvals,
            group2_pvals=group2_pvals,
            comparison_pvals=comparison_pvals,
            params=params
        )
    else:
        # Single-group plot
        yerr = _convert_ci_to_yerr(group1_cis, group1_values)

        # Bars
        ax.bar(
            x_positions,
            group1_values,
            width=params['bar_width'],
            color=group1_color,
            edgecolor=params['bar_edgecolor'],
            linewidth=params['bar_linewidth'],
            alpha=params['bar_alpha'],
            label=group1_label
        )

        # Error bars
        ax.errorbar(
            x_positions,
            group1_values,
            yerr=yerr,
            fmt='none',
            ecolor='black',
            elinewidth=params['errorbar_linewidth'],
            capsize=params['errorbar_capsize'],
            zorder=2
        )

        # Significance stars
        if group1_pvals is not None:
            _add_significance_stars(ax, x_positions, group1_values, yerr, group1_pvals, params)

    # Add labels and styling (figure-level)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_ylabel(ylabel, fontsize=params['font_size_label'])

    if ylim is not None:
        ax.set_ylim(ylim)

    if add_zero_line:
        ax.axhline(0, color='black', linestyle='--',
                  linewidth=params['plot_linewidth'], alpha=0.3, zorder=1)

    set_axis_title(ax, title, subtitle=subtitle, params=params)

    if show_legend:
        ax.legend(loc=legend_loc, ncol=2, frameon=False,
                  fontsize=params['font_size_legend'])

    plt.tight_layout()
    sns.despine(trim=False)
    style_spines(ax, visible_spines=['left', 'bottom'], params=params)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                   facecolor=params['facecolor'], dpi=params['dpi'])
        plt.close(fig)

    if return_ax:
        return fig, ax
    return fig


## plot_bars_with_ci has been removed; use plot_grouped_bars_with_ci in single-group mode


def compute_stimulus_palette(stimuli_df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Compute colors and alpha transparencies for chess stimuli visualization.

    Assigns colors by checkmate status and alpha by strategy within each
    checkmate group, preserving the input row order of stimuli_df.

    Parameters
    ----------
    stimuli_df : pd.DataFrame
        Stimulus metadata with columns: 'stim_id', 'check', 'strategy'.
        The returned lists are aligned to the order of rows in this DataFrame.

    Returns
    -------
    colors : list of str
        Hex color per stimulus (checkmate vs non_checkmate)
    alphas : list of float
        Alpha in (0, 1] per stimulus, scaled by strategy within check groups

    Notes
    -----
    - Colors come from COLORS_CHECKMATE_NONCHECKMATE in this module
    - Alpha mapping is uniform across the number of unique strategies in each
      check group, using increasing steps from 1/n .. 1.0
    - The function does not reorder; it preserves the row order of stimuli_df
    """
    # Validate expected columns
    expected = {"check", "strategy"}
    missing = expected - set(stimuli_df.columns)
    if missing:
        raise ValueError(f"stimuli_df missing required columns: {sorted(missing)}")

    # Build per-check group alpha mapping
    alpha_maps = {}
    for check_value in ["checkmate", "non_checkmate"]:
        group = stimuli_df[stimuli_df["check"] == check_value]
        if len(group) == 0:
            continue
        unique_strategies = list(dict.fromkeys(group["strategy"].tolist()))
        n_strat = max(1, len(unique_strategies))
        # Alphas 1/n .. 1.0 in equal steps
        alphas_seq = [(i + 1) / n_strat for i in range(n_strat)]
        alpha_maps[check_value] = {s: a for s, a in zip(unique_strategies, alphas_seq)}

    # Assign color/alpha preserving original row order
    colors, alphas = [], []
    for _, row in stimuli_df.iterrows():
        is_checkmate = str(row["check"]) == "checkmate"
        color = COLORS_CHECKMATE_NONCHECKMATE['checkmate' if is_checkmate else 'non_checkmate']
        strategy = row["strategy"]
        alpha = alpha_maps.get('checkmate' if is_checkmate else 'non_checkmate', {}).get(strategy, 1.0)
        colors.append(color)
        alphas.append(alpha)

    return colors, alphas


def _should_use_hatching(group1_color: Union[str, List[str]],
                        group2_color: Union[str, List[str]]) -> bool:
    """
    Determine if hatching should be used to distinguish groups.

    Hatching is needed when groups share the same colors (e.g., ROI-specific colors),
    but NOT when groups have distinct colors (e.g., expert green vs novice red).

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
    >>> _should_use_hatching('#198019', '#a90f0f')  # Different colors
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
    # Calculate dynamic offset (2% of y-range)
    y_offset = _calculate_offset_from_range(ax, offset_pct=0.02)

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

                # Line is at the offset level (bottom reference)
                line_y = max_top + y_offset

                # Draw connecting line first (at offset level)
                ax.plot(
                    [x_positions[i] - bar_width/2, x_positions[i] + bar_width/2],
                    [line_y, line_y],
                    color='black', linewidth=1, zorder=3
                )

                # Star is above the line
                star_y = line_y + y_offset * 0.5

                # Draw star
                ax.text(x_positions[i], star_y, stars,
                       ha='center', va='bottom',
                       fontsize=params['base_font_size'],
                       fontweight='bold')
    else:
        # Single-bar mode (above/below individual bars)
        for x, val, err, pval in zip(x_positions, values, yerr.T, pvals):
            stars = significance_stars(pval)
            if stars:
                # Determine reference point and direction based on bar value
                if val >= 0:
                    # Positive bar: star above
                    ref_y = val + err[1]  # Top of error bar
                    star_y = ref_y + y_offset
                    va = 'bottom'
                else:
                    # Negative bar: star below
                    ref_y = val - err[0]  # Bottom of error bar
                    star_y = ref_y - y_offset
                    va = 'top'

                # Draw star
                ax.text(x, star_y, stars,
                       ha='center', va=va,
                       fontsize=params['base_font_size'],
                       fontweight='bold')


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
    bar_width_multiplier: float = 1.0,
    params: dict = PLOT_PARAMS
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
    bar_width_multiplier : float, default=1.0
        Multiplier for bar width. Use 2.0 for single-group plots to make bars wider.
    params : dict
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
    ...     comparison_pvals=pvals
    ... )
    """
    bw = params['bar_width'] * bar_width_multiplier
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

    if is_grouped:
        # === GROUPED MODE: Two groups side-by-side ===
        g2_yerr = _convert_ci_to_yerr(group2_cis, group2_values)

        # Determine if hatching is needed (only when colors are the same)
        use_hatch = _should_use_hatching(group1_color, group2_color)

        # Group 1 bars (solid)
        ax.bar(x_positions - bw/2, group1_values,
               width=bw, color=group1_color,
               edgecolor=params['bar_edgecolor'],
               linewidth=params['bar_linewidth'],
               alpha=params['bar_alpha'],
               label=group1_label)

        # Group 1 error bars
        ax.errorbar(x_positions - bw/2, group1_values, yerr=g1_yerr,
                    fmt='none', ecolor='black',
                    elinewidth=params['errorbar_linewidth'],
                    capsize=params['errorbar_capsize'], zorder=2)

        # Group 2 bars (hatched only if colors are the same)
        ax.bar(x_positions + bw/2, group2_values,
               width=bw, color=group2_color,
               edgecolor=params['bar_edgecolor'],
               linewidth=params['bar_linewidth'],
               alpha=params['bar_alpha'],
               hatch=params['bar_hatch_novice'] if use_hatch else None,
               label=group2_label)

        # Group 2 error bars
        ax.errorbar(x_positions + bw/2, group2_values, yerr=g2_yerr,
                    fmt='none', ecolor='black',
                    elinewidth=params['errorbar_linewidth'],
                    capsize=params['errorbar_capsize'], zorder=2)

        # Within-group significance stars
        if group1_pvals is not None:
            _add_significance_stars(
                ax, x_positions - bw/2, group1_values, g1_yerr, group1_pvals, params
            )
        if group2_pvals is not None:
            _add_significance_stars(
                ax, x_positions + bw/2, group2_values, g2_yerr, group2_pvals, params
            )

        # Between-group comparison stars
        if comparison_pvals is not None:
            _add_significance_stars(
                ax, x_positions, group1_values, g1_yerr, comparison_pvals, params,
                comparison_mode=True,
                group2_values=group2_values,
                group2_yerr=g2_yerr,
                bar_width=bw
            )

        return g1_yerr, g2_yerr

    else:
        # === SINGLE-GROUP MODE: One set of bars ===
        # Bars (centered, wider if multiplier > 1)
        ax.bar(x_positions, group1_values,
               width=bw, color=group1_color,
               edgecolor=params['bar_edgecolor'],
               linewidth=params['bar_linewidth'],
               alpha=params['bar_alpha'],
               label=group1_label)

        # Error bars
        ax.errorbar(x_positions, group1_values, yerr=g1_yerr,
                    fmt='none', ecolor='black',
                    elinewidth=params['errorbar_linewidth'],
                    capsize=params['errorbar_capsize'], zorder=2)

        # Significance stars
        if group1_pvals is not None:
            _add_significance_stars(
                ax, x_positions, group1_values, g1_yerr, group1_pvals, params
            )

        return g1_yerr


def add_roi_color_legend(ax, roi_info_df, ncol: int = 4, loc: str = 'upper center',
                         bbox_to_anchor: Tuple[float, float] = (0.5, -0.35),
                         params: dict = PLOT_PARAMS):
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
    params : dict
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
    from matplotlib.patches import Patch

    # Get unique color/name combinations
    name_col = 'pretty_name' if 'pretty_name' in roi_info_df.columns else 'region_name'
    unique_pairs = roi_info_df[['color', name_col]].drop_duplicates(subset='color')

    # Create legend handles
    handles = [Patch(facecolor=row['color'], edgecolor='black', label=row[name_col])
               for _, row in unique_pairs.iterrows()]

    # Add legend
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor,
             ncol=ncol, frameon=False,
             fontsize=params['font_size_legend'] - 1,
             title='ROI Groups', title_fontsize=params['font_size_legend'])


def plot_rdm(
    rdm: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    colors: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    show_colorbar: bool = True,
    colorbar_label: str = "Dissimilarity",
    figsize: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    params: dict = PLOT_PARAMS
) -> plt.Figure:
    """
    Plot RDM as heatmap with colored category bars and optional colorbar.

    CRITICAL: ALL RDMs use CMAP_BRAIN with center=0 (no exceptions).
    CRITICAL: ALL RDMs show colored category bars on sides (always, if colors provided).
    CRITICAL: ALL RDMs show vertical colorbar on right (by default).

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
    figsize : tuple, optional
        Figure size (width, height) in inches
    vmin, vmax : float, optional
        Color scale limits. If not provided, computed as symmetric around 0
        based on max absolute value in this RDM. For consistent coloring across
        multiple RDMs, compute global max and pass explicitly.
    params : dict
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
    - For consistent coloring across multiple RDMs (e.g., expert vs novice),
      compute global vmin/vmax and pass to all plot_rdm calls

    Example
    -------
    >>> from common.plotting_utils import plot_rdm, CMAP_BRAIN
    >>> rdm = np.random.randn(20, 20)
    >>> colors = ['red'] * 10 + ['blue'] * 10
    >>> fig = plot_rdm(rdm, "Example RDM", colors=colors, output_path=Path("rdm.pdf"))

    >>> # For consistent coloring across multiple RDMs:
    >>> global_max = max(np.abs(rdm1).max(), np.abs(rdm2).max())
    >>> plot_rdm(rdm1, "RDM 1", vmin=-global_max, vmax=global_max)
    >>> plot_rdm(rdm2, "RDM 2", vmin=-global_max, vmax=global_max)
    """
    if figsize is None:
        figsize = params['figure_sizes']['large']

    figure_style(params)
    fig, ax = plt.subplots(figsize=figsize, facecolor=params['facecolor'])

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
        add_rdm_category_bars(ax, colors, alphas, axis='both')

    # Set title
    set_axis_title(ax, title, subtitle=subtitle, params=params)

    # Hide tick labels
    hide_ticks(ax, hide_x=True, hide_y=True)

    # Apply consistent styling (keep all 4 spines for heatmaps)
    plt.tight_layout()
    style_spines(ax, visible_spines=['left', 'right', 'top', 'bottom'], params=params)

    # Save if path provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                   facecolor=params['facecolor'], dpi=params['dpi'])
        plt.close(fig)

    return fig


def plot_2d_embedding(
    coords: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    point_colors: Optional[List[str]] = None,
    point_alphas: Optional[List[float]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    fill: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    params: dict = PLOT_PARAMS,
) -> plt.Figure:
    """
    Plot a 2D embedding scatter using precomputed coordinates.

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
    figsize : tuple, optional
        Figure size in inches. Defaults to params['figure_sizes']['large'].
    params : dict
        Centralized plotting parameters.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    if figsize is None:
        figsize = params['figure_sizes']['large']

    figure_style(params)
    fig, ax = plt.subplots(figsize=figsize, facecolor=params['facecolor'])

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

    for (x, y), c, a in zip(coords, point_colors, point_alphas):
        ax.scatter(x, y, color=c, marker='o', alpha=a, s=200)

    # Axis labels (ticks remain hidden per style)
    if x_label:
        ax.set_xlabel(x_label, fontsize=params['font_size_label'])
    if y_label:
        ax.set_ylabel(y_label, fontsize=params['font_size_label'])

    # Title and style
    set_axis_title(ax, title, subtitle=subtitle, params=params)
    hide_ticks(ax, hide_x=True, hide_y=True)
    plt.tight_layout()
    style_spines(ax, visible_spines=['left', 'right', 'top', 'bottom'], params=params)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                    facecolor=params['facecolor'], dpi=params['dpi'])
        plt.close(fig)

    return fig
