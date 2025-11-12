#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature-compliant plotting style configuration.

Provides:
- PLOT_PARAMS: Centralized parameter dictionary
- apply_nature_rc(): Apply rcParams to matplotlib/seaborn
- figure_size(): Compute Nature-compliant figure dimensions
- auto_bar_figure_size(): Compute figure size for fixed bar widths
"""

from typing import Tuple, Optional, Literal
import matplotlib
import seaborn as sns

# =============================================================================
# Physical Constants
# =============================================================================

CM_TO_INCHES = 1 / 2.54  # Centimeters to inches conversion factor
# Internal millimeter-to-inches for sizing helpers that accept mm
MM_TO_INCHES = 1 / 25.4
_MM = MM_TO_INCHES
_NATURE_SINGLE_COL_MM = 89.0
_NATURE_DOUBLE_COL_MM = 183.0
_NATURE_MAX_HEIGHT_MM = 170.0


def cm_to_inches(cm: float) -> float:
    """
    Convert centimeters to inches for figure sizing.

    Pylustrator and our panel templates use centimeters for figure dimensions,
    while matplotlib requires inches. This helper ensures consistent conversion
    across plotting scripts (DRY principle).

    Parameters
    ----------
    cm : float
        Size in centimeters

    Returns
    -------
    float
        Size in inches (cm / 2.54)

    Examples
    --------
    >>> from common.plotting import cm_to_inches
    >>> fig.set_size_inches(cm_to_inches(18.0), cm_to_inches(12.0))
    """
    return float(cm) * CM_TO_INCHES

# =============================================================================
# Font Hierarchy (Nature: 5-7pt, panel labels 8pt exception)
# =============================================================================

_FONT_SIZE_PANEL_LABEL = 8.0
_FONT_SIZE_TITLE = 7.0
_FONT_SIZE_LABEL = 6.0
_FONT_SIZE_TICK = 5.0
_FONT_SIZE_LEGEND = 5.0
_FONT_SIZE_ANNOTATION = 5.0

# =============================================================================
# Line Width Hierarchy (all derived from base)
# =============================================================================

_BASE_LINEWIDTH = 0.3  # Base line width in points
_COMPARISON_LINE_FRACTION = 1.0  # Comparison significance lines (same as base)

# =============================================================================
# Physical Bar Width
# =============================================================================

_TARGET_BAR_WIDTH_MM = 8.0

# =============================================================================
# PLOT_PARAMS Dictionary
# =============================================================================

PLOT_PARAMS = {
    # Font sizes (all 5-7pt except panel labels 8pt)
    'font_size_panel_label': _FONT_SIZE_PANEL_LABEL,
    'font_size_title': _FONT_SIZE_TITLE,
    'font_size_label': _FONT_SIZE_LABEL,
    'font_size_tick': _FONT_SIZE_TICK,
    'font_size_legend': _FONT_SIZE_LEGEND,
    'font_size_annotation': _FONT_SIZE_ANNOTATION * 1.4,  # Larger asterisks for visibility

    # Line widths (all derived from base linewidth)
    'base_linewidth': _BASE_LINEWIDTH,
    'spine_linewidth': _BASE_LINEWIDTH,
    'axes_linewidth': _BASE_LINEWIDTH,
    'errorbar_linewidth': _BASE_LINEWIDTH,
    'bar_linewidth': _BASE_LINEWIDTH,
    'plot_linewidth': _BASE_LINEWIDTH,
    'comparison_linewidth': _BASE_LINEWIDTH * _COMPARISON_LINE_FRACTION,
    'reference_line_width': _BASE_LINEWIDTH * 2.5,  # 0.75pt for zero/reference lines (DRY: was hardcoded as 0.75)

    # Tick parameters
    'tick_major_size': 3.0,
    'tick_minor_size': 1.5,
    'tick_major_width': _BASE_LINEWIDTH,
    'tick_minor_width': _BASE_LINEWIDTH,
    'tick_max_nbins': 6,  # MaxNLocator for legible tick counts

    # Marker size for scatter/MDS (Matplotlib scatter 's' in points^2)
    'marker_size': 8.0,

    # Alpha/transparency presets by element type (DRY: eliminates hardcoded alphas)
    'line_alpha': 0.5,                      # Generic line elements
    'bar_alpha': 0.7,                       # Bar chart fills
    'reference_line_alpha': 0.6,            # Zero reference lines (axhline at y=0)
    'comparison_line_alpha': 0.3,           # Comparison/chance-level reference lines
    'error_band_alpha': 0.2,                # Confidence interval bands (fill_between)
    'scatter_alpha': 0.6,                   # Scatter plot point transparency

    # Bar parameters
    'target_bar_width_mm': _TARGET_BAR_WIDTH_MM,
    'bar_edgecolor': 'black',
    'bar_hatch_novice': '////',

    # Error bars (capsize set to 0 - no caps)
    'errorbar_capsize': 0,

    # Spacing
    'title_pad': 10.0,  # Points above plot
    # Vertical gap between title and subtitle, as a multiple of title font size
    # Increased for better separation (was implicitly ~0.9× in helper)
    'title_subtitle_gap_factor': 1.2,
    'panel_label_offset_mm': (-5, 6),  # (x, y) from upper-left corner (increased y for higher placement)
    'significance_offset_pct': 0.02,  # Significance star offset as percentage of y-range (2%)

    # Export settings
    'dpi': 450,
    'format': 'pdf',
    'font_family': 'Arial',
    'facecolor': 'white',
    'transparent': False,

    # RDM-specific
    'rdm_category_bar_offset': -0.06,
    'rdm_category_bar_thickness': 0.035,

    # Canonical labels (DRY)
    'ylabel_correlation_r': 'Correlation (r)',
}

# =============================================================================
# Y-Axis Limit Presets by Analysis Type
# =============================================================================
#
# Consistent y-axis limits for bar plots of the same analysis type.
# These ensure visual comparability across figures and consistency within
# analysis categories (e.g., all RSA plots have same scale).
#
# **Usage Guidelines:**
# - Apply these limits during plot creation (before Pylustrator adjustments)
# - Use None to allow automatic limit computation
# - Brain surface plots should ALWAYS use automatic computation (not listed here)
# - If a limit is None, do NOT call set_ylim() - let matplotlib auto-scale
#
# **Not Included Here (kept hardcoded in their respective scripts):**
# - decoding_eyetracking: (0, 1.0) - script-specific, kept local
# - rdm_intercorr_full: (-1, 1) - full correlation range, kept in module
# - rdm_reliability: (0, 1.05) - reliability metric, kept in module
# - scatter_jitter xlim: (-0.15, 0.15) - plot-specific, kept local
#
# =============================================================================

PLOT_YLIMITS = {
    # -------------------------------------------------------------------------
    # RSA (Representational Similarity Analysis) - Neural Pattern Correlations
    # All RSA plots show Spearman correlation (r) between model and neural RDMs
    # -------------------------------------------------------------------------
    'rsa_neural': (-0.06, 0.25),
    # Used by: chess-mvpa/92_plot_mvpa_rsa.py (ROI-based RSA)
    #          chess-mvpa/93_plot_mvpa_decoding.py (RSA panel in decoding figure)
    # Range: Slightly below zero to capture negative correlations, up to r=0.25

    'rsa_neural_rois_finer': (-0.02, 0.18),
    # Used by: chess-supplementary/mvpa-finer/92_plot_mvpa_finer_panel.py
    # Range: Tighter range for finer-grained ROI analysis

    'rsa_neurosynth_corr': (-0.15, 0.25),
    # Used by: chess-neurosynth/92_plot_neurosynth_rsa.py (term correlation bars)
    #          chess-neurosynth/91_plot_neurosynth_univariate.py (within-group)
    # Range: Meta-analytic term correlations (within Expert or Novice group)
    # NOTE: Surface plots in same scripts use auto-computed limits (not this)

    'rsa_neurosynth_diff': (-0.15, 0.35),
    # Used by: chess-neurosynth/92_plot_neurosynth_rsa.py (difference bars)
    #          chess-neurosynth/91_plot_neurosynth_univariate.py (Expert - Novice)
    # Range: Asymmetric to capture larger positive differences
    # NOTE: Surface plots in same scripts use auto-computed limits (not this)

    # -------------------------------------------------------------------------
    # Decoding (Classification Accuracy - Above Chance Level)
    # Decoding plots show SVM classification accuracy minus chance level
    # -------------------------------------------------------------------------
    'decoding': (-0.06, 0.25),
    # Used by: chess-mvpa/93_plot_mvpa_decoding.py (ROI-based decoding)
    # Range: Chance-subtracted accuracy, same scale as rsa_neural for comparison

    'decoding_rois_finer': (-0.02, 0.18),
    # Used by: chess-supplementary/mvpa-finer/92_plot_mvpa_finer_panel.py
    # Range: Finer-grained ROI decoding, same scale as rsa_neural_rois_finer

    # -------------------------------------------------------------------------
    # Behavioral Measures
    # Behavioral correlations and choice patterns
    # -------------------------------------------------------------------------
    'behavioral_correlation': (-0.2, 1.0),
    # Used by: chess-behavioral/91_plot_behavioral_panels.py (RDM correlations)
    # Range: Partial correlation range, accommodates negative and strong positive

    'behavioral_choice_counts': (0, 450),
    # Used by: chess-behavioral/91_plot_behavioral_panels.py (choice histograms)
    # Range: Choice frequency counts across 40 stimuli

    'behavioral_rsa_models': (-0.25, 0.8),
    # Used by: chess-behavioral/91_plot_behavioral_panels.py (model comparison)
    # Range: RSA model fit correlations

    # -------------------------------------------------------------------------
    # Neurosynth Meta-Analytic Correlations (Univariate Activation)
    # Correlations between univariate activation and meta-analytic term maps
    # -------------------------------------------------------------------------
    'neurosynth_univariate_corr': (-0.15, 0.20),
    # Used by: chess-neurosynth/91_plot_neurosynth_univariate.py (term correlations)
    # Range: Within-group correlations with meta-analytic maps

    'neurosynth_univariate_diff': (-0.15, 0.35),
    # Used by: chess-neurosynth/91_plot_neurosynth_univariate.py (group differences)
    # Range: Expert - Novice differences, asymmetric for positive effects
}

# =============================================================================
# Apply Nature RC Parameters
# =============================================================================

def apply_nature_rc(params: dict = None) -> None:
    """
    Apply Nature-compliant rcParams to matplotlib/seaborn.

    Sets:
    - Font family: Arial/Helvetica/DejaVu Sans
    - Font sizes: 5-7pt (from params)
    - Transparency: disabled (savefig.transparent=False)
    - Font embedding: TrueType 42 (PDF/PS), editable SVG
    - Mathtext: DejaVu Sans, regular weight (for Greek letters)
    - Constrained layout by default

    Parameters
    ----------
    params : dict, optional
        PLOT_PARAMS override. If None, uses global PLOT_PARAMS.

    Examples
    --------
    >>> from common.plotting import apply_nature_rc
    >>> apply_nature_rc()
    >>> fig, ax = plt.subplots()  # Will use Nature settings
    """
    if params is None:
        params = PLOT_PARAMS

    # Font family
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # Font sizes
    matplotlib.rcParams['font.size'] = params['font_size_label']
    matplotlib.rcParams['axes.titlesize'] = params['font_size_title']
    matplotlib.rcParams['axes.labelsize'] = params['font_size_label']
    matplotlib.rcParams['xtick.labelsize'] = params['font_size_tick']
    matplotlib.rcParams['ytick.labelsize'] = params['font_size_tick']
    matplotlib.rcParams['legend.fontsize'] = params['font_size_legend']
    matplotlib.rcParams['figure.titlesize'] = params['font_size_title']

    # Line widths
    matplotlib.rcParams['axes.linewidth'] = params['axes_linewidth']
    matplotlib.rcParams['xtick.major.width'] = params['tick_major_width']
    matplotlib.rcParams['ytick.major.width'] = params['tick_major_width']
    matplotlib.rcParams['xtick.minor.width'] = params['tick_minor_width']
    matplotlib.rcParams['ytick.minor.width'] = params['tick_minor_width']
    matplotlib.rcParams['lines.linewidth'] = params['plot_linewidth']
    matplotlib.rcParams['hatch.linewidth'] = params['base_linewidth']  # Same as base

    # Tick sizes
    matplotlib.rcParams['xtick.major.size'] = params['tick_major_size']
    matplotlib.rcParams['ytick.major.size'] = params['tick_major_size']
    matplotlib.rcParams['xtick.minor.size'] = params['tick_minor_size']
    matplotlib.rcParams['ytick.minor.size'] = params['tick_minor_size']

    # Tick direction
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    # Grid (disabled for Nature)
    matplotlib.rcParams['axes.grid'] = False

    # Transparency (disabled for production)
    matplotlib.rcParams['savefig.transparent'] = False
    matplotlib.rcParams['figure.facecolor'] = params['facecolor']
    matplotlib.rcParams['axes.facecolor'] = 'white'

    # Font embedding (editable text)
    matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType
    matplotlib.rcParams['ps.fonttype'] = 42   # TrueType
    matplotlib.rcParams['svg.fonttype'] = 'none'  # Editable SVG text

    # Mathtext (Greek letters)
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
    matplotlib.rcParams['mathtext.default'] = 'regular'

    # Constrained layout (not tight_layout)
    matplotlib.rcParams['figure.constrained_layout.use'] = True
    matplotlib.rcParams['figure.constrained_layout.h_pad'] = 0.05  # inches
    matplotlib.rcParams['figure.constrained_layout.w_pad'] = 0.05  # inches

    # DPI and format
    matplotlib.rcParams['savefig.dpi'] = params['dpi']
    matplotlib.rcParams['savefig.format'] = params['format']

    # Seaborn style (minimal, compatible with Nature)
    sns.set_style('ticks', {
        'axes.grid': False,
        'axes.linewidth': params['axes_linewidth'],
    })


# =============================================================================
# Figure Sizing
# =============================================================================

def figure_size(
    columns: Literal[1, 2],
    height_mm: Optional[float] = None,
    aspect: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute Nature-compliant figure size.

    Parameters
    ----------
    columns : 1 or 2
        Figure width: 1=89mm (single column), 2=183mm (double column)
    height_mm : float, optional
        Figure height in millimeters. If provided, aspect is ignored.
        Max 170mm (Nature limit).
    aspect : float, optional
        Width/height ratio (e.g., 1.5 for 3:2 aspect).
        Only used if height_mm is None.

    Returns
    -------
    figsize : (width_inches, height_inches)

    Examples
    --------
    >>> # Double-column figure, 120mm tall
    >>> figsize = figure_size(columns=2, height_mm=120)
    >>> fig, ax = plt.subplots(figsize=figsize)

    >>> # Single-column figure, 4:3 aspect ratio
    >>> figsize = figure_size(columns=1, aspect=4/3)
    """
    # Width
    if columns == 1:
        width_mm = _NATURE_SINGLE_COL_MM
    elif columns == 2:
        width_mm = _NATURE_DOUBLE_COL_MM
    else:
        raise ValueError(f"columns must be 1 or 2, got {columns}")

    # Height
    if height_mm is not None:
        if height_mm > _NATURE_MAX_HEIGHT_MM:
            import warnings
            warnings.warn(
                f"height_mm={height_mm:.1f} exceeds Nature max ({_NATURE_MAX_HEIGHT_MM}mm). "
                f"Capping at {_NATURE_MAX_HEIGHT_MM}mm."
            )
            height_mm = _NATURE_MAX_HEIGHT_MM
    elif aspect is not None:
        height_mm = width_mm / aspect
        if height_mm > _NATURE_MAX_HEIGHT_MM:
            height_mm = _NATURE_MAX_HEIGHT_MM
    else:
        # Default: Golden ratio
        height_mm = width_mm / 1.618

    # Convert to inches
    width_in = width_mm * _MM
    height_in = height_mm * _MM

    return (width_in, height_in)


def auto_bar_figure_size(
    n_categories: int,
    is_grouped: bool = True,
    target_bar_width_mm: float = None,
    max_width_mm: float = None,
    height_mm: Optional[float] = None
) -> Tuple[Tuple[float, float], float]:
    """
    Compute figure size for bar plots with fixed physical bar width.

    The bar width in millimeters is constant across plots. Figure width
    adjusts to accommodate the number of categories.

    Parameters
    ----------
    n_categories : int
        Number of category groups on x-axis
    is_grouped : bool, default=True
        If True, two bars per category (4mm each, pair=8mm).
        If False, one bar per category (8mm each).
    target_bar_width_mm : float, optional
        Target bar width in mm. Defaults to PLOT_PARAMS['target_bar_width_mm'].
        For grouped: each bar = target/2, for single: each bar = target.
    max_width_mm : float, optional
        Maximum figure width. Defaults to double-column (183mm).
        If computed width exceeds this, bars are scaled down proportionally.
    height_mm : float, optional
        Figure height. If None, uses default (100mm).

    Returns
    -------
    figsize : (width_inches, height_inches)
        Figure size in inches
    bar_width_data : float
        Bar width in data coordinates (for use with ax.bar(..., width=bar_width_data))

    Examples
    --------
    >>> # 22 ROIs, grouped bars (experts vs novices)
    >>> figsize, bar_width = auto_bar_figure_size(n_categories=22, is_grouped=True)
    >>> fig, ax = plt.subplots(figsize=figsize)
    >>> ax.bar(x - bar_width/2, expert_vals, width=bar_width, ...)
    >>> ax.bar(x + bar_width/2, novice_vals, width=bar_width, ...)

    Notes
    -----
    - Bar width is FIXED in physical units (mm), not data units
    - Figure width adjusts to maintain constant bar width
    - For grouped plots: each bar is 4mm (pair = 8mm total)
    - For single plots: each bar is 8mm
    - Category spacing: 2mm between categories
    """
    if target_bar_width_mm is None:
        target_bar_width_mm = _TARGET_BAR_WIDTH_MM
    if max_width_mm is None:
        max_width_mm = _NATURE_DOUBLE_COL_MM

    # Determine bar width per bar
    if is_grouped:
        bar_mm = target_bar_width_mm / 2.0  # 4mm per bar
    else:
        bar_mm = target_bar_width_mm  # 8mm per bar

    # Spacing between categories
    category_spacing_mm = 2.0

    # Category footprint
    # Grouped: 4mm + 4mm + 2mm spacing = 10mm total per category
    # Single: 8mm + 2mm spacing = 10mm total per category
    if is_grouped:
        footprint_mm = bar_mm * 2 + category_spacing_mm
    else:
        footprint_mm = bar_mm + category_spacing_mm

    # Margins
    left_margin_mm = 20.0
    right_margin_mm = 10.0

    # Compute figure width
    figure_width_mm = left_margin_mm + (n_categories * footprint_mm) + right_margin_mm

    # Scale down if exceeds max
    scale_factor = 1.0
    if figure_width_mm > max_width_mm:
        scale_factor = max_width_mm / figure_width_mm
        figure_width_mm = max_width_mm
        bar_mm = bar_mm * scale_factor

    # Convert to inches
    figure_width_in = figure_width_mm * _MM

    # Height
    if height_mm is None:
        height_mm = 100.0  # Default
    height_in = height_mm * _MM

    # Compute bar_width in data coordinates
    # Assume categories at integer positions: 0, 1, 2, ..., n_categories-1
    # Assume axes occupy ~70% of figure width (accounting for margins/labels)
    axes_fraction = 0.70
    axes_width_mm = figure_width_mm * axes_fraction

    # Data range for n_categories at positions 0 to n-1 is n_categories
    # (with x-axis from -0.5 to n_categories-0.5, but data range is still n_categories)
    data_range = n_categories

    # 1 data unit corresponds to (axes_width_mm / data_range) mm
    # To get bar_width_data: bar_mm / (axes_width_mm / data_range)
    bar_width_data = (bar_mm * data_range) / axes_width_mm

    return ((figure_width_in, height_in), bar_width_data)
