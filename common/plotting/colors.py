#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color palettes and colormaps for Nature-compliant figures.

Provides:
- CMAP_BRAIN: Diverging colormap for RDMs/brain maps (cyan-purple, center=0)
- COLORS_EXPERT_NOVICE: Blue (#0072B2) + Vermillion (#D55E00) - colorblind safe
- COLORS_CHECKMATE_NONCHECKMATE: Brown + Blue for stimulus categories
- COLORS_WONG: Reference palette for accessibility
- compute_stimulus_palette(): Compute colors/alphas for stimuli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from typing import List, Tuple


# =============================================================================
# Brain Colormap (UNCHANGED - DO NOT MODIFY)
# =============================================================================

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
    >>> from common.plotting import CMAP_BRAIN
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


# =============================================================================
# Color Palettes
# =============================================================================

# Expert vs Novice (colorblind safe)
COLORS_EXPERT_NOVICE = {
    'expert': '#228833',   # Green
    'novice': '#aa3377',   # Purple
}

# Checkmate vs Non-Checkmate (colorblind safe, distinct from expert/novice)
# Used for stimulus visualizations in behavioral RDMs, MDS, etc.
COLORS_CHECKMATE_NONCHECKMATE = {
    'checkmate': '#fdb338',      # Orange
    'non_checkmate': '#025196',  # Blue
}

# Wong Colorblind-Safe Palette (Reference)
# Source: Wong, B. (2011). Points of view: Color blindness. Nature Methods 8, 441.
# Use these colors for any new visualizations to ensure accessibility
COLORS_WONG = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000',
}


# =============================================================================
# Stimulus Palette Computation
# =============================================================================

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
    - Colors come from COLORS_CHECKMATE_NONCHECKMATE
    - Alpha mapping is uniform across the number of unique strategies in each
      check group, using increasing steps from 1/n .. 1.0
    - The function does not reorder; it preserves the row order of stimuli_df

    Examples
    --------
    >>> from common.plotting import compute_stimulus_palette
    >>> colors, alphas = compute_stimulus_palette(stimuli_df)
    >>> plt.scatter(x, y, c=colors, alpha=alphas)
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


# =============================================================================
# Color utilities
# =============================================================================

def lighten_color(color: str | tuple, amount: float = 0.5) -> tuple:
    """
    Blend the input color with white to obtain a lighter tone.

    Parameters
    ----------
    color : str or tuple
        Matplotlib-friendly color specification (hex, name, or RGB tuple)
    amount : float, default=0.5
        Blend factor in [0, 1]. 0 → original color, 1 → pure white.

    Returns
    -------
    tuple
        Lightened RGB tuple (each channel in [0, 1])
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    base = np.array(to_rgb(color))
    white = np.ones(3)
    blended = base + (white - base) * amount
    return tuple(blended)


# =============================================================================
# Centralized Sequential Colormap for Matrices
# =============================================================================

import seaborn as sns  # explicit dependency; ImportError should surface

# Public sequential colormap (mako), centralized (no wrappers)
CMAP_SEQUENTIAL = sns.color_palette("mako", as_cmap=True)
