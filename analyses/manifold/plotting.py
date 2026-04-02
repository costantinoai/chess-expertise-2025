"""
Manifold-specific plotting helpers (module-level, DRY and readable).

Provides small, well-documented helpers to keep panel scripts tidy while
delegating heavy plotting primitives to `common.plotting`.

Functions
---------
compute_limits_with_padding : Derive x/y limits from 2D coords with padding
pca_axis_labels_from_explained : Build axis labels from explained variance
plot_topn_feature_importance_on_ax : Horizontal bar plot of top-N classifier weights
"""

from __future__ import annotations

from typing import Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common import PLOT_PARAMS, COLORS_EXPERT_NOVICE
from common.plotting import set_axis_title


def compute_limits_with_padding(coords: np.ndarray, padding: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute axis limits for 2D coordinates with symmetric padding.

    Parameters
    ----------
    coords : np.ndarray of shape (n_samples, 2)
        2D coordinates (x, y) to bound.
    padding : float, default=0.05
        Fractional padding relative to data range added on each side.

    Returns
    -------
    xlim, ylim : (xmin, xmax), (ymin, ymax)
        Axis limits with padding applied.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (n_samples, 2)")

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    xlim = (x_min - padding * x_range, x_max + padding * x_range)
    ylim = (y_min - padding * y_range, y_max + padding * y_range)
    return xlim, ylim


def pca_axis_labels_from_explained(explained: Sequence[float]) -> Tuple[str, str]:
    """
    Build PC axis labels from explained variance percentages.

    Parameters
    ----------
    explained : sequence of float
        Explained variance percentages for PCs. Must contain at least two values.

    Returns
    -------
    x_label, y_label : str, str
        Labels formatted as "PC 1 (X% var)" and "PC 2 (Y% var)".
    """
    if len(explained) < 2:
        raise ValueError("explained must contain at least two elements for PC1 and PC2")
    return (f"PC 1 ({explained[0]:.1f}% var)", f"PC 2 ({explained[1]:.1f}% var)")


def plot_topn_feature_importance_on_ax(
    ax: plt.Axes,
    weights: np.ndarray,
    roi_info: pd.DataFrame,
    *,
    top_n: int = 10,
    params: dict | None = None,
    title: str = 'Top 10 Contributions to Classification',
) -> None:
    """
    Plot top-N absolute classifier weights as a horizontal bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    weights : np.ndarray of shape (n_rois,)
        Classifier weights in original ROI space.
    roi_info : pd.DataFrame
        ROI metadata with columns 'roi_id' and 'pretty_name'. ROI IDs are
        expected to map to weights via 1-based indexing (id = index + 1).
    top_n : int, default=10
        Number of top absolute weights to display.
    params : dict, optional
        PLOT_PARAMS override.
    title : str, default 'Top 10 Contributions to Classification'
        Axis title to display.
    """
    if params is None:
        params = PLOT_PARAMS

    abs_weights = np.abs(weights)
    top_idx = np.argsort(abs_weights)[-top_n:]
    top_weights = weights[top_idx]

    # Map ROI index (0-based) to pretty name using 1-based ROI IDs
    id_to_pretty = dict(zip(roi_info['roi_id'], roi_info['pretty_name']))
    top_names = [id_to_pretty.get(i + 1, f'ROI_{i+1}') for i in top_idx]
    top_names = [str(name).replace('\n', ' ') for name in top_names]

    # Colors by sign
    top_colors = [COLORS_EXPERT_NOVICE['novice'] if w < 0 else COLORS_EXPERT_NOVICE['expert'] for w in top_weights]

    y_pos = np.arange(len(top_names))
    ax.barh(
        y_pos,
        top_weights,
        color=top_colors,
        alpha=params.get('bar_alpha', 0.7),
        edgecolor='black',
        linewidth=params['plot_linewidth']
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=params['font_size_label'], color="gray")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Weight in Original ROI Space', fontsize=params['font_size_label'])
    ax.tick_params(axis='x', labelsize=params['font_size_tick'])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1, float(top_n))
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.axvline(x=0, color='black', linestyle='-', linewidth=params['plot_linewidth'], alpha=params.get('line_alpha', 0.5))

    set_axis_title(ax, title=title, subtitle='')

    # Directional annotation
    from matplotlib.font_manager import FontProperties
    arrow_font = FontProperties(family=params["font_family"], size=params['font_size_title'])
    ax.text(0.15, 1.04, "← Higher PR predictive of Novices", ha="center", va="center",
            transform=ax.transAxes, fontproperties=arrow_font,
            color=COLORS_EXPERT_NOVICE['novice'])
    ax.text(0.85, 1.04, "Higher PR predictive of Experts →", ha="center", va="center",
            transform=ax.transAxes, fontproperties=arrow_font,
            color=COLORS_EXPERT_NOVICE['expert'])


__all__ = [
    'compute_limits_with_padding',
    'pca_axis_labels_from_explained',
    'plot_topn_feature_importance_on_ax',
]

