"""
Plotting helpers for the RDM intercorrelation supplementary analysis.

These utilities use common.plotting bar functions following DRY principles.
Only analysis-specific logic (labels, data prep) is kept here.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from common import CONFIG
from common.plotting import plot_grouped_bars_on_ax, lighten_color, PLOT_PARAMS


# Short labels for RDM intercorrelation figures
_SHORT_LABELS = {
    'visual': 'Visual',
    'strategy': 'Strategy',
    'check': 'Checkmate',
}


def plot_correlation_bars(
    ax: Axes,
    target: str,
    predictors: list[str],
    pairwise_df: pd.DataFrame,
    partial_lookup: Dict[Tuple[str, str], float],
    model_colors: Dict[str, tuple],
    ylabel: bool = False,
) -> None:
    """
    Draw pairwise vs partial correlation bars using common bar plotting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    target : str
        Target model key.
    predictors : list of str
        Predictor model keys (order determines bar order).
    pairwise_df : pd.DataFrame
        Pairwise correlation matrix.
    partial_lookup : dict[(str, str) -> float]
        Lookup for partial correlations keyed by (target, predictor).
    model_colors : dict[str -> tuple]
        RGB colors for each model (from seaborn colorblind palette).
    ylabel : bool, default=False
        If True, annotate y-axis with correlation label.
    """
    # Extract values
    pairwise_vals = [pairwise_df.loc[target, pred] for pred in predictors]
    partial_vals = [partial_lookup.get((target, pred), np.nan) for pred in predictors]

    # Colors: pairwise=full, partial=lightened
    pairwise_colors = [model_colors[p] for p in predictors]
    partial_colors = [lighten_color(model_colors[p], 0.45) for p in predictors]

    # X positions and labels
    x_pos = np.arange(len(predictors))
    xlabels = [_SHORT_LABELS.get(pred, pred.capitalize()) for pred in predictors]

    # Use common grouped bar plotting function
    target_label = _SHORT_LABELS.get(target, target.capitalize())
    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x_pos,
        group1_values=pairwise_vals,
        group1_color=pairwise_colors,
        group2_values=partial_vals,
        group2_color=partial_colors,
        group1_label='Pairwise',
        group2_label='Partial',
        ylim=(-1, 1),
        show_errorbars=False,
        add_value_labels=True,
        value_label_format='.2f',
        bar_width_multiplier=.5,  # Wider bars for single-group
        # DRY formatting in helper
        y_label=('Correlation' if ylabel else None),
        subtitle=target_label,
        xtick_labels=xlabels,
        x_tick_rotation=0,
        x_tick_align='center',
        visible_spines=['left','bottom'],
        show_legend=True,
        legend_loc='upper right',
        params=PLOT_PARAMS,
    )


def plot_variance_partition_bars(
    ax: Axes,
    target: str,
    var_part_df: pd.DataFrame,
    model_colors: Dict[str, tuple],
    color_shared: tuple,
    color_unexplained: tuple,
    ylabel: bool = False,
) -> None:
    """
    Draw variance partitioning bars using common bar plotting.

    Shows separate bars for each component EXCEPT the target itself.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    target : str
        Target model key.
    var_part_df : pd.DataFrame
        Variance partitioning results with columns: target, unique_*, shared, residual.
    model_colors : dict[str -> tuple]
        RGB colors for each model.
    color_shared : tuple
        RGB color for shared variance.
    color_unexplained : tuple
        RGB color for unexplained variance.
    ylabel : bool, default=False
        If True, annotate y-axis.
    """
    subset = var_part_df[var_part_df['target'] == target]
    if subset.empty:
        raise ValueError(f"No variance partitioning results for target '{target}'")

    row = subset.iloc[0]

    # Extract predictors (columns like 'unique_visual', 'unique_strategy', etc.)
    # IMPORTANT: Exclude the target itself from predictors
    predictor_cols = [col for col in row.index if col.startswith('unique_')]
    available_predictors = [col.replace('unique_', '') for col in predictor_cols]
    available_predictors = [p for p in available_predictors if p != target]

    # Order predictors by MODEL_ORDER
    ordered_predictors = [
        p for p in CONFIG['MODEL_ORDER'] if p in available_predictors
    ]
    ordered_predictors += [p for p in available_predictors if p not in ordered_predictors]

    # Build bar data: unique for each predictor, then shared, then unexplained
    bar_labels = []
    bar_values = []
    bar_colors = []

    for predictor in ordered_predictors:
        value = float(row[f'unique_{predictor}'])
        label = _SHORT_LABELS.get(predictor, predictor.capitalize())
        bar_labels.append(label)
        bar_values.append(value)
        bar_colors.append(model_colors.get(predictor, '#808080'))

    # Shared and unexplained
    bar_labels.extend(['Shared', 'Unexplained'])
    bar_values.extend([float(row['shared']), float(row['residual'])])
    bar_colors.extend([color_shared, color_unexplained])

    # X positions
    x_pos = np.arange(len(bar_labels))

    # Use common single-group bar plotting
    target_label = _SHORT_LABELS.get(target, target.capitalize())
    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x_pos,
        group1_values=bar_values,
        group1_color=bar_colors,
        ylim=(0, 1.05),
        bar_width_multiplier=1.0,  # Wider bars for single-group
        show_errorbars=False,
        add_value_labels=True,
        value_label_format='.2f',
        # DRY formatting in helper
        y_label=('Variance explained (R²)' if ylabel else None),
        subtitle=f'{target_label}',
        xtick_labels=bar_labels,
        x_tick_rotation=30,
        x_tick_align='right',
        visible_spines=['left','bottom'],
        params=PLOT_PARAMS,
    )


__all__ = [
    'plot_correlation_bars',
    'plot_variance_partition_bars',
]
