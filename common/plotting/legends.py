"""
Legend utilities for Nature-style figures.

Currently provides ROI group legends that mirror the supplementary material
style, with optional colorblind palette handling and 1-row/2-row layouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..constants import CONFIG
from .style import PLOT_PARAMS, apply_nature_rc, figure_size
from .helpers import save_figure


def create_roi_group_legend(
    roi_metadata_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    single_row: bool = False,
    colorblind: bool = False,
    params: dict | None = None,
) -> plt.Figure:
    """
    Create a standalone legend figure showing ROI group colors and names.

    Parameters
    ----------
    roi_metadata_path : Path, optional
        Path to region_info.tsv file. Defaults to CONFIG['ROI_GLASSER_22']/region_info.tsv.
    output_path : Path, optional
        If provided, the legend figure is saved to this path.
    single_row : bool, default=False
        If True, display all groups on a single row.
        If False, display two centered rows with the first four groups on top.
    colorblind : bool, default=False
        If True, use the 'color_cb' column for colors.
        Otherwise, use the default 'color' column.
    params : dict, optional
        Plotting parameters (defaults to PLOT_PARAMS).
    """
    if params is None:
        params = PLOT_PARAMS
    apply_nature_rc(params)

    import pandas as pd

    if roi_metadata_path is None:
        roi_metadata_path = CONFIG['ROI_GLASSER_22'] / 'region_info.tsv'

    roi_info = pd.read_csv(roi_metadata_path, sep='\t')

    if 'group' in roi_info.columns:
        group_col = 'group'
    elif 'family' in roi_info.columns:
        group_col = 'family'
    else:
        raise ValueError("ROI metadata must have 'group' or 'family' column")

    color_col = 'color_cb' if colorblind else 'color'
    if color_col not in roi_info.columns:
        raise ValueError(f"ROI metadata must have '{color_col}' column")

    unique_groups = roi_info[[group_col, color_col]].drop_duplicates(subset=group_col)

    if 'order' in roi_info.columns:
        order_info = (
            roi_info[[group_col, 'order']]
            .drop_duplicates(subset=group_col)
            .sort_values('order')
        )
        unique_groups = unique_groups.merge(order_info, on=group_col, how='left').sort_values('order')
    else:
        unique_groups = unique_groups.sort_values(group_col)

    handles: list[Patch] = [
        Patch(
            facecolor=row[color_col],
            edgecolor='black',
            linewidth=params['plot_linewidth'],
            label=row[group_col],
        )
        for _, row in unique_groups.iterrows()
    ]

    n_groups = len(handles)
    if single_row:
        ncol = max(n_groups, 1)
        fig_h_mm = 20
        first_row_items = n_groups
    else:
        first_row_items = min(4, n_groups)
        ncol = max(first_row_items, 1)
        fig_h_mm = 35

    fig_w, fig_h = figure_size(columns=2, height_mm=fig_h_mm)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    legend_handles: Sequence[Patch] = handles
    legend_labels = [h.get_label() for h in handles]

    if not single_row and n_groups > first_row_items:
        top_handles = handles[:first_row_items]
        top_labels = legend_labels[:first_row_items]
        bottom_handles = handles[first_row_items:]
        bottom_labels = legend_labels[first_row_items:]

        reordered_handles: list[Patch] = []
        reordered_labels: list[str] = []
        for idx in range(ncol):
            if idx < len(top_handles):
                reordered_handles.append(top_handles[idx])
                reordered_labels.append(top_labels[idx])
        for idx in range(ncol):
            if idx < len(bottom_handles):
                reordered_handles.append(bottom_handles[idx])
                reordered_labels.append(bottom_labels[idx])

        legend_handles = reordered_handles
        legend_labels = reordered_labels

    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        ncol=ncol,
        frameon=True,
        fontsize=params['font_size_legend'],
        title='ROI Groups',
        title_fontsize=params['font_size_legend'],
        edgecolor='black',
        fancybox=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(params['plot_linewidth'])

    if not single_row and n_groups > first_row_items:
        try:
            legend._legend_box.align = "center"  # type: ignore[attr-defined]
            for child in legend._legend_box.get_children():  # type: ignore[attr-defined]
                if hasattr(child, "align"):
                    child.align = "center"
        except AttributeError:
            pass

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, Path(output_path))
    return fig
