"""
Plotting utilities for behavioral RSA analysis.

This module provides specialized plotting functions for visualizing MDS
embeddings, choice frequencies, and model correlations. RDM heatmaps are
centralized via common.plotting_utils.plot_rdm to adhere to DRY principles.

All plotting functions accept stimulus metadata as parameters instead of
hardcoded values, ensuring flexibility and adaptability to different datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional

from common import CONFIG
from common.plotting import (
    apply_nature_rc, style_spines, hide_ticks, set_axis_title,
    plot_grouped_bars_with_ci, PLOT_PARAMS,
    COLORS_EXPERT_NOVICE,
    compute_stimulus_palette,
)
from common import MODEL_ORDER, MODEL_LABELS_PRETTY


## RDM heatmaps are provided by common.plotting.plot_rdm


## 2D embedding plotting is centralized in common.plotting.plot_2d_embedding


def plot_choice_frequency(
    pairwise_df: pd.DataFrame,
    expertise_label: str,
    output_path: Path,
    stimuli_df: pd.DataFrame
) -> None:
    """
    Plot stimulus selection frequency as bar chart.

    Shows how often each stimulus was chosen as "better" in pairwise comparisons.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data with 'better' column.
        Can be either raw data (one row per comparison) or aggregated data (with 'count' column).
    expertise_label : str
        Expertise group label
    output_path : Path
        Path to save figure
    stimuli_df : pd.DataFrame
        Stimulus metadata for color/alpha computation

    Returns
    -------
    None
        Saves figure to output_path

    Notes
    -----
    - Handles both raw and aggregated pairwise data
    - If aggregated (has 'count' column), uses weighted counts
    - Otherwise, uses simple value_counts()

    Example
    -------
    >>> pairwise_df = create_pairwise_df(trial_df)
    >>> plot_choice_frequency(pairwise_df, "Experts", output_dir / "freq.pdf", stimuli_df)
    """
    # Use centralized styling
    apply_nature_rc()

    # Compute frequency of each stimulus being chosen
    # Handle both raw and aggregated data
    if 'count' in pairwise_df.columns:
        # Aggregated data - weight by count column
        frequency = pairwise_df.groupby('better')['count'].sum().sort_index()
    else:
        # Raw data - simple value counts
        frequency = pairwise_df['better'].value_counts().sort_index()

    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS['figure_sizes']['large'],
        facecolor=PLOT_PARAMS['facecolor']
    )

    # Compute stimulus colors and alphas
    strat_colors, strat_alphas = compute_stimulus_palette(stimuli_df)

    # Create bar plot with strategy colors
    bar_plot = sns.barplot(
        x=frequency.index,
        y=frequency.values,
        palette=strat_colors[:len(frequency)],
        ax=ax
    )

    # Set alpha for each bar
    for bar, alpha in zip(bar_plot.patches, strat_alphas[:len(frequency)]):
        bar.set_alpha(alpha)

    # Set title with subtitle using centralized function (title=bold, subtitle=normal)
    set_axis_title(ax, "Stimulus Selection Frequency", subtitle=expertise_label)

    # Set labels using centralized parameters
    ax.set_xlabel('Stimulus ID', fontsize=PLOT_PARAMS['font_size_label'])
    ax.set_ylabel('Selection Count', fontsize=PLOT_PARAMS['font_size_label'])
    ax.set_xticks([])

    # Apply consistent styling (standard bar plot - left and bottom spines only)
    plt.tight_layout()
    sns.despine(trim=False)
    style_spines(ax, visible_spines=['left', 'bottom'])

    # Save and close
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
               facecolor=PLOT_PARAMS['facecolor'], dpi=PLOT_PARAMS['dpi'])
    plt.close()


def plot_model_correlations(
    expert_results: List[Tuple[str, float, float, float, float]],
    novice_results: List[Tuple[str, float, float, float, float]],
    output_path: Path
) -> None:
    """
    Plot side-by-side bar chart of model-behavior correlations for experts vs novices.

    Creates a grouped bar plot showing correlation coefficients with 95% CIs
    for each model dimension, comparing experts and novices.

    This function is a thin wrapper around the centralized plot_grouped_bars_with_ci()
    utility from common.plot_utils. It handles data extraction and formatting specific
    to behavioral RSA correlation results.

    Parameters
    ----------
    expert_results : list of tuple
        Expert correlation results, each tuple is (column, r, p, ci_lower, ci_upper)
    novice_results : list of tuple
        Novice correlation results, same format as expert_results
    output_path : Path
        Path to save figure

    Returns
    -------
    None
        Saves figure to output_path

    Notes
    -----
    - Reorders columns to: Visual Similarity, Strategy, Checkmate
    - Adds significance stars (* p<0.05, ** p<0.01, *** p<0.001)
    - Colors from COLORS_EXPERT_NOVICE in common.constants
    - Includes dashed horizontal line at y=0
    - Uses centralized grouped bar plot function (DRY principle)

    Example
    -------
    >>> expert_res, _ = correlate_with_all_models(expert_rdm, category_df)
    >>> novice_res, _ = correlate_with_all_models(novice_rdm, category_df)
    >>> plot_model_correlations(expert_res, novice_res, output_dir / "correlations.pdf")
    """
    # Get colors from centralized constants
    COL_EXPERT = COLORS_EXPERT_NOVICE['expert']
    COL_NOVICE = COLORS_EXPERT_NOVICE['novice']

    # === Step 1: Extract data from results tuples ===
    # Each result is: (column_name, r_value, p_value, ci_lower, ci_upper)
    r_exp = [res[1] for res in expert_results]
    ci_exp = [(res[3], res[4]) for res in expert_results]
    p_exp = [res[2] for res in expert_results]

    r_nov = [res[1] for res in novice_results]
    ci_nov = [(res[3], res[4]) for res in novice_results]
    p_nov = [res[2] for res in novice_results]

    # === Step 2: Reorder to desired display order using centralized constants ===
    # MODEL_ORDER and MODEL_LABELS_PRETTY are imported from common.plotting_utils
    column_labels = [res[0] for res in expert_results]

    # Find indices for reordering
    idx_order = [column_labels.index(lbl) for lbl in MODEL_ORDER]

    # Reorder all data
    r_exp_ordered = [r_exp[i] for i in idx_order]
    ci_exp_ordered = [ci_exp[i] for i in idx_order]
    p_exp_ordered = [p_exp[i] for i in idx_order]

    r_nov_ordered = [r_nov[i] for i in idx_order]
    ci_nov_ordered = [ci_nov[i] for i in idx_order]
    p_nov_ordered = [p_nov[i] for i in idx_order]

    # === Step 3: Use centralized grouped bar plot function ===
    # This follows DRY principle - all grouped bar plotting logic is in one place
    # Uses MODEL_LABELS_PRETTY from centralized constants
    plot_grouped_bars_with_ci(
        group1_values=r_exp_ordered,
        group2_values=r_nov_ordered,
        group1_cis=ci_exp_ordered,
        group2_cis=ci_nov_ordered,
        x_labels=MODEL_LABELS_PRETTY,  # Centralized labels
        group1_pvals=p_exp_ordered,
        group2_pvals=p_nov_ordered,
        group1_label="Experts",
        group2_label="Novices",
        group1_color=COL_EXPERT,
        group2_color=COL_NOVICE,
        ylabel=r"Pearson $\it{r}$ (95% CI via bootstrapping)",
        title="Behavioral-Model RDMs Correlations",
        subtitle="Experts vs. Novices",
        output_path=output_path,
        figsize=PLOT_PARAMS['figure_sizes']['large'],
        ylim=(-0.2, 1.0),
        add_zero_line=True,
        legend_loc="upper left"
    )
