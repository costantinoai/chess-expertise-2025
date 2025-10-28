"""
Manifold analysis visualization functions.

This module provides plotting functions specific to manifold dimensionality analysis
(participation ratio studies). All functions use centralized plotting utilities
from common.plotting_utils to ensure consistency across analyses.

The module follows DRY principles by:
- Inheriting style from common plotting functions
- Using centralized bar plotting and styling
- Employing standardized title/subtitle formatting
- Utilizing shared color schemes from common.constants

Functions
---------
plot_pr_roi_bars : Two-panel bar plots (group means + differences)
plot_pr_voxel_correlations : Scatter plots of PR vs voxel count
plot_pr_feature_importance : ROI contributions to classification
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Centralized plotting params and helpers
from common.plotting_utils import (
    figure_style,
    style_spines,
    set_axis_title,
    plot_grouped_bars_with_ci,  # Unified function (handles both single and per-item colors)
    plot_grouped_bars_on_ax,    # For multi-panel layouts (now self-contained)
    add_roi_color_legend,
    PLOT_PARAMS,
)
from common import COLORS_EXPERT_NOVICE

logger = logging.getLogger(__name__)


def plot_pr_roi_bars(
    summary_stats: pd.DataFrame,
    stats_results: pd.DataFrame,
    roi_info: pd.DataFrame,
    output_dir: Path,
    alpha: float = 0.05,
    use_fdr: bool = True,
    figsize: Tuple[float, float] = None,
    show_legend: bool = False,
) -> Path:
    """
    Create combined two-panel participation ratio figure.

    Top panel: Expert vs Novice means with 95% CIs (grouped bars)
    Bottom panel: Expert-Novice differences with 95% CIs (single bars)

    The top panel shares x-axis with bottom panel and has no x-tick labels.

    Parameters
    ----------
    show_legend : bool, default=False
        Whether to show the ROI color legend at the bottom

    Returns
    -------
    Path
        Path to saved combined figure (pr_roi_combined_panel.pdf)
    """
    figure_style(PLOT_PARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)

    if figsize is None:
        figsize = PLOT_PARAMS['figure_sizes']['full_width']

    # Merge to get pretty names and ROI colors
    summary_with_info = summary_stats.merge(
        roi_info[['ROI_idx', 'pretty_name', 'color']],
        left_on='ROI_Label', right_on='ROI_idx', how='left'
    )
    stats_with_info = stats_results.merge(
        roi_info[['ROI_idx', 'pretty_name', 'color']],
        left_on='ROI_Label', right_on='ROI_idx', how='left'
    )

    # Sort by ROI_Label for consistent ordering
    expert_data = summary_with_info[summary_with_info['group'] == 'expert'].sort_values('ROI_Label')
    novice_data = summary_with_info[summary_with_info['group'] == 'novice'].sort_values('ROI_Label')

    roi_names = expert_data['pretty_name'].tolist()
    roi_names = [roi_name.replace("\\n", " ") for roi_name in roi_names]
        
    roi_colors = expert_data['color'].tolist()

    # Determine significance
    sig_col = 'p_val_fdr' if use_fdr else 'p_val'
    pvals = stats_with_info[sig_col].tolist()
    is_significant = [p < alpha for p in pvals]

    # =============== Create two-panel figure ===============
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(figsize[0], figsize[1]),  # Taller to accommodate both panels
        sharex=True,  # Share x-axis
        gridspec_kw={'height_ratios': [1, 1.5]},  # Equal height
        dpi=PLOT_PARAMS['dpi'],
        facecolor=PLOT_PARAMS['facecolor']
    )

    x = np.arange(len(roi_names))

    # =============== TOP PANEL: Group means (experts vs novices) ===============
    # Extract data for both groups
    exp_vals = expert_data['mean_PR'].tolist()
    exp_cis = list(zip(expert_data['ci_low'].values, expert_data['ci_high'].values))

    nov_vals = novice_data['mean_PR'].tolist()
    nov_cis = list(zip(novice_data['ci_low'].values, novice_data['ci_high'].values))

    # Use centralized helper for grouped bars with per-item colors
    # Includes comparison stars automatically via comparison_pvals parameter
    exp_yerr, nov_yerr = plot_grouped_bars_on_ax(
        ax=ax_top,
        x_positions=x,
        group1_values=exp_vals,
        group1_cis=exp_cis,
        group1_color=roi_colors,  # Per-item colors
        group2_values=nov_vals,
        group2_cis=nov_cis,
        group2_color=roi_colors,  # Same colors for both groups
        group1_label='Expert',
        group2_label='Novice',
        comparison_pvals=pvals,   # Adds stars with connecting lines
        params=PLOT_PARAMS
    )

    # Top panel styling - increased text sizes
    ax_top.set_ylabel('Mean PR (95% CI)', fontsize=PLOT_PARAMS['font_size_label'])
    ax_top.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
    ax_top.set_ylim(top=35, bottom=15)  # Start y-axis at 10 instead of 0

    # Set title and subtitle with increased pad to move them up
    title_params = PLOT_PARAMS.copy()
    title_params['title_pad'] = PLOT_PARAMS['title_pad'] + 200  # Increase pad to move title up
    title_params['font_size_title'] = PLOT_PARAMS['font_size_title']
    title_params['font_size_subtitle'] = PLOT_PARAMS['font_size_subtitle']*.9
    
    set_axis_title(ax_top, 'Participation Ratio', subtitle='FDR p < .05', params=title_params)

    # Legend at center above plot
    ax_top.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, .9),
                 ncol=2, fontsize=PLOT_PARAMS['font_size_legend'])

    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis labels
    style_spines(ax_top, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)

    # =============== BOTTOM PANEL: Differences (Expert - Novice) ===============
    # Use unified plotting function in single-group mode (DRY)
    diff_vals = stats_with_info['mean_diff'].tolist()
    diff_cis = list(zip(stats_with_info['ci95_low'].values, stats_with_info['ci95_high'].values))

    # Plot using centralized function (single-group mode)
    plot_grouped_bars_on_ax(
        ax=ax_bottom,
        x_positions=x,
        group1_values=diff_vals,
        group1_cis=diff_cis,
        group1_color=roi_colors,  # Per-item ROI colors
        group1_pvals=pvals,
        bar_width_multiplier=2.0,  # Wider bars for single-group plot
        params=PLOT_PARAMS
    )

    # Add zero line
    ax_bottom.axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)

    # Bottom panel styling - increased text sizes
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(roi_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    ax_bottom.set_ylabel('ΔPR (Expert − Novice) (95% CI)', fontsize=PLOT_PARAMS['font_size_label'])
    ax_bottom.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
    ax_bottom.set_xlabel('')  # No x-axis label needed

    # Update title params for bottom panel too
    bottom_title_params = PLOT_PARAMS.copy()
    bottom_title_params['font_size_title'] = PLOT_PARAMS['font_size_title']
    set_axis_title(ax_bottom, 'Participation Ratio Difference', subtitle='', params=bottom_title_params)

    # Color x-tick labels: matching color for significant, grey for non-significant
    for i, (ticklabel, sig, color) in enumerate(zip(ax_bottom.get_xticklabels(), is_significant, roi_colors)):
        if sig:
            ticklabel.set_color(color)
        else:
            ticklabel.set_color('#999999')  # Grey for non-significant

    style_spines(ax_bottom, visible_spines=['left'], params=PLOT_PARAMS)

    # Add ROI color legend below bottom panel using centralized helper (optional)
    if show_legend:
        add_roi_color_legend(ax_bottom, roi_info, ncol=7,
                            bbox_to_anchor=(0.5, -0.35), params=PLOT_PARAMS)

    plt.tight_layout()

    # Save combined figure
    combined_path = output_dir / 'pr_roi_combined_panel.pdf'
    fig.savefig(combined_path, bbox_inches='tight', pad_inches=0.1,
                facecolor=PLOT_PARAMS['facecolor'], dpi=PLOT_PARAMS['dpi'])
    plt.close(fig)
    logger.info(f"Saved combined ROI panel figure: {combined_path}")

    return combined_path


def plot_pr_voxel_correlations(
    group_avg: pd.DataFrame,
    diff_data: pd.DataFrame,
    stats: dict,
    output_dir: Path,
    figsize: Tuple[float, float] = None,
    params: dict = PLOT_PARAMS,
) -> None:
    """
    Generate voxel count vs PR correlation plots for each group.

    Creates scatter plots to examine the relationship between ROI size
    (voxel count) and participation ratio values.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Long-format PR results with columns: subject_id, ROI_Label, PR, n_voxels
    participants_df : pd.DataFrame
        Participant metadata with columns: participant_id, group
    roi_info : pd.DataFrame
        ROI metadata with columns: ROI_idx, region_name, color
    output_dir : Path
        Output directory for saving plots
    figsize : tuple, default=(1.75, 1.75)
        Figure size in inches
    params : dict
        Plotting parameters dictionary

    Notes
    -----
    Creates three plots:
    1. Expert PR vs ROI size
    2. Novice PR vs ROI size
    3. PR difference (Expert - Novice) vs ROI size
    """
    figure_style(params)

    # Data is precomputed in analysis; no calculations here

    if figsize is None:
        # Use smaller figure size (half of 'large')
        base_size = params['figure_sizes']['large']
        figsize = (base_size[0], base_size[1])

    # Plot for each group
    for group in ['expert', 'novice']:
        group_data = group_avg[group_avg['group'] == group]

        # Use precomputed correlation
        slope = stats[group]['slope']
        intercept = stats[group]['intercept']
        r = stats[group]['r']
        p = stats[group]['p']

        fig, ax = plt.subplots(figsize=figsize, dpi=params['dpi'])

        # Scatter plot
        ax.scatter(
            group_data['n_voxels'],
            group_data['PR'],
            c=group_data['color'].values,  # Convert to array
            s=params.get('scatter_size', 250),
            alpha=params.get('bar_alpha', 0.7),
            edgecolors='black',
            linewidths=params.get("base_line_width", 1.0)
        )

        # Regression line
        x_range = np.array([group_data['n_voxels'].min(), group_data['n_voxels'].max()])
        ax.plot(x_range, intercept + slope * x_range, 'k--', linewidth=params.get("base_line_width", 1.0), alpha=params.get('line_alpha', 0.5))

        # Labels
        ax.set_xlabel('Voxel Count', fontsize=params['font_size_label'])
        ax.set_ylabel('PR', fontsize=params['font_size_label'])
        ax.set_title(f'{group.capitalize()}\n(r={r:.2f}, p={p:.3f})', fontsize=params['font_size_tick'])

        sns.despine(trim=False)
        plt.tight_layout()

        # Save
        output_path = output_dir / f"pr_voxels_{group}.pdf"
        fig.savefig(output_path, bbox_inches='tight', dpi=params['dpi'])
        plt.close()

        logger.info(f"Saved correlation plot: {output_path}")

    # Plot for difference (Expert - Novice) using precomputed stats
    slope = stats['diff']['slope']
    intercept = stats['diff']['intercept']
    r = stats['diff']['r']
    p = stats['diff']['p']

    fig, ax = plt.subplots(figsize=figsize, dpi=params['dpi'])

    ax.scatter(
        diff_data['n_voxels_avg'],
        diff_data['PR_diff'],
        c=diff_data['color'].values,  # Convert to array
        s=params.get('scatter_size', 250),
        alpha=params.get('bar_alpha', 0.7),
        edgecolors='black',
        linewidths=params.get('base_line_width', 1.0),
    )

    # Regression line
    x_range = np.array([diff_data['n_voxels_avg'].min(), diff_data['n_voxels_avg'].max()])
    ax.plot(x_range, intercept + slope * x_range, 'k--', linewidth=params.get('base_line_width', 1.0), alpha=params.get('line_alpha', 0.5))

    # Zero reference line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=params.get('base_line_width', 1.0), alpha=params.get('line_alpha', 0.3))

    ax.set_xlabel('Avg Voxel Count', fontsize=params['font_size_label'])
    ax.set_ylabel('ΔPR (Experts−Novices)', fontsize=params['font_size_label'])
    ax.set_title(f'Difference\n(r = {r:.2f}, p = {p:.3f})', fontsize=params['font_size_tick'])

    sns.despine(trim=False)
    plt.tight_layout()

    output_path = output_dir / "pr_voxels_difference.pdf"
    fig.savefig(output_path, bbox_inches='tight', dpi=params['dpi'])
    plt.close()

    logger.info(f"Saved difference correlation plot: {output_path}")


# Removed: plot_pr_pca_projections (computations moved to analysis)


def plot_pr_feature_importance(
    clf: Any,
    roi_info: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
    figsize: Tuple[float, float] = None,
) -> None:
    """
    Plot top ROI contributions to expert vs novice classification.

    Shows top 5 novice-predictive ROIs (negative weights) followed by
    top 5 expert-predictive ROIs (positive weights), sorted by magnitude.

    Parameters
    ----------
    clf : LogisticRegression
        Trained classifier on full ROI space
    roi_info : pd.DataFrame
        ROI metadata with columns: ROI_idx, region_name, color
    output_dir : Path
        Output directory for saving plot
    top_n : int, default=10
        Total number of ROIs to display (top_n/2 for each direction)
    figsize : tuple, default=(3.5, 5.0)
        Figure size in inches (increased height for better readability)

    Notes
    -----
    Negative weights (red bars): Higher PR predictive of novice group.
    Positive weights (green bars): Higher PR predictive of expert group.
    ROI names appear on right y-axis.
    """
    figure_style(PLOT_PARAMS)
    if figsize is None:
        # Use wide figure format for horizontal bars
        figsize = (7.0, 3.5)  # Wide aspect ratio

    # Extract weights
    weights = clf.coef_[0]

    # Get ROI names - map by ROI_idx
    roi_name_map = dict(zip(roi_info['ROI_idx'], roi_info['pretty_name']))
    # weights[i] corresponds to ROI label i+1 (since Python is 0-indexed but ROI labels start at 1)

    # Get top N by absolute magnitude, regardless of sign
    # Sort by absolute value and take top_n
    abs_weights = np.abs(weights)
    top_idx = np.argsort(abs_weights)[-top_n:]  # Get indices of top N by magnitude

    # Sort these top N from lowest to highest magnitude (for bottom-to-top display)
    # Since barh plots from bottom (index 0) to top (index n-1)
    top_weights = weights[top_idx]
    top_names = [roi_name_map.get(i+1, f'ROI_{i+1}') for i in top_idx]
    top_names = [name.replace("\\n", " ") for name in top_names]

    # Color by sign: negative = novice (red), positive = expert (green)
    top_colors = [COLORS_EXPERT_NOVICE['novice'] if w < 0 else COLORS_EXPERT_NOVICE['expert'] for w in top_weights]

    combined_weights = top_weights
    combined_names = top_names
    combined_colors = top_colors

    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_PARAMS['dpi'])

    y_pos = np.arange(len(combined_names))
    ax.barh(
        y_pos,
        combined_weights,
        color=combined_colors,
        alpha=PLOT_PARAMS.get('bar_alpha', 0.7),
        edgecolor='black',
        linewidth=PLOT_PARAMS.get('base_line_width', 1.0)
    )

    # ROI names on right y-axis (larger font size for wide format)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined_names, fontsize=PLOT_PARAMS['font_size_label'], color="gray")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.set_xlabel('Weight in Original ROI Space', fontsize=PLOT_PARAMS['font_size_label'])
    ax.tick_params(axis='x', labelsize=PLOT_PARAMS['font_size_tick'])

    # Set x-axis limits to -1 to +1 with ticks at 0.5 intervals
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # Zero reference line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=PLOT_PARAMS.get('base_line_width', 1.0), alpha=PLOT_PARAMS.get('line_alpha', 0.5))

    # Add title FIRST (appears on top)
    ax.set_title('Top 10 Contributions to Classification',
                 fontweight='bold',
                 fontsize=PLOT_PARAMS['font_size_title'],
                 pad=50)  # Extra padding to separate from directional text

    # Add directional arrow text BELOW title
    from matplotlib.font_manager import FontProperties
    arrow_font = FontProperties(family=PLOT_PARAMS["font_family"], size=PLOT_PARAMS['font_size_label']*.75)

    # Create text with color coding (positioned below title)
    novice_text = "← Higher PR predictive of Novices"
    expert_text = "Higher PR predictive of Experts →"

    ax.text(0.20, 1.05, novice_text, ha="center", va="center",
            transform=ax.transAxes, fontproperties=arrow_font,
            color=COLORS_EXPERT_NOVICE['novice'])
    ax.text(0.80, 1.05, expert_text, ha="center", va="center",
            transform=ax.transAxes, fontproperties=arrow_font,
            color=COLORS_EXPERT_NOVICE['expert'])

    sns.despine(ax=ax, trim=False, right=False, left=True)  # Keep right spine, remove left
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pr_feature_importance.pdf"
    fig.savefig(output_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()

    logger.info(f"Saved feature importance plot: {output_path}")




def plot_pr_matrix_and_loadings(
    pr_matrix: np.ndarray,
    n_experts: int,
    roi_labels: np.ndarray,
    output_dir: Path,
    pca_components: np.ndarray,
    roi_info: pd.DataFrame,
    figsize: Tuple[float, float] = (7.0, 24.0),
) -> Path:
    """
    Plot PR matrix (subjects × ROIs) and PCA loadings stacked vertically.

    Both heatmaps have black outlines and squares are sized to match.
    Uses 'mako' colormap for PR matrix (matching old implementation).
    """
    figure_style(PLOT_PARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map labels to pretty names (passed in, not loaded here)
    roi_name_map = dict(zip(roi_info['ROI_idx'].values, roi_info['pretty_name'].values))
    roi_pretty_names = [roi_name_map.get(int(label), str(label)) for label in roi_labels]

    matrix = pr_matrix
    loadings = pca_components

    # Calculate height ratios to ensure square sizes match
    # Both matrices have same width (n_rois), so height ratio should match row counts
    n_subjects = matrix.shape[0]
    n_components = loadings.shape[0]
    height_ratios = [n_subjects, n_components]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=figsize, dpi=PLOT_PARAMS['dpi'],
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.1}
    )

    # Top: PR matrix heatmap with 'mako' colormap (matching old implementation)
    sns.heatmap(matrix, ax=ax_top, cmap='mako', cbar=False,
                xticklabels=False, yticklabels=False,
                linewidths=0, linecolor='none', square=False)  # No internal grid lines
    ax_top.set_title('PR Profiles across Subjects', fontweight='bold', fontsize=PLOT_PARAMS['font_size_tick'])
    ax_top.set_xlabel('')  # No x-axis label
    ax_top.set_ylabel('Subjects (Experts on top)', fontsize=PLOT_PARAMS['font_size_tick'])

    # Add black outline to top panel
    for spine in ax_top.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        spine.set_visible(True)

    # Draw separator line between groups (black)
    if 0 < n_experts < matrix.shape[0]:
        ax_top.axhline(n_experts, color='black', linewidth=2)

    # Bottom: PCA loadings heatmap with ROI names as x-tick labels
    from common.plotting_utils import _make_brain_cmap
    brain_cmap = _make_brain_cmap()
    max_abs = np.abs(loadings).max()
    sns.heatmap(loadings, ax=ax_bottom, cmap=brain_cmap, center=0,
                vmin=-max_abs, vmax=max_abs, cbar=False,
                xticklabels=roi_pretty_names, yticklabels=['PC1', 'PC2'],
                linewidths=0, linecolor='none', square=False)  # No internal grid lines
    ax_bottom.set_title('ROI Contributions to PCA Components', fontweight='bold', fontsize=PLOT_PARAMS['font_size_tick'])
    ax_bottom.set_xlabel('')  # No x-axis label

    # Style x-tick labels (ROI names) - all grey
    roi_pretty_names = [roi_name.replace("\\n", " ") for roi_name in roi_pretty_names]
    ax_bottom.set_xticklabels(roi_pretty_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick']*0.6)
    for ticklabel in ax_bottom.get_xticklabels():
        ticklabel.set_color('gray')  # Grey for all ROI labels
    for ticklabel in ax_bottom.get_yticklabels():
        ticklabel.set_fontsize(PLOT_PARAMS['font_size_tick']*0.7)  # Grey for all ROI labels

    # Add black outline to bottom panel
    for spine in ax_bottom.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        spine.set_visible(True)

    plt.tight_layout()

    out_path = output_dir / 'pr_matrix_and_loadings.pdf'
    fig.savefig(out_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close(fig)

    logger.info(f"Saved PR matrix + loadings figure: {out_path}")
    return out_path


__all__ = [
    'plot_pr_roi_bars',
    'plot_pr_voxel_correlations',
    'plot_pr_matrix_and_loadings',
    'plot_pr_feature_importance',
]
