"""
LaTeX table generation for manifold analysis results.

This module provides functions for creating publication-ready tables
from participation ratio analysis results.

Functions
---------
generate_pr_results_table : Create multi-column LaTeX table with group comparisons
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Add repo root to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.report_utils import save_table_with_manuscript_copy
from common.tables import generate_styled_table, build_c_only_colspec
from common.formatters import format_ci as fmt_ci_global, format_p_cell, shorten_roi_name

logger = logging.getLogger(__name__)


def generate_pr_results_table(
    summary_stats: pd.DataFrame,
    stats_results: pd.DataFrame,
    roi_info: pd.DataFrame,
    output_dir: Path,
    use_fdr: bool = True,
    filename_tex: str = "pr_results_table.tex",
    filename_csv: str = "pr_results_table.csv",
) -> Tuple[Path, Path]:
    """
    Generate LaTeX and CSV tables of participation ratio results.

    Creates a multi-column table with:
    - ROI names
    - Expert group: Mean PR and 95% CI
    - Novice group: Mean PR and 95% CI
    - Group difference (Expert - Novice): Mean Δ and 95% CI
    - Statistical significance (p-value or FDR-corrected p-value)

    Parameters
    ----------
    summary_stats : pd.DataFrame
        Descriptive statistics with columns: ROI_Label, group, mean_PR, ci_low, ci_high
    stats_results : pd.DataFrame
        Statistical test results with columns: ROI_Label, mean_diff, ci95_low, ci95_high,
        p_val, p_val_fdr
    roi_info : pd.DataFrame
        ROI metadata with columns: roi_id, pretty_name
    output_dir : Path
        Output directory for saving tables
    use_fdr : bool, default=True
        Whether to report FDR-corrected p-values
    filename_tex : str, default="pr_results_table.tex"
        Output filename for LaTeX table
    filename_csv : str, default="pr_results_table.csv"
        Output filename for CSV table

    Returns
    -------
    Tuple[Path, Path]
        Paths to saved LaTeX and CSV files

    Notes
    -----
    LaTeX table uses booktabs package (toprule, midrule, bottomrule).
    Values are formatted to 2 decimal places.
    P-values < 0.001 are shown as "$<.001$".
    """
    # Helper formatting functions
    def fmt_val(v: float) -> str:
        """Format value to 3 decimals or '--' if NaN."""
        return "--" if np.isnan(v) else f"{float(v):.3f}"

    def fmt_ci(low: float, high: float) -> str:
        """Format confidence interval to 3 decimals."""
        if np.isnan(low) or np.isnan(high):
            return "{--}{--}"
        return fmt_ci_global(float(low), float(high), precision=3, latex=False, use_numrange=True)

    def fmt_p(p: float) -> str:
        """Format p-value cell (APA style)."""
        if np.isnan(p):
            return "--"
        return format_p_cell(float(p))

    # Merge data sources
    # Get expert and novice stats
    expert_stats = summary_stats[summary_stats['group'] == 'expert'].copy()
    novice_stats = summary_stats[summary_stats['group'] == 'novice'].copy()

    # Merge with ROI names (use pretty_name for tables)
    expert_stats = expert_stats.merge(
        roi_info[['roi_id', 'pretty_name']],
        left_on='ROI_Label',
        right_on='roi_id',
        how='left'
    )
    novice_stats = novice_stats.merge(
        roi_info[['roi_id', 'pretty_name']],
        left_on='ROI_Label',
        right_on='roi_id',
        how='left'
    )
    stats_with_names = stats_results.merge(
        roi_info[['roi_id', 'pretty_name']],
        left_on='ROI_Label',
        right_on='roi_id',
        how='left'
    )

    # Build table rows
    table_rows = []
    for _, expert_row in expert_stats.iterrows():
        roi_label = expert_row['ROI_Label']
        roi_name = shorten_roi_name(expert_row['pretty_name'])

        # Get corresponding novice and stats rows
        novice_row = novice_stats[novice_stats['ROI_Label'] == roi_label].iloc[0]
        stats_row = stats_with_names[stats_with_names['ROI_Label'] == roi_label].iloc[0]

        table_rows.append({
            'ROI': roi_name,
            'Expert_mean': fmt_val(expert_row['mean_PR']),
            'Expert_CI': fmt_ci(expert_row['ci_low'], expert_row['ci_high']),
            'Novice_mean': fmt_val(novice_row['mean_PR']),
            'Novice_CI': fmt_ci(novice_row['ci_low'], novice_row['ci_high']),
            'Delta_mean': fmt_val(stats_row['mean_diff']),
            'Delta_CI': fmt_ci(stats_row['ci95_low'], stats_row['ci95_high']),
            'p_raw': fmt_p(stats_row['p_val']),
            'pFDR': fmt_p(stats_row['p_val_fdr']),
        })

    table_df = pd.DataFrame(table_rows)

    # ========== Save Files ==========
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / filename_tex
    csv_path = output_dir / filename_csv

    multicolumn = {
        'Experts': ['Expert_mean', 'Expert_CI'],
        'Novices': ['Novice_mean', 'Novice_CI'],
        'Experts $-$ Novices': ['Delta_mean', 'Delta_CI'],
    }
    generate_styled_table(
        df=table_df,
        output_path=tex_path,
        caption=(
            "Participation Ratio (PR) by ROI: group means with 95% confidence intervals, "
            "group differences (Expert $-$ Novice) from Welch's $t$-tests (t-based CIs), "
            "and both raw and FDR-corrected $p$-values."
        ),
        label='tab:pr_results',
        multicolumn_headers=multicolumn,
        column_format=build_c_only_colspec(table_df, multicolumn_headers=multicolumn),
        logger=logger,
        manuscript_name='pr_ttest.tex',
    )

    table_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV table: {csv_path}")

    return tex_path, csv_path


__all__ = [
    'generate_pr_results_table',
]
