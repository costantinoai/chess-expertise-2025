#!/usr/bin/env python3
"""
RSA ROIs — Manuscript Table Generation (Significant ROIs Only)
===============================================================

This script generates publication-ready LaTeX tables showing only significant
ROI-level RSA results where experts show stronger model-brain correspondence
than novices. This creates the supplementary "roi_maps_rsa.tex" table for the
manuscript.

METHODS (Academic Manuscript Section)
--------------------------------------
Supplementary tables present ROI-level results from whole-brain RSA using all
180 bilateral cortical regions from the Glasser multimodal parcellation. Tables
show only ROIs where experts exhibited significantly stronger model-brain
correspondence than novices (FDR-corrected p < .05, Expert > Novice).

For each significant ROI, we report:
1. **ROI name**: Glasser parcellation label
2. **Harvard-Oxford label**: Secondary anatomical label for reference
3. **M_diff**: Mean group difference (Expert - Novice) in Fisher z-transformed
   correlation
4. **t statistic**: From Welch's two-sample t-test
5. **95% CI**: Confidence interval for the group difference
6. **p_FDR**: FDR-corrected p-value (Benjamini-Hochberg, α=0.05)

Separate tables are generated for:
- **Checkmate model RSA**: ROIs showing stronger checkmate model correspondence
  in experts
- **Strategy model RSA**: ROIs showing stronger strategy model correspondence
  in experts

Inputs
------
- rsa_group_stats.pkl: Group-level statistics from 01_rsa_roi_summary.py
  Contains t-statistics, p-values, mean differences, CIs for all ROIs
- roi_info_with_ho_labels.tsv: ROI metadata with pre-computed Harvard-Oxford labels
  from 01_rsa_roi_summary.py

Outputs
-------
- tables/roi_maps_rsa.tex: Combined LaTeX table with both models (copied to manuscript)
- tables/roi_maps_rsa.csv: Combined CSV for reference

Dependencies
------------
- common.report_utils: save_table_with_manuscript_copy for dual-saving
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-supplementary/rsa-rois/82_table_roi_maps_rsa.py

Supplementary Analysis: Significant ROI-level RSA results (180 ROIs)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pickle

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from common import CONFIG
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common import setup_script
from common.report_utils import save_table_with_manuscript_copy
from common.formatters import format_ci, format_p_cell, shorten_roi_name

# ============================================================================
# Configuration & Setup
# ============================================================================

# Locate the latest RSA ROIs results directory
results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='rsa_rois',
    output_subdirs=['tables'],
    log_name='tables_roi_maps_rsa.log',
)

# Set up logging
tables_dir = dirs['tables']

# ============================================================================
# Load Data
# ============================================================================

logger.info("Loading RSA group statistics...")

# Load pickle file with t-statistics and group comparisons
with open(results_dir / 'rsa_group_stats.pkl', 'rb') as f:
    group_stats = pickle.load(f)

# Load ROI metadata with Harvard-Oxford labels (pre-computed in 01_rsa_roi_summary.py)
logger.info("Loading ROI metadata with Harvard-Oxford labels...")
roi_info = pd.read_csv(results_dir / 'roi_info_with_ho_labels.tsv', sep='\t')

logger.info(f"Loaded metadata for {len(roi_info)} ROIs with Harvard-Oxford mapping")

# ============================================================================
# Helper Function: Filter and Format Significant Results
# ============================================================================

def filter_and_format_significant(target_key: str, target_label: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Extract significant ROIs for one target where Expert > Novice.

    Parameters
    ----------
    target_key : str
        Key for the target in group_stats (e.g., 'checkmate', 'strategy')
    target_label : str
        Display name for logging (e.g., 'Checkmate', 'Strategy')
    alpha : float
        Significance threshold for FDR-corrected p-value

    Returns
    -------
    pd.DataFrame
        Filtered and formatted DataFrame with significant ROIs
    """
    logger.info(f"Processing {target_label} model...")

    # Extract Welch t-test results
    welch_df = group_stats['rsa_corr'][target_key]['welch_expert_vs_novice'].copy()

    logger.info(f"  Total ROIs: {len(welch_df)}")

    # Merge with ROI metadata to get pretty names and Harvard-Oxford labels
    welch_df = welch_df.merge(
        roi_info[['roi_id', 'pretty_name', 'harvard_oxford_label']],
        left_on='ROI_Label',
        right_on='roi_id',
        how='left'
    )

    # Filter for significant results where Expert > Novice
    # mean_diff > 0 means Expert mean > Novice mean
    sig_df = welch_df[
        (welch_df['p_val_fdr'] < alpha) &
        (welch_df['mean_diff'] > 0)
    ].copy()

    logger.info(f"  Significant ROIs (p_FDR < {alpha}, Expert > Novice): {len(sig_df)}")

    # Sort by p-value (most significant first)
    sig_df = sig_df.sort_values('p_val_fdr')

    return sig_df

# ============================================================================
# Extract Significant Results
# ============================================================================

logger.info("Filtering for significant ROIs...")

# Process checkmate model
checkmate_sig = filter_and_format_significant('checkmate', 'Checkmate', alpha=0.05)

# Process strategy model
strategy_sig = filter_and_format_significant('strategy', 'Strategy', alpha=0.05)

# ============================================================================
# Build LaTeX Tables
# ============================================================================

logger.info("Building LaTeX tables...")

def build_latex_table(df: pd.DataFrame, model_label: str, table_label: str) -> str:
    """
    Build a LaTeX table for one model's significant ROIs.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame with significant results
    model_label : str
        Display name for the model (e.g., "Checkmate model")
    table_label : str
        LaTeX label for the table

    Returns
    -------
    str
        Complete LaTeX table code
    """
    lines = []

    # Table header
    lines.append(r'\begin{table}[p]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\linewidth}{!}{%')
    lines.append(r'\begin{tabular}{llSScc}')
    lines.append(r'\toprule')
    lines.append(r'ROI & Harvard-Oxford Label & $M_{\text{diff}}$ & $t$ & 95\% CI & $p$ \\')
    lines.append(r'\midrule')

    # Table rows
    for _, row in df.iterrows():
        roi_name = shorten_roi_name(row['pretty_name'])
        ho_label = row['harvard_oxford_label'] if pd.notna(row['harvard_oxford_label']) else 'Unlabeled'
        m_diff = row['mean_diff']
        t_stat = row['t_stat']
        ci_low = row['ci95_low']
        ci_high = row['ci95_high']
        p_fdr = row['p_val_fdr']

        # Format p-value and CI
        p_str = format_p_cell(p_fdr)
        ci_str = format_ci(float(ci_low), float(ci_high), precision=3, latex=False)

        lines.append(
            f'{roi_name} & {ho_label} & {float(m_diff):.3f} & '
            f'{float(t_stat):.3f} & {ci_str} & {p_str} \\\\'
        )

    # Table footer
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')

    # Caption
    lines.append(
        f'\\caption{{\\textbf{{{model_label} (RSA): Experts $>$ Novices.}} '
        f'Glasser ROIs showing stronger model–brain correspondence in Experts than Novices '
        f'when summarizing searchlight RSA within parcels. '
        f'Columns report the Glasser ROI label, secondary Harvard–Oxford label '
        f'(assigned by ROI center of mass), '
        f'mean group difference $M_{{\\text{{diff}}}}$ (Experts $-$ Novices), '
        f'$t$ statistic, 95\\% CI of the difference, and FDR-corrected $p$ value ($\\alpha<.05$).}}'
    )

    lines.append(f'\\label{{{table_label}}}')
    lines.append(r'\end{table}')
    lines.append('')  # Blank line between tables

    return '\n'.join(lines)

# Build both tables
checkmate_tex = build_latex_table(
    checkmate_sig,
    'Checkmate model',
    'supptab:roi_analysis_rsa_check'
)

strategy_tex = build_latex_table(
    strategy_sig,
    'Strategy model',
    'supptab:roi_analysis_rsa_strategy'
)

# Combine tables
combined_tex = checkmate_tex + '\n' + strategy_tex

# ============================================================================
# Save Tables
# ============================================================================

logger.info("Saving LaTeX table...")

# Save combined LaTeX table
tex_path = tables_dir / 'roi_maps_rsa.tex'
save_table_with_manuscript_copy(
    combined_tex,
    tex_path,
    manuscript_name='roi_maps_rsa.tex',
    logger=logger
)

# Save combined CSV for reference
combined_csv = pd.concat([
    checkmate_sig.assign(Model='Checkmate')[['Model', 'pretty_name', 'harvard_oxford_label', 'mean_diff', 't_stat', 'ci95_low', 'ci95_high', 'p_val_fdr']],
    strategy_sig.assign(Model='Strategy')[['Model', 'pretty_name', 'harvard_oxford_label', 'mean_diff', 't_stat', 'ci95_low', 'ci95_high', 'p_val_fdr']]
], ignore_index=True)

csv_path = tables_dir / 'roi_maps_rsa.csv'
combined_csv.to_csv(csv_path, index=False)
logger.info(f"Saved combined CSV: {csv_path}")

# ============================================================================
# Summary
# ============================================================================

logger.info("="*80)
logger.info(f"ROI Maps RSA Table Generation Complete")
logger.info(f"  Checkmate model: {len(checkmate_sig)} significant ROIs")
logger.info(f"  Strategy model: {len(strategy_sig)} significant ROIs")
logger.info(f"  Combined LaTeX saved to: {tex_path}")
logger.info(f"  Copied to manuscript: final_results/tables/roi_maps_rsa.tex")
logger.info("="*80)

log_script_end(logger)
