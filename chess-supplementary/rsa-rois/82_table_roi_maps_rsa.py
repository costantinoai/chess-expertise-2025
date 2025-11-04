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
# Build LaTeX Tables (central generator)
# ============================================================================

logger.info("Building tables with centralized generator...")

from common.tables import generate_styled_table

def build_df(sig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in sig.iterrows():
        roi_name = shorten_roi_name(row['pretty_name'])
        ho_label = row['harvard_oxford_label'] if pd.notna(row['harvard_oxford_label']) else 'Unlabeled'
        ci_str = format_ci(float(row['ci95_low']), float(row['ci95_high']), precision=3, latex=False, use_numrange=True)
        p_str = format_p_cell(row['p_val_fdr'])
        rows.append({
            'ROI': roi_name,
            'Harvard–Oxford Label': ho_label,
            'M_diff': float(row['mean_diff']),
            't': float(row['t_stat']),
            '95% CI': ci_str,
            'p': p_str,
        })
    return pd.DataFrame(rows)

df_check = build_df(checkmate_sig)
df_strategy = build_df(strategy_sig)

check_path = generate_styled_table(
    df=df_check,
    output_path=tables_dir / 'roi_maps_rsa_check.tex',
    caption=(
        'Checkmate model (RSA): Experts > Novices. Glasser ROIs showing stronger model–brain correspondence '
        'in Experts than Novices. Columns: Glasser ROI label; Harvard–Oxford label; '
        '$M_{\text{diff}}$ (Experts $-$ Novices); $t$; 95% CI; FDR-corrected $p$.'
    ),
    label='supptab:roi_analysis_rsa_check',
    column_format='llcccc',
    logger=logger,
    manuscript_name='roi_maps_rsa_check.tex',
)

strategy_path = generate_styled_table(
    df=df_strategy,
    output_path=tables_dir / 'roi_maps_rsa_strategy.tex',
    caption=(
        'Strategy model (RSA): Experts > Novices. Glasser ROIs showing stronger model–brain correspondence '
        'in Experts than Novices. Columns: Glasser ROI label; Harvard–Oxford label; '
        '$M_{\text{diff}}$ (Experts $-$ Novices); $t$; 95% CI; FDR-corrected $p$.'
    ),
    label='supptab:roi_analysis_rsa_strategy',
    column_format='llcccc',
    logger=logger,
    manuscript_name='roi_maps_rsa_strategy.tex',
)

# Combined file (backward-compat single include)
combined_tex = check_path.read_text() + "\n\n" + strategy_path.read_text()
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

# Remove intermediate per-model files to keep only the combined table
try:
    check_path.unlink(missing_ok=True)
    strategy_path.unlink(missing_ok=True)
    logger.info("Removed intermediate per-model .tex files (kept combined only)")
except Exception:
    pass

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
