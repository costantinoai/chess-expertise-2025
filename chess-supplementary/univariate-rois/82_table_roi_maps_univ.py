#!/usr/bin/env python3
"""Univariate ROIs — Manuscript Table (Significant ROIs Only)"""
from pathlib import Path
import pandas as pd
import pickle  # noqa: S403 — used only to load trusted internal artefacts

from common import setup_script, log_script_end
from common.formatters import format_ci, format_p_cell, shorten_roi_name

# Unified results tree: reads 01_univariate_roi_summary.py's outputs from
# results/supplementary/univariate-rois/data/ and writes the final
# manuscript table into results/supplementary/univariate-rois/tables/.
results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='univariate_rois',
    output_subdirs=['tables'],
    log_name='tables_roi_maps_univ.log',
)
tables_dir = dirs['tables']

# Load data (pickle produced by chess-supplementary/univariate-rois/01_univariate_roi_summary.py)
logger.info("Loading univariate group statistics...")
with open(results_dir / 'univ_group_stats.pkl', 'rb') as f:
    group_stats = pickle.load(f)

logger.info("Loading ROI metadata with Harvard-Oxford labels...")
roi_info = pd.read_csv(results_dir / 'roi_info_with_ho_labels.tsv', sep='\t')
logger.info(f"Loaded metadata for {len(roi_info)} ROIs")

# Filter significant ROIs (con_0002 is All > Rest)
logger.info("Filtering for significant ROIs (All > Rest, Expert > Novice)...")
welch_df = group_stats['contrasts']['con_0002']['welch_expert_vs_novice'].copy()
welch_df = welch_df.merge(
    roi_info[['roi_id', 'pretty_name', 'harvard_oxford_label']],
    left_on='ROI_Label',
    right_on='roi_id',
    how='left'
)

sig_df = welch_df[(welch_df['p_val_fdr'] < 0.05) & (welch_df['mean_diff'] > 0)].copy()
sig_df = sig_df.sort_values('p_val_fdr')
logger.info(f"Significant ROIs: {len(sig_df)}")

logger.info("Formatting table with centralized generator...")
rows = []
for _, row in sig_df.iterrows():
    ho = row['harvard_oxford_label'] if pd.notna(row['harvard_oxford_label']) else 'Unlabeled'
    p_raw_str = format_p_cell(row['p_val'])
    p_fdr_str = format_p_cell(row['p_val_fdr'])
    ci_str = format_ci(float(row['ci95_low']), float(row['ci95_high']), precision=3, latex=False, use_numrange=True)
    roi_name = shorten_roi_name(row['pretty_name'])
    rows.append({
        'ROI': roi_name,
        'Harvard–Oxford Label': ho,
        'M_diff': float(row['mean_diff']),
        't': float(row['t_stat']),
        '95% CI': ci_str,
        'p': p_raw_str,
        'pFDR': p_fdr_str,
    })

_df_out = pd.DataFrame(rows)
from common.tables import generate_styled_table

generate_styled_table(
    df=_df_out,
    output_path=tables_dir / 'roi_maps_univ.tex',
    caption=(
        'Univariate contrast \\emph{All} > \\emph{Rest}: Experts > Novices. '
        'Glasser ROIs with higher second-level GLM contrast values in Experts than Novices. '
        'Columns show Glasser ROI label, Harvard–Oxford label, '
        'mean group difference $M_{\\text{diff}}$ (contrast units; Experts $-$ Novices), '
        '$t$ statistic, 95\\% CI, raw $p$, and FDR-corrected $p$ value ($\\alpha<.05$).'
    ),
    label='supptab:roi_analysis_univ_allrest',
    column_format='llccccc',
    logger=logger,
    manuscript_name='roi_maps_univ.tex',
)
logger.info("Saved LaTeX table via centralized generator")

sig_df[['pretty_name', 'harvard_oxford_label', 'mean_diff', 't_stat', 'ci95_low', 'ci95_high', 'p_val', 'p_val_fdr']].to_csv(
    tables_dir / 'roi_maps_univ.csv',
    index=False
)
logger.info(f"Saved CSV: {tables_dir / 'roi_maps_univ.csv'}")

logger.info("="*80)
logger.info(f"ROI Maps Univariate Complete: {len(sig_df)} significant ROIs")
logger.info("="*80)
log_script_end(logger)
