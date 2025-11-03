#!/usr/bin/env python3
"""Univariate ROIs — Manuscript Table (Significant ROIs Only)"""
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.report_utils import save_table_with_manuscript_copy
from common.formatters import format_ci, format_p_cell, shorten_roi_name

# Find results directory
results_dir = find_latest_results_directory(
    Path(__file__).parent / 'results',
    pattern='*univariate_rois',
    create_subdirs=['tables'],
    require_exists=True,
    verbose=True
)

# Setup logging
_, _, logger = setup_analysis_in_dir(
    results_dir,
    script_file=__file__,
    extra_config={"RESULTS_DIR": str(results_dir)},
    suppress_warnings=True,
    log_name='tables_roi_maps_univ.log'
)

tables_dir = results_dir / 'tables'

# Load data
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

# Build LaTeX table
lines = [
    r'\begin{table}[p]',
    r'\centering',
    r'\resizebox{\linewidth}{!}{%',
    r'\begin{tabular}{llSScc}',
    r'\toprule',
    r'ROI & Harvard-Oxford Label & $M_{\text{diff}}$ & $t$ & 95\% CI & $p$ \\',
    r'\midrule'
]

for _, row in sig_df.iterrows():
    ho = row['harvard_oxford_label'] if pd.notna(row['harvard_oxford_label']) else 'Unlabeled'
    p_str = format_p_cell(row['p_val_fdr'])
    ci_str = format_ci(float(row['ci95_low']), float(row['ci95_high']), precision=3, latex=False)
    roi_name = shorten_roi_name(row['pretty_name'])
    lines.append(
        f'{roi_name} & {ho} & {float(row["mean_diff"]):.3f} & {float(row["t_stat"]):.3f} & '
        f'{ci_str} & {p_str} \\\\'
    )

lines.extend([
    r'\bottomrule',
    r'\end{tabular}',
    r'}',
    r'\caption{\textbf{Univariate contrast \emph{All} $>$ \emph{Rest}: Experts $>$ Novices.} '
    r'Glasser ROIs with higher second-level GLM contrast values in Experts than Novices. '
    r'Columns show Glasser ROI label, Harvard–Oxford label, '
    r'mean group difference $M_{\text{diff}}$ (contrast units; Experts $-$ Novices), '
    r'$t$ statistic, 95\% CI, and FDR-corrected $p$ value ($\alpha<.05$).}',
    r'\label{supptab:roi_analysis_univ_allrest}',
    r'\end{table}'
])

# Save
save_table_with_manuscript_copy(
    '\n'.join(lines),
    tables_dir / 'roi_maps_univ.tex',
    manuscript_name='roi_maps_univ.tex',
    logger=logger
)

sig_df[['pretty_name', 'harvard_oxford_label', 'mean_diff', 't_stat', 'ci95_low', 'ci95_high', 'p_val_fdr']].to_csv(
    tables_dir / 'roi_maps_univ.csv',
    index=False
)
logger.info(f"Saved CSV: {tables_dir / 'roi_maps_univ.csv'}")

logger.info("="*80)
logger.info(f"ROI Maps Univariate Complete: {len(sig_df)} significant ROIs")
logger.info("="*80)
log_script_end(logger)
