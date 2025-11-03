#!/usr/bin/env python3
"""MVPA Extended Dimensions — Manuscript Table (Finer-grained dimensions, 22 ROIs)"""
import sys
from pathlib import Path
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import CONFIG
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.report_utils import save_table_with_manuscript_copy
from common.formatters import format_ci, format_p_cell, shorten_roi_name

# Find RSA and decoding results directories
rsa_dir = find_latest_results_directory(Path(__file__).parent / 'results', pattern='*_rsa', require_exists=True, verbose=True)
dec_dir = find_latest_results_directory(Path(__file__).parent / 'results', pattern='*_decoding', require_exists=True, verbose=True)

# Use RSA dir for logging/output
_, _, logger = setup_analysis_in_dir(rsa_dir, script_file=__file__,
                                      extra_config={"RESULTS_DIR": str(rsa_dir)},
                                      suppress_warnings=True, log_name='tables_mvpa_extended.log')
tables_dir = rsa_dir.parent / 'tables'
tables_dir.mkdir(exist_ok=True)

# Load data
logger.info("Loading MVPA RSA and decoding statistics...")
with open(rsa_dir / 'mvpa_group_stats.pkl', 'rb') as f:
    rsa_stats = pickle.load(f)
with open(dec_dir / 'mvpa_group_stats.pkl', 'rb') as f:
    dec_stats = pickle.load(f)

# Load ROI metadata (22 bilateral regions)
roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
logger.info(f"Loaded {len(roi_info)} ROIs")

# Define finer-grained dimensions (checkmate boards only)
finer_dims = ['strategy_half', 'check_n_half', 'legal_moves_half', 'motif_half', 'total_pieces_half']
dim_labels = ['Strategy', 'Moves to Checkmate', 'Legal Moves', 'Motifs', 'Total Pieces']

# Build RSA table
logger.info("Building RSA table...")
rsa_lines = [
    r'\begin{table}[p]',
    r'\centering',
    r'\resizebox{\linewidth}{!}{%',
    r'\begin{tabular}{lScc|Scc|Scc|Scc|Scc}',
    r'\toprule',
    r'\multirow{2}{*}{ROI}',
    r'  & \multicolumn{3}{c|}{Strategy}',
    r'  & \multicolumn{3}{c|}{Moves to Checkmate}',
    r'  & \multicolumn{3}{c|}{Legal Moves}',
    r'  & \multicolumn{3}{c|}{Motifs}',
    r'  & \multicolumn{3}{c}{Total Pieces} \\',
    r'  & $\Delta r$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta r$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta r$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta r$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta r$ & 95\% CI & $p_\mathrm{FDR}$ \\',
    r'\midrule'
]

for _, roi_row in roi_info.iterrows():
    roi_name = shorten_roi_name(roi_row['pretty_name'])
    line = roi_name
    for dim in finer_dims:
        if 'rsa_corr' in rsa_stats and dim in rsa_stats['rsa_corr']:
            welch = rsa_stats['rsa_corr'][dim]['welch_expert_vs_novice']
            roi_data = welch[welch['ROI_Label'] == roi_row['roi_id']]
            if len(roi_data) > 0:
                row = roi_data.iloc[0]
                dr = float(row['mean_diff'])
                ci_l, ci_h = float(row['ci95_low']), float(row['ci95_high'])
                p = row['p_val_fdr']
                p_str = format_p_cell(p)
                ci_str = format_ci(ci_l, ci_h, precision=3, latex=False)
                line += f' & {dr:.3f} & {ci_str} & {p_str}'
            else:
                line += ' & -- & -- & --'
        else:
            line += ' & -- & -- & --'
    line += ' \\\\'
    rsa_lines.append(line)

rsa_lines.extend([
    r'\bottomrule',
    r'\end{tabular}',
    r'}',
    r'\caption{Summary of ROI-level RSA results using correlation-based representational similarity. '
    r'Values reflect the expert–novice difference in RSA for five finer (i.e., derived from checkmate boards only) regressors. '
    r'All values are bootstrapped means with 95\% confidence intervals and FDR-corrected $p$-values.}',
    r'\label{tab:rsa_roi_summary_checkonly}',
    r'\end{table}',
    ''
])

# Build decoding table (same structure but with Δacc)
logger.info("Building decoding table...")
dec_lines = [
    r'\begin{table}[p]',
    r'\centering',
    r'\resizebox{\linewidth}{!}{%',
    r'\begin{tabular}{lScc|Scc|Scc|Scc|Scc}',
    r'\toprule',
    r'\multirow{2}{*}{ROI}',
    r'  & \multicolumn{3}{c|}{Strategy}',
    r'  & \multicolumn{3}{c|}{Moves to Mate}',
    r'  & \multicolumn{3}{c|}{Legal Moves}',
    r'  & \multicolumn{3}{c|}{Motifs}',
    r'  & \multicolumn{3}{c}{Total Pieces} \\',
    r'  & $\Delta acc$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta acc$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta acc$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta acc$ & 95\% CI & $p_\mathrm{FDR}$',
    r'  & $\Delta acc$ & 95\% CI & $p_\mathrm{FDR}$ \\',
    r'\midrule'
]

for _, roi_row in roi_info.iterrows():
    roi_name = shorten_roi_name(roi_row['pretty_name'])
    line = roi_name
    for dim in finer_dims:
        if 'decoding' in dec_stats and dim in dec_stats['decoding']:
            welch = dec_stats['decoding'][dim]['welch_expert_vs_novice']
            roi_data = welch[welch['ROI_Label'] == roi_row['roi_id']]
            if len(roi_data) > 0:
                row = roi_data.iloc[0]
                dacc = float(row['mean_diff'])
                ci_l, ci_h = float(row['ci95_low']), float(row['ci95_high'])
                p = row['p_val_fdr']
                p_str = format_p_cell(p)
                ci_str = format_ci(ci_l, ci_h, precision=3, latex=False)
                line += f' & {dacc:.3f} & {ci_str} & {p_str}'
            else:
                line += ' & -- & -- & --'
        else:
            line += ' & -- & -- & --'
    line += ' \\\\'
    dec_lines.append(line)

dec_lines.extend([
    r'\bottomrule',
    r'\end{tabular}',
    r'}',
    r'\caption{Summary of ROI-level decoding results using SVM classification. '
    r'Values reflect decoding accuracy differences (\(\Delta acc\)) between expert and novice participants '
    r'for five regressors: Strategy, Moves to Checkmate, Legal Moves, Motifs, and Total Pieces (checkmate boards only). '
    r'All values are bootstrapped means with 95\% confidence intervals and FDR-corrected \(p\)-values.}',
    r'\label{tab:svm_roi_summary_checkonly}',
    r'\end{table}'
])

# Combine and save
combined_tex = '\n'.join(rsa_lines) + '\n\n\n' + '\n'.join(dec_lines)
save_table_with_manuscript_copy(
    combined_tex,
    tables_dir / 'mvpa_extended_dimensions.tex',
    manuscript_name='mvpa_extended_dimensions.tex',
    logger=logger
)

logger.info("="*80)
logger.info(f"MVPA Extended Dimensions Table Complete")
logger.info(f"  RSA: 5 dimensions × {len(roi_info)} ROIs")
logger.info(f"  Decoding: 5 dimensions × {len(roi_info)} ROIs")
logger.info("="*80)
log_script_end(logger)
