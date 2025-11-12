#!/usr/bin/env python3
"""
Neurosynth RSA — LaTeX Table Generation
========================================

This script generates a publication-ready LaTeX table summarizing RSA searchlight
correlations between neural pattern dissimilarity and Neurosynth meta-analytic
maps for cognitive terms.

The table presents correlations for three model RDMs (Visual Similarity, Strategy,
Checkmate) across cognitive terms, organized by map type (Positive, Negative,
Difference).

LaTeX tables include correlations with 95% bootstrap confidence intervals formatted
as r [CI_low, CI_high]. Full statistical details (p-values, FDR correction,
significance flags) are exported to CSV.

Inputs
------
- searchlight_visual_similarity_term_corr_*.csv: Visual Similarity RDM correlations
  (with bootstrap CIs, p-values, FDR correction)
- searchlight_strategy_term_corr_*.csv: Strategy RDM correlations
- searchlight_checkmate_term_corr_*.csv: Checkmate RDM correlations

Outputs
-------
- tables/neurosynth_rsa_summary.tex: Combined LaTeX table (correlations only)
- tables/neurosynth_rsa_full_stats.csv: Full statistical results with CIs and p-values
- Manuscript copy: neurosynth_rsa.tex

Dependencies
------------
- common.report_utils: save_table_with_manuscript_copy
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-neurosynth/82_table_neurosynth_rsa.py

Supplementary Analysis: Neurosynth meta-analytic decoding of RSA patterns
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from common import setup_script, log_script_end
from common.tables import generate_styled_table, build_c_only_colspec

# ============================================================================
# Configuration & Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='neurosynth_rsa',
    output_subdirs=['tables'],
    log_name='tables_neurosynth_rsa.log',
)
RESULTS_DIR = results_dir
tables_dir = dirs['tables']

# ============================================================================
# Load Data for All Three Model RDMs
# ============================================================================

logger.info("Loading Neurosynth RSA correlation data...")

# Define the three model RDMs (in centrally defined order: visual, strategy, check)
models = {
    'visual': 'searchlight_visual_similarity',
    'strategy': 'searchlight_strategy',
    'check': 'searchlight_checkmate'
}

# Load data for each model
data = {}
for key, base_name in models.items():
    pos_file = RESULTS_DIR / f"{base_name}_term_corr_positive.csv"
    neg_file = RESULTS_DIR / f"{base_name}_term_corr_negative.csv"
    diff_file = RESULTS_DIR / f"{base_name}_term_corr_difference.csv"

    if not (pos_file.exists() and neg_file.exists() and diff_file.exists()):
        logger.error(f"Missing files for model: {key}")
        raise FileNotFoundError(f"Missing correlation files for {base_name}")

    data[key] = {
        'pos': pd.read_csv(pos_file),
        'neg': pd.read_csv(neg_file),
        'diff': pd.read_csv(diff_file)
    }
    logger.info(f"Loaded data for {key}")

# ============================================================================
# Build Combined Table (using centralized styling)
# ============================================================================

logger.info("Building Neurosynth RSA table via centralized style system...")

# Get terms (should be same for all files)
terms = data['visual']['pos']['term'].tolist()

# Helper function to format correlation with CI for LaTeX
def format_r_with_ci(r, ci_low, ci_high):
    """Format correlation as 'r [ci_low, ci_high]' for LaTeX."""
    return f"{r:.2f} [{ci_low:.2f}, {ci_high:.2f}]"

# Build clean DataFrame with formatted columns for publication
# Include correlation values with 95% bootstrap confidence intervals
rows = []
for i, term in enumerate(terms):
    row = {'Term': term.replace('_', ' ').title()}
    for key, name in [('visual', 'Visual Similarity'), ('strategy', 'Strategy'), ('check', 'Checkmate')]:
        # Positive correlations with CI
        pos_data = data[key]['pos'].iloc[i]
        if 'CI_low' in pos_data and 'CI_high' in pos_data:
            row[f'Pos+_{name}'] = format_r_with_ci(
                float(pos_data['r']),
                float(pos_data['CI_low']),
                float(pos_data['CI_high'])
            )
        else:
            row[f'Pos+_{name}'] = f"{float(pos_data['r']):.2f}"

        # Negative correlations with CI
        neg_data = data[key]['neg'].iloc[i]
        if 'CI_low' in neg_data and 'CI_high' in neg_data:
            row[f'Neg-_{name}'] = format_r_with_ci(
                float(neg_data['r']),
                float(neg_data['CI_low']),
                float(neg_data['CI_high'])
            )
        else:
            row[f'Neg-_{name}'] = f"{float(neg_data['r']):.2f}"

        # Difference with CI
        diff_data = data[key]['diff'].iloc[i]
        if 'CI_low' in diff_data and 'CI_high' in diff_data:
            row[f'Δ_{name}'] = format_r_with_ci(
                float(diff_data['r_diff']),
                float(diff_data['CI_low']),
                float(diff_data['CI_high'])
            )
        else:
            row[f'Δ_{name}'] = f"{float(diff_data['r_diff']):.2f}"
    rows.append(row)

df_out = pd.DataFrame(rows)

# Define multicolumn header groups
multicolumn = {
    'Visual Similarity': ['Pos+_Visual Similarity', 'Neg-_Visual Similarity', 'Δ_Visual Similarity'],
    'Strategy': ['Pos+_Strategy', 'Neg-_Strategy', 'Δ_Strategy'],
    'Checkmate': ['Pos+_Checkmate', 'Neg-_Checkmate', 'Δ_Checkmate']
}

# Generate and save LaTeX table (saves to both results and manuscript folders)
latex_path = tables_dir / 'neurosynth_rsa_summary.tex'
generate_styled_table(
    df=df_out,
    output_path=latex_path,
    caption='Neurosynth RSA: correlations between neural dissimilarity and meta-analytic maps with 95\\% bootstrap confidence intervals. For each model RDM (Visual Similarity, Strategy, Checkmate), columns show correlations with positive and negative maps and their difference. Values formatted as r [CI$_{low}$, CI$_{high}$].',
    label='tab:rsa_searchlight',
    multicolumn_headers=multicolumn,
    column_format=build_c_only_colspec(df_out, multicolumn),
    logger=logger,
    manuscript_name='neurosynth_rsa.tex',
)

# Save CSV version for reference (correlations only)
csv_path = tables_dir / 'neurosynth_rsa_summary.csv'
df_out.to_csv(csv_path, index=False)
logger.info(f"CSV summary table saved to: {csv_path}")

# Save full statistics CSV with all columns (CI, p-values, FDR, significance)
# This provides complete statistical details not shown in the LaTeX table
full_stats_rows = []
for i, term in enumerate(terms):
    for key, name in [('visual', 'Visual Similarity'), ('strategy', 'Strategy'), ('check', 'Checkmate')]:
        # Positive correlations
        pos_data = data[key]['pos'].iloc[i]
        full_stats_rows.append({
            'Model': name,
            'Map_Type': 'Positive',
            'Term': term.replace('_', ' ').title(),
            'r': float(pos_data['r']),
            'CI_low': float(pos_data['CI_low']) if 'CI_low' in pos_data else None,
            'CI_high': float(pos_data['CI_high']) if 'CI_high' in pos_data else None,
            'p_raw': float(pos_data['p_raw']) if 'p_raw' in pos_data else None,
            'p_fdr': float(pos_data['p_fdr']) if 'p_fdr' in pos_data else None,
            'sig': bool(pos_data['sig']) if 'sig' in pos_data else None,
        })
        # Negative correlations
        neg_data = data[key]['neg'].iloc[i]
        full_stats_rows.append({
            'Model': name,
            'Map_Type': 'Negative',
            'Term': term.replace('_', ' ').title(),
            'r': float(neg_data['r']),
            'CI_low': float(neg_data['CI_low']) if 'CI_low' in neg_data else None,
            'CI_high': float(neg_data['CI_high']) if 'CI_high' in neg_data else None,
            'p_raw': float(neg_data['p_raw']) if 'p_raw' in neg_data else None,
            'p_fdr': float(neg_data['p_fdr']) if 'p_fdr' in neg_data else None,
            'sig': bool(neg_data['sig']) if 'sig' in neg_data else None,
        })
        # Difference
        diff_data = data[key]['diff'].iloc[i]
        full_stats_rows.append({
            'Model': name,
            'Map_Type': 'Difference',
            'Term': term.replace('_', ' ').title(),
            'r': float(diff_data['r_diff']),
            'CI_low': float(diff_data['CI_low']) if 'CI_low' in diff_data else None,
            'CI_high': float(diff_data['CI_high']) if 'CI_high' in diff_data else None,
            'p_raw': float(diff_data['p_raw']) if 'p_raw' in diff_data else None,
            'p_fdr': float(diff_data['p_fdr']) if 'p_fdr' in diff_data else None,
            'sig': bool(diff_data['sig']) if 'sig' in diff_data else None,
        })

df_full_stats = pd.DataFrame(full_stats_rows)
full_csv_path = tables_dir / 'neurosynth_rsa_full_stats.csv'
df_full_stats.to_csv(full_csv_path, index=False)
logger.info(f"Full statistics CSV saved to: {full_csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
