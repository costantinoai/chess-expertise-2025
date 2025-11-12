#!/usr/bin/env python3
"""
Neurosynth Univariate — LaTeX Table Generation
===============================================

This script generates a publication-ready LaTeX table summarizing univariate
correlations between brain activation patterns and Neurosynth meta-analytic
maps for cognitive terms.

METHODS (Academic Manuscript Section)
--------------------------------------
For each univariate contrast (All Boards > Rest, Checkmate > Non-Checkmate),
we computed voxel-wise spatial correlations between the Experts > Novices group
contrast map and Neurosynth meta-analytic z-score maps. Correlations were
computed separately for positive (Z+) and negative (Z−) meta-analytic maps,
with the difference (Z+ minus Z−) representing net directional association.

Bootstrap confidence intervals (10,000 resamples) and FDR-corrected p-values
provide robust statistical inference accounting for spatial dependencies.

The table presents correlations for seven cognitive terms across two contrasts,
organized by map type (Positive, Negative, Difference).

LaTeX tables include correlations with 95% bootstrap confidence intervals formatted
as r [CI_low, CI_high]. Full statistical details (p-values, FDR correction,
significance flags) are exported to CSV.

Inputs
------
- spmT_exp-gt-nonexp_all-gt-rest_term_corr_*.csv: All Boards > Rest correlations
  (with bootstrap CIs, p-values, FDR correction)
- spmT_exp-gt-nonexp_check-gt-nocheck_term_corr_*.csv: Checkmate correlations

Outputs
-------
- tables/neurosynth_univariate_summary.tex: Combined LaTeX table (correlations only)
- tables/neurosynth_univariate_full_stats.csv: Full statistical results with CIs and p-values
- Manuscript copy: neurosynth_univ.tex

Dependencies
------------
- common.report_utils: save_table_with_manuscript_copy
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-neurosynth/81_table_neurosynth_univariate.py

Supplementary Analysis: Neurosynth meta-analytic decoding of univariate contrasts
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
    results_pattern='neurosynth_univariate',
    output_subdirs=['tables'],
    log_name='tables_neurosynth_univariate.log',
)
RESULTS_DIR = results_dir
tables_dir = dirs['tables']

# ============================================================================
# Load Data for Both Contrasts
# ============================================================================

logger.info("Loading Neurosynth univariate correlation data...")

# Define the two contrasts we need
contrasts = {
    'all_rest': 'spmT_exp-gt-nonexp_all-gt-rest',
    'check': 'spmT_exp-gt-nonexp_check-gt-nocheck'
}

# Load data for each contrast
data = {}
for key, base_name in contrasts.items():
    pos_file = RESULTS_DIR / f"{base_name}_term_corr_positive.csv"
    neg_file = RESULTS_DIR / f"{base_name}_term_corr_negative.csv"
    diff_file = RESULTS_DIR / f"{base_name}_term_corr_difference.csv"

    if not (pos_file.exists() and neg_file.exists() and diff_file.exists()):
        logger.error(f"Missing files for contrast: {key}")
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

logger.info("Building Neurosynth univariate table via centralized style system...")

# Get terms (should be same for all files)
terms = data['all_rest']['pos']['term'].tolist()

# Helper function to format correlation with CI for LaTeX
def format_r_with_ci(r, ci_low, ci_high):
    """Format correlation as 'r [ci_low, ci_high]' for LaTeX."""
    return f"{r:.2f} [{ci_low:.2f}, {ci_high:.2f}]"

# Build clean DataFrame with formatted columns for publication
# Include correlation values with 95% bootstrap confidence intervals
rows = []
for i, term in enumerate(terms):
    row = {'Term': term.replace('_', ' ').title()}

    # All > Rest contrast
    for key, label_prefix in [('all_rest', 'All > Rest'), ('check', 'Check vs Non-Check')]:
        # Positive correlations with CI
        pos_data = data[key]['pos'].iloc[i]
        if 'CI_low' in pos_data and 'CI_high' in pos_data:
            row[f'{label_prefix}_Pos'] = format_r_with_ci(
                float(pos_data['r']),
                float(pos_data['CI_low']),
                float(pos_data['CI_high'])
            )
        else:
            row[f'{label_prefix}_Pos'] = f"{float(pos_data['r']):.2f}"

        # Negative correlations with CI
        neg_data = data[key]['neg'].iloc[i]
        if 'CI_low' in neg_data and 'CI_high' in neg_data:
            row[f'{label_prefix}_Neg'] = format_r_with_ci(
                float(neg_data['r']),
                float(neg_data['CI_low']),
                float(neg_data['CI_high'])
            )
        else:
            row[f'{label_prefix}_Neg'] = f"{float(neg_data['r']):.2f}"

        # Difference with CI
        diff_data = data[key]['diff'].iloc[i]
        if 'CI_low' in diff_data and 'CI_high' in diff_data:
            row[f'{label_prefix}_Diff'] = format_r_with_ci(
                float(diff_data['r_diff']),
                float(diff_data['CI_low']),
                float(diff_data['CI_high'])
            )
        else:
            row[f'{label_prefix}_Diff'] = f"{float(diff_data['r_diff']):.2f}"

    rows.append(row)

df_out = pd.DataFrame(rows)

# Define multicolumn header groups
multicolumn = {
    'All > Rest': ['All > Rest_Pos', 'All > Rest_Neg', 'All > Rest_Diff'],
    'Check vs Non-Check': ['Check vs Non-Check_Pos', 'Check vs Non-Check_Neg', 'Check vs Non-Check_Diff']
}

# Generate and save LaTeX table (saves to both results and manuscript folders)
latex_path = tables_dir / 'neurosynth_univariate_summary.tex'
generate_styled_table(
    df=df_out,
    output_path=latex_path,
    caption='Neurosynth meta-analytic correlations with 95\\% bootstrap confidence intervals for two contrasts: All > Rest and Check vs Non-Check. For each term, columns show correlations with positive and negative maps and their difference. Values formatted as r [CI$_{low}$, CI$_{high}$].',
    label='tab:neurosynth_univ',
    multicolumn_headers=multicolumn,
    column_format=build_c_only_colspec(df_out, multicolumn),
    logger=logger,
    manuscript_name='neurosynth_univ.tex',
)

# Save CSV version for reference (correlations only)
csv_path = tables_dir / 'neurosynth_univariate_summary.csv'
df_out.to_csv(csv_path, index=False)
logger.info(f"CSV summary table saved to: {csv_path}")

# Save full statistics CSV with all columns (CI, p-values, FDR, significance)
# This provides complete statistical details not shown in the LaTeX table
full_stats_rows = []
for i, term in enumerate(terms):
    for key, name in [('all_rest', 'All > Rest'), ('check', 'Check vs Non-Check')]:
        # Positive correlations
        pos_data = data[key]['pos'].iloc[i]
        full_stats_rows.append({
            'Contrast': name,
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
            'Contrast': name,
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
            'Contrast': name,
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
full_csv_path = tables_dir / 'neurosynth_univariate_full_stats.csv'
df_full_stats.to_csv(full_csv_path, index=False)
logger.info(f"Full statistics CSV saved to: {full_csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
