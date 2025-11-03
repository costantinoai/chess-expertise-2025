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

The table presents correlations for seven cognitive terms across two contrasts,
organized by map type (Positive, Negative, Difference).

Inputs
------
- spmT_exp-gt-nonexp_all-gt-rest_term_corr_*.csv: All Boards > Rest correlations
- spmT_exp-gt-nonexp_check-gt-nocheck_term_corr_*.csv: Checkmate correlations

Outputs
-------
- tables/neurosynth_univariate_summary.tex: Combined LaTeX table
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
from common.tables import generate_styled_table

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

# Build clean DataFrame with formatted columns for publication
rows = []
for i, term in enumerate(terms):
    row = {'Term': term.replace('_', ' ').title()}
    row['All > Rest_Pos'] = float(data['all_rest']['pos'].iloc[i]['r'])
    row['All > Rest_Neg'] = float(data['all_rest']['neg'].iloc[i]['r'])
    row['All > Rest_Diff'] = float(data['all_rest']['diff'].iloc[i]['r_diff'])
    row['Check vs Non-Check_Pos'] = float(data['check']['pos'].iloc[i]['r'])
    row['Check vs Non-Check_Neg'] = float(data['check']['neg'].iloc[i]['r'])
    row['Check vs Non-Check_Diff'] = float(data['check']['diff'].iloc[i]['r_diff'])
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
    caption='Neurosynth meta-analytic correlations for two contrasts: All > Rest and Check vs Non-Check. For each term, columns show correlations with positive and negative maps and their difference.',
    label='tab:neurosynth_univ',
    multicolumn_headers=multicolumn,
    column_format='lSSS|SSS',
    logger=logger,
    manuscript_name='neurosynth_univ.tex',
)

# Save CSV version for reference
csv_path = tables_dir / 'neurosynth_univariate_summary.csv'
df_out.to_csv(csv_path, index=False)
logger.info(f"CSV table saved to: {csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
