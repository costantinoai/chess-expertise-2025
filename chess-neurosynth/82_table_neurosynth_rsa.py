#!/usr/bin/env python3
"""
Neurosynth RSA — LaTeX Table Generation
========================================

This script generates a publication-ready LaTeX table summarizing RSA searchlight
correlations between neural pattern dissimilarity and Neurosynth meta-analytic
maps for cognitive terms.

The table presents correlations for three model RDMs (Visual Similarity, Strategy,
Checkmate) across cognitive terms, organized by map type (Positive, Negative,
Difference). Simplified format with only correlation values (no CI or p-values).

Inputs
------
- searchlight_visual_similarity_term_corr_*.csv: Visual Similarity RDM correlations
- searchlight_strategy_term_corr_*.csv: Strategy RDM correlations
- searchlight_checkmate_term_corr_*.csv: Checkmate RDM correlations

Outputs
-------
- tables/neurosynth_rsa_summary.tex: Combined LaTeX table
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

# Build clean DataFrame with formatted columns for publication
# Use column names that will shorten nicely: Pos+/Neg+/Diff+ for each group
rows = []
for i, term in enumerate(terms):
    row = {'Term': term.replace('_', ' ').title()}
    for key, name in [('visual', 'Visual Similarity'), ('strategy', 'Strategy'), ('check', 'Checkmate')]:
        row[f'Pos+_{name}'] = float(data[key]['pos'].iloc[i]['r'])
        row[f'Neg-_{name}'] = float(data[key]['neg'].iloc[i]['r'])
        row[f'Δ_{name}'] = float(data[key]['diff'].iloc[i]['r_diff'])
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
    caption='Neurosynth RSA: correlations between neural dissimilarity and meta-analytic maps. For each model RDM (Visual Similarity, Strategy, Checkmate), columns show correlations with positive and negative maps and their difference.',
    label='tab:rsa_searchlight',
    multicolumn_headers=multicolumn,
    column_format=build_c_only_colspec(df_out, multicolumn),
    logger=logger,
    manuscript_name='neurosynth_rsa.tex',
)

# Save CSV version for reference
csv_path = tables_dir / 'neurosynth_rsa_summary.csv'
df_out.to_csv(csv_path, index=False)
logger.info(f"CSV table saved to: {csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
