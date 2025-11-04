#!/usr/bin/env python3
"""
Behavioral RSA — LaTeX Table Generation (Experts vs Novices)
==============================================================

This script generates publication-ready LaTeX and CSV tables summarizing the
correlations between behavioral representational dissimilarity matrices (RDMs)
and theoretical model RDMs for expert and novice chess players.

METHODS (Academic Manuscript Section)
--------------------------------------
Statistical tables were generated from the behavioral RSA correlation results.
For each theoretical model (checkmate status, strategy type, visual similarity),
we report Spearman correlation coefficients (r) with behavioral RDMs, 95%
confidence intervals (bootstrapped using 10,000 iterations via pingouin),
uncorrected p-values, and false discovery rate (FDR)-corrected p-values
(Benjamini-Hochberg procedure, α=0.05). Results are presented separately for
experts and novices to allow direct comparison of model fit between expertise
levels.

The output LaTeX table uses multicolumn headers to group expert and novice
statistics, with columns for: correlation coefficient (r), 95% confidence
interval, uncorrected p-value, and FDR-corrected p-value. Significance markers
(*, **, ***) indicate FDR-corrected significance levels (p<0.05, p<0.01, p<0.001).

Inputs
------
- correlation_results.pkl: Dictionary containing:
  - 'expert': List of tuples (model_name, r, p, ci_low, ci_high) for experts
  - 'novice': List of tuples (model_name, r, p, ci_low, ci_high) for novices
  - 'expert_p_fdr': Dict mapping model names to FDR-corrected p-values (experts)
  - 'novice_p_fdr': Dict mapping model names to FDR-corrected p-values (novices)
  - 'model_columns': List of model names (e.g., ['check', 'strategy', 'visual'])
  - 'alpha_fdr': FDR alpha level (default: 0.05)
  - 'fdr_method': FDR correction method (default: 'fdr_bh')

Outputs
-------
- tables/behavioral_rsa_correlations.tex: LaTeX table with multicolumn headers
- tables/behavioral_rsa_correlations.csv: CSV version for reference

Dependencies
------------
- common.report_utils: Table formatting utilities (create_correlation_table, generate_latex_table)
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-behavioral/81_table_behavioral_correlations.py

Analysis 1 from manuscript: Supplementary Table, Methods Sec 3.5.3
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from common import (
    CONFIG,
    MODEL_LABELS,
    setup_script,
    log_script_end,
    generate_expert_novice_table,
)

# ============================================================================
# Configuration & Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='behavioral_rsa',
    output_subdirs=['tables'],
    log_name='tables_behavioral.log',
)
tables_dir = dirs['tables']

# ============================================================================
# Load Correlation Results
# ============================================================================

logger.info("Generating behavioral RSA correlation table (Experts vs Novices)...")

def _format_behavioral_correlations(data: dict):
    from common.report_utils import create_correlation_table
    return create_correlation_table(
        expert_results=data['expert'],
        novice_results=data['novice'],
        model_labels=MODEL_LABELS,
        exp_p_fdr=data.get('expert_p_fdr', {}),
        nov_p_fdr=data.get('novice_p_fdr', {}),
    )

tex_path, csv_path = generate_expert_novice_table(
    results_dir=results_dir,
    output_dir=tables_dir,
    table_name='behavioral_rsa_correlations',
    caption='Behavioral-model RSA correlations (Experts vs Novices).',
    label='tab:behavioral_rsa_correlations',
    formatter_func=_format_behavioral_correlations,
    pickle_name='correlation_results.pkl',
    # Let central style infer multicolumn headers and c-only colspec robustly
    column_format=None,
    manuscript_name='behavioral_rsa_correlations.tex',
    logger=logger,
)

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
logger.info(f"LaTeX table written to: {tex_path}")
