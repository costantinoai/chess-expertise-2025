#!/usr/bin/env python3
"""
Behavioral RSA — LaTeX table generation (Experts vs Novices)

Loads correlation_results.pkl from the latest behavioral_rsa results and
produces a multicolumn LaTeX table summarizing model correlations with
95% CIs and FDR-corrected p-values for Experts and Novices.
"""

import os
import sys
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.report_utils import (
    create_correlation_table,
    generate_latex_table,
)
from common import CONFIG, MODEL_LABELS


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_behavioral_rsa',
    create_subdirs=['tables'],
    require_exists=True,
    verbose=True,
)

extra = {"RESULTS_DIR": str(RESULTS_DIR)}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='tables_behavioral.log',
)

tables_dir = RESULTS_DIR / 'tables'

with open(RESULTS_DIR / 'correlation_results.pkl', 'rb') as f:
    corr = pickle.load(f)

expert = corr['expert']
novice = corr['novice']
model_columns = corr.get('model_columns', CONFIG.get('MODEL_COLUMNS', []))
exp_p_fdr = corr.get('expert_p_fdr', {})
nov_p_fdr = corr.get('novice_p_fdr', {})

# Build DataFrame rows
df = create_correlation_table(
    expert_results=expert,
    novice_results=novice,
    model_labels=MODEL_LABELS,
    exp_p_fdr=exp_p_fdr,
    nov_p_fdr=nov_p_fdr,
)

# Multicolumn headers mapping
multicolumn = {
    'Experts': ['r_Experts', '95%_CI_Experts', 'p_Experts', 'pFDR_Experts'],
    'Novices': ['r_Novices', '95%_CI_Novices', 'p_Novices', 'pFDR_Novices'],
}

tables_dir.mkdir(parents=True, exist_ok=True)
tex_path = generate_latex_table(
    df=df,
    output_path=tables_dir / 'behavioral_rsa_correlations.tex',
    caption='Behavioral-model RSA correlations (Experts vs Novices).',
    label='tab:behavioral_rsa_correlations',
    column_format='lcccc|cccc',
    multicolumn_headers=multicolumn,
    escape=False,
    logger=logger,
)

csv_path = tables_dir / 'behavioral_rsa_correlations.csv'
df.to_csv(csv_path, index=False)
logger.info(f"Saved CSV table: {csv_path}")

log_script_end(logger)
logger.info(f"LaTeX table written to: {tex_path}")
