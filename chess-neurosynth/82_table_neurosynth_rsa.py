#!/usr/bin/env python3
"""
Neurosynth RSA — LaTeX table generation

Reads per-pattern correlation CSVs produced by 02_rsa_neurosynth.py and
builds publication-ready LaTeX tables:
 - POS vs NEG correlations (simple r values)
 - DIFF correlations (Δr = r_POS − r_NEG)
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.report_utils import generate_latex_table


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_neurosynth_rsa',
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
    log_name='tables_neurosynth_rsa.log',
)

tables_dir = RESULTS_DIR / 'tables'
tables_dir.mkdir(parents=True, exist_ok=True)

pos_files = sorted(RESULTS_DIR.glob('*_term_corr_positive.csv'))

for pos in pos_files:
    base = pos.name.replace('_term_corr_positive.csv', '')
    neg = RESULTS_DIR / f"{base}_term_corr_negative.csv"
    diff = RESULTS_DIR / f"{base}_term_corr_difference.csv"
    if not (neg.exists() and diff.exists()):
        raise FileNotFoundError(f"Missing companion CSV for pattern '{base}'")

    df_pos = pd.read_csv(pos)
    df_neg = pd.read_csv(neg)
    df_diff = pd.read_csv(diff)

    # POS vs NEG combined table (simplified: just r values, no CIs or p-values)
    df_comb = pd.DataFrame({
        'Term': df_pos['term'].str.title(),
        'r_POS': df_pos['r'].round(3),
        'r_NEG': df_neg['r'].round(3),
    })
    multicolumn = {
        'Z+': ['r_POS'],
        'Z−': ['r_NEG'],
    }
    tex1 = generate_latex_table(
        df=df_comb,
        output_path=tables_dir / f'{base}_pos_neg.tex',
        caption=f'Neurosynth RSA correlations — {base} (Z+ and Z−).',
        label=f'tab:neuro_rsa_{base}_posneg',
        column_format='lc|c',
        multicolumn_headers=multicolumn,
        escape=False,
        logger=logger,
    )

    # DIFF table (Δr = r_POS − r_NEG)
    df_d = pd.DataFrame({
        'Term': df_diff['term'].str.title(),
        'r_POS': df_diff['r_pos'].round(3),
        'r_NEG': df_diff['r_neg'].round(3),
        'Δr': df_diff['r_diff'].round(3),
    })
    tex2 = generate_latex_table(
        df=df_d,
        output_path=tables_dir / f'{base}_diff.tex',
        caption=f'Neurosynth RSA correlation differences — {base} (Δr = r_POS − r_NEG).',
        label=f'tab:neuro_rsa_{base}_diff',
        column_format='lccc',
        multicolumn_headers=None,
        escape=False,
        logger=logger,
    )
    logger.info(f"Saved tables for pattern {base}: {tex1}, {tex2}")

log_script_end(logger)

