#!/usr/bin/env python3
"""
Neurosynth Univariate — LaTeX table generation

Reads per-run correlation CSVs produced by 01_univariate_neurosynth.py and
builds publication-ready LaTeX tables:
 - POS vs NEG correlations (multicolumn)
 - DIFF correlations (Δr with 95% CI)
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.report_utils import generate_latex_table


def _fmt_ci(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]", axis=1)


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_neurosynth_univariate',
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
    log_name='tables_neurosynth_univariate.log',
)

tables_dir = RESULTS_DIR / 'tables'
tables_dir.mkdir(parents=True, exist_ok=True)

# Discover run IDs based on saved CSV files
pos_files = sorted(RESULTS_DIR.glob('*_term_corr_positive.csv'))

for pos in pos_files:
    base = pos.name.replace('_term_corr_positive.csv', '')
    neg = RESULTS_DIR / f"{base}_term_corr_negative.csv"
    diff = RESULTS_DIR / f"{base}_term_corr_difference.csv"
    if not (neg.exists() and diff.exists()):
        raise FileNotFoundError(f"Missing companion CSV for run '{base}'")

    df_pos = pd.read_csv(pos)
    df_neg = pd.read_csv(neg)
    df_diff = pd.read_csv(diff)

    # POS vs NEG combined table
    df_comb = pd.DataFrame({
        'Term': df_pos['term'].str.title(),
        'r_POS': df_pos['r'].round(3),
        '95%_CI_POS': _fmt_ci(df_pos),
        'pFDR_POS': df_pos['p_fdr'].map(lambda x: f"{x:.3e}"),
        'r_NEG': df_neg['r'].round(3),
        '95%_CI_NEG': _fmt_ci(df_neg),
        'pFDR_NEG': df_neg['p_fdr'].map(lambda x: f"{x:.3e}"),
    })
    multicolumn = {
        'POS': ['r_POS', '95%_CI_POS', 'pFDR_POS'],
        'NEG': ['r_NEG', '95%_CI_NEG', 'pFDR_NEG'],
    }
    tex1 = generate_latex_table(
        df=df_comb,
        output_path=tables_dir / f'{base}_pos_neg.tex',
        caption=f'Neurosynth univariate correlations — {base} (POS and NEG).',
        label=f'tab:neuro_uni_{base}_posneg',
        column_format='lccc|ccc',
        multicolumn_headers=multicolumn,
        escape=False,
        logger=logger,
    )

    # DIFF table
    df_d = pd.DataFrame({
        'Term': df_diff['term'].str.title(),
        'Δr': df_diff['r_diff'].round(3),
        '95%_CI': _fmt_ci(df_diff),
        'pFDR': df_diff['p_fdr'].map(lambda x: f"{x:.3e}"),
    })
    tex2 = generate_latex_table(
        df=df_d,
        output_path=tables_dir / f'{base}_diff.tex',
        caption=f'Neurosynth univariate correlation differences — {base} (Δr = r_POS − r_NEG).',
        label=f'tab:neuro_uni_{base}_diff',
        column_format='lccc',
        multicolumn_headers=None,
        escape=False,
        logger=logger,
    )
    logger.info(f"Saved tables for run {base}: {tex1}, {tex2}")

log_script_end(logger)

