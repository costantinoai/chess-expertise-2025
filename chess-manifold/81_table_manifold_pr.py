#!/usr/bin/env python3
"""
Manifold PR — LaTeX table generation

Loads pr_results.pkl from the latest manifold analysis and builds a
multicolumn LaTeX table showing group means (95% CI) and group differences
with FDR p-values per ROI using modules.tables.generate_pr_results_table.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from modules.tables import generate_pr_results_table


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_manifold',
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
    log_name='tables_manifold.log',
)

tables_dir = RESULTS_DIR / 'tables'

with open(RESULTS_DIR / 'pr_results.pkl', 'rb') as f:
    res = pickle.load(f)

summary_stats = res['summary_stats']
stats_results = res['stats_results']
roi_info = res['roi_info']

tex_path, csv_path = generate_pr_results_table(
    summary_stats=summary_stats,
    stats_results=stats_results,
    roi_info=roi_info,
    output_dir=tables_dir,
    use_fdr=True,
    filename_tex='manifold_pr_results.tex',
    filename_csv='manifold_pr_results.csv',
)

log_script_end(logger)
logger.info(f"Saved PR results LaTeX: {tex_path}")
logger.info(f"Saved PR results CSV: {csv_path}")

