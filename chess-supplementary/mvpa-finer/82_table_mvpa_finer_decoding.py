#!/usr/bin/env python3
"""
MVPA-Finer Decoding — LaTeX tables (Experts vs Novices)

Loads mvpa_group_stats.pkl from the latest MVPA finer decoding group analysis and
produces per-target LaTeX tables with group means (95% CI) and FDR p-values.
Values are reported as (accuracy − chance).
"""

import os
import sys
from pathlib import Path
import pickle
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add repo root for 'common' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.report_utils import generate_latex_table, format_roi_stats_table
from common import CONFIG


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_mvpa_finer_group_decoding',
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
    log_name='tables_mvpa_finer_decoding.log',
)

tables_dir = RESULTS_DIR / 'tables'

with open(RESULTS_DIR / 'mvpa_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
targets = sorted(index.get('svm', {}).keys())

for tgt in targets:
    blocks = index['svm'][tgt]
    chance = float(blocks.get('chance', 0.0))
    df = format_roi_stats_table(
        blocks['welch_expert_vs_novice'],
        blocks['experts_desc'],
        blocks['novices_desc'],
        roi_info,
        subtract_chance=chance,
    )
    df = df.rename(columns={
        'Expert_mean': 'Expert_mean_minusChance',
        'Expert_CI': 'Expert_CI_minusChance',
        'Novice_mean': 'Novice_mean_minusChance',
        'Novice_CI': 'Novice_CI_minusChance',
    })
    multicolumn = {
        'Experts (acc−chance)': ['Expert_mean_minusChance', 'Expert_CI_minusChance'],
        'Novices (acc−chance)': ['Novice_mean_minusChance', 'Novice_CI_minusChance'],
    }
    tex = generate_latex_table(
        df=df,
        output_path=tables_dir / f'mvpa_finer_svm_{tgt}.tex',
        caption=f'MVPA finer decoding (ROI-level) — target: {tgt}.',
        label=f'tab:mvpa_finer_svm_{tgt}',
        column_format='lcc|cc|c',
        multicolumn_headers=multicolumn,
        escape=False,
        logger=logger,
    )
    csv_path = tables_dir / f'mvpa_finer_svm_{tgt}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved decoding finer table for {tgt}: {tex}")

log_script_end(logger)

