#!/usr/bin/env python3
"""
Generate LaTeX tables for RSA ROI summary.

Loads rsa_group_stats.pkl from the latest rsa-rois run and produces
per-target tables: ROI, Experts mean (95% CI), Novices mean (95% CI),
Experts−Novices difference (95% CI), and FDR-corrected p-values.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd

# Enable repo root imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common import CONFIG
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.report_utils import generate_latex_table, format_roi_stats_table
from modules import RSA_TARGETS


results_base = Path(__file__).parent / 'results'
results_dir = find_latest_results_directory(
    results_base,
    pattern='*_rsa_rois',
    create_subdirs=['tables'],
    require_exists=True,
    verbose=True,
)

extra = {"RESULTS_DIR": str(results_dir)}
config, _, logger = setup_analysis_in_dir(
    results_dir,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='tables_rsa_rois.log',
)

tables_dir = results_dir / 'tables'

with open(results_dir / 'rsa_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_180'])
targets = list(RSA_TARGETS.keys())


for tgt in targets:
    if 'rsa_corr' not in index or tgt not in index['rsa_corr']:
        continue
    blocks = index['rsa_corr'][tgt]
    df = format_roi_stats_table(
        blocks['welch_expert_vs_novice'],
        blocks['experts_desc'],
        blocks['novices_desc'],
        roi_info,
    )
    tgt_label = blocks.get('label', tgt)

    multicolumn = {
        'Experts': ['Expert_mean', 'Expert_CI'],
        'Novices': ['Novice_mean', 'Novice_CI'],
        'Experts−Novices': ['Delta_mean', 'Delta_CI'],
    }
    tex_path = tables_dir / f'rsa_{tgt}.tex'
    csv_path = tables_dir / f'rsa_{tgt}.csv'

    _ = generate_latex_table(
        df=df,
        output_path=tex_path,
        caption=f'RSA ROI summary — {tgt_label}.',
        label=f'tab:rsa_{tgt}',
        column_format='lcc|cc|cc|c',
        multicolumn_headers=multicolumn,
        escape=False,
        logger=logger,
    )
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved RSA table for {tgt}: {tex_path}")

log_script_end(logger)
