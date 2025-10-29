#!/usr/bin/env python3
"""
MVPA Decoding — LaTeX table generation (Experts vs Novices)

Loads mvpa_group_stats.pkl from the latest SVM group analysis and
produces per-target LaTeX tables with means (95% CI) and FDR p-values.
Values are reported as (accuracy − chance) using the stored chance in the
group stats index.
"""

import os
import sys
from pathlib import Path
import pickle
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.report_utils import generate_latex_table
from common import CONFIG


def _fmt(m, lo, hi, subtract: float) -> tuple[str, str]:
    if any(pd.isna([m, lo, hi])):
        return '--', '[--, --]'
    return f"{(m - subtract):.3f}", f"[{(lo - subtract):.3f}, {(hi - subtract):.3f}]"


def _build_table(blocks: dict, roi_info: pd.DataFrame) -> pd.DataFrame:
    welch = blocks['welch_expert_vs_novice']
    exp_desc = blocks['experts_desc']
    nov_desc = blocks['novices_desc']
    chance = float(blocks.get('chance', 0.0))

    df = welch.merge(roi_info[['roi_id', 'pretty_name']], left_on='ROI_Label', right_on='roi_id', how='left')
    df['ROI'] = df['pretty_name'].str.replace('\n', ' ', regex=False)

    exp_vals = []
    exp_cis = []
    for t in exp_desc:
        v, c = _fmt(*t, subtract=chance)
        exp_vals.append(v)
        exp_cis.append(c)

    nov_vals = []
    nov_cis = []
    for t in nov_desc:
        v, c = _fmt(*t, subtract=chance)
        nov_vals.append(v)
        nov_cis.append(c)

    df_out = pd.DataFrame({
        'ROI': df['ROI'],
        'Expert_mean_minusChance': exp_vals,
        'Expert_CI_minusChance': exp_cis,
        'Novice_mean_minusChance': nov_vals,
        'Novice_CI_minusChance': nov_cis,
        'pFDR': df['p_val_fdr'].map(lambda p: '--' if pd.isna(p) else f"{p:.3e}"),
    })
    return df_out


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_mvpa_group_decoding',
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
    log_name='tables_mvpa_decoding.log',
)

tables_dir = RESULTS_DIR / 'tables'

with open(RESULTS_DIR / 'mvpa_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
targets = sorted(index.get('svm', {}).keys())

for tgt in targets:
    blocks = index['svm'][tgt]
    df = _build_table(blocks, roi_info)
    multicolumn = {
        'Experts (acc−chance)': ['Expert_mean_minusChance', 'Expert_CI_minusChance'],
        'Novices (acc−chance)': ['Novice_mean_minusChance', 'Novice_CI_minusChance'],
    }
    tex = generate_latex_table(
        df=df,
        output_path=tables_dir / f'mvpa_svm_{tgt}.tex',
        caption=f'MVPA decoding (ROI-level) — target: {tgt}.',
        label=f'tab:mvpa_svm_{tgt}',
        column_format='lcc|cc|c',
        multicolumn_headers=multicolumn,
        escape=False,
        logger=logger,
    )
    csv_path = tables_dir / f'mvpa_svm_{tgt}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved decoding table for {tgt}: {tex}")

log_script_end(logger)

