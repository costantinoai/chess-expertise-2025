#!/usr/bin/env python3
"""
MVPA RSA — LaTeX table generation (Experts vs Novices)

Loads mvpa_group_stats.pkl from the latest RSA group analysis and
produces per-target LaTeX tables with Experts/Novices means (95% CI),
group differences (95% CI), and FDR-corrected p-values.
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


def _build_table(blocks: dict, roi_info: pd.DataFrame) -> pd.DataFrame:
    welch = blocks['welch_expert_vs_novice']
    exp_desc = blocks['experts_desc']
    nov_desc = blocks['novices_desc']

    # Merge pretty names
    df = welch.merge(roi_info[['roi_id', 'pretty_name']], left_on='ROI_Label', right_on='roi_id', how='left')
    df['ROI'] = df['pretty_name'].str.replace('\n', ' ', regex=False)

    # Means and CI
    def _fmt_triplet(t):
        m, lo, hi = t
        if pd.isna(m) or pd.isna(lo) or pd.isna(hi):
            return '--', '[--, --]'
        return f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]"

    exp_vals, exp_cis = zip(*(_fmt_triplet(t) for t in exp_desc))
    nov_vals, nov_cis = zip(*(_fmt_triplet(t) for t in nov_desc))

    # Difference CI from welch df
    df_out = pd.DataFrame({
        'ROI': df['ROI'],
        'Expert_mean': list(exp_vals),
        'Expert_CI': list(exp_cis),
        'Novice_mean': list(nov_vals),
        'Novice_CI': list(nov_cis),
        'Delta_mean': df['mean_diff'].map(lambda v: '--' if pd.isna(v) else f"{v:.3f}"),
        'Delta_CI': pd.Series(zip(df['ci95_low'], df['ci95_high'])).map(
            lambda x: '[--, --]' if any(pd.isna(list(x))) else f"[{x[0]:.3f}, {x[1]:.3f}]"
        ),
        'pFDR': df['p_val_fdr'].map(lambda p: '--' if pd.isna(p) else f"{p:.3e}"),
    })
    return df_out


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_mvpa_group_rsa',
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
    log_name='tables_mvpa_rsa.log',
)

tables_dir = RESULTS_DIR / 'tables'

with open(RESULTS_DIR / 'mvpa_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
targets = sorted(index.get('rsa_corr', {}).keys())

for tgt in targets:
    blocks = index['rsa_corr'][tgt]
    df = _build_table(blocks, roi_info)
    multicolumn = {
        'Experts': ['Expert_mean', 'Expert_CI'],
        'Novices': ['Novice_mean', 'Novice_CI'],
        'Experts−Novices': ['Delta_mean', 'Delta_CI'],
    }
    tex = generate_latex_table(
        df=df,
        output_path=tables_dir / f'mvpa_rsa_{tgt}.tex',
        caption=f'MVPA RSA (ROI-level) — target: {tgt}.',
        label=f'tab:mvpa_rsa_{tgt}',
        column_format='lcc|cc|cc|c',
        multicolumn_headers=multicolumn,
        escape=False,
        logger=logger,
    )
    csv_path = tables_dir / f'mvpa_rsa_{tgt}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved RSA table for {tgt}: {tex}")

log_script_end(logger)

