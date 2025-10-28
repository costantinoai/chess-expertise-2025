#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MVPA Decoding/RSA Group Results — Fine Dimensions

Loads artifacts from 03_mvpa_group_analysis_fine.py and generates publication-ready
figures and LaTeX tables using centralized plotting and reporting utilities.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent  # supplementary/ -> repo
sys.path.insert(0, str(repo_root))

from common import CONFIG, apply_nature_rc
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.neuro_utils import get_roi_names_and_colors
from common.plotting import plot_grouped_bars_with_ci, select_roi_labels_for_plot
from common.report_utils import generate_latex_table, create_figure_summary


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

RESULTS_DIR_NAME = None  # if None, use latest *_mvpa_group_fine


# -----------------------------------------------------------------------------
# Locate results and set up logging in-place
# -----------------------------------------------------------------------------
RESULTS_BASE = script_dir / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_mvpa_group_fine',
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=['figures', 'tables'],
    require_exists=True,
    verbose=True,
)

FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config={'RESULTS_DIR': str(RESULTS_DIR)},
    suppress_warnings=True,
    log_name='plotting.log',
)


# -----------------------------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------------------------
with open(RESULTS_DIR / 'mvpa_group_fine_stats.pkl', 'rb') as f:
    group_stats = pickle.load(f)

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
roi_names_default, roi_colors_default = get_roi_names_and_colors(CONFIG['ROI_GLASSER_22'])


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
apply_nature_rc()

for method, results in group_stats.items():
    for tgt, blocks in results.items():
        welch = blocks['welch_expert_vs_novice']
        # Choose ROI labels/colors dynamically if mismatch with defaults
        roi_names, roi_colors = select_roi_labels_for_plot(
            welch_df=welch,
            default_names=roi_names_default,
            default_colors=roi_colors_default,
        )

        # Use descriptive stats saved by group analysis (mean and 95% CI)
        exp_desc = blocks['experts_desc']
        nov_desc = blocks['novices_desc']
        exp_means = np.array([m for (m, _, _) in exp_desc])
        exp_cis = [(lo, hi) for (_, lo, hi) in exp_desc]
        nov_means = np.array([m for (m, _, _) in nov_desc])
        nov_cis = [(lo, hi) for (_, lo, hi) in nov_desc]

        pvals = welch['p_val'].values
        out_pdf = FIGURES_DIR / f"mvpa_fine_{method}_{tgt}_experts_vs_novices.pdf"

        plot_grouped_bars_with_ci(
            group1_values=exp_means,
            group1_cis=exp_cis,
            group2_values=nov_means,
            group2_cis=nov_cis,
            x_labels=roi_names,
            group1_color=roi_colors,
            group2_color=roi_colors,
            comparison_pvals=pvals,
            title=f"FINE {method.upper()} • {tgt}",
            output_path=out_pdf,
        )
        logger.info(f"Saved {out_pdf.name}")


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
rows = []
for method, results in group_stats.items():
    for tgt, blocks in results.items():
        welch = blocks['welch_expert_vs_novice'].copy()
        welch.insert(0, 'Method', method)
        welch.insert(1, 'Target', tgt)
        rows.append(welch)

if rows:
    all_welch = pd.concat(rows, axis=0, ignore_index=True)
    csv_path = TABLES_DIR / 'mvpa_fine_welch_all.csv'
    all_welch.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path.name}")

    sig = all_welch[all_welch['significant_fdr'] == True][['Method', 'Target', 'ROI_Name', 't_stat', 'p_val_fdr', 'cohen_d']]
    if not sig.empty:
        latex_path = generate_latex_table(
            df=sig,
            output_path=TABLES_DIR / 'mvpa_fine_welch_significant.tex',
            caption='Fine MVPA group differences (Experts vs Novices), Welch t-tests with FDR correction.',
            label='tab:mvpa_fine_welch_significant',
            column_format='l l l r r r',
        )
        logger.info(f"Saved {latex_path.name}")


# -----------------------------------------------------------------------------
# Figure summary
# -----------------------------------------------------------------------------
create_figure_summary(
    results_dir=RESULTS_DIR,
    figures_dir=FIGURES_DIR,
    tables_dir=TABLES_DIR,
    analysis_name='MVPA Decoding/RSA (Fine)',
    logger=logger,
)

log_script_end(logger)
logger.info(f"All outputs saved to: {RESULTS_DIR}")
