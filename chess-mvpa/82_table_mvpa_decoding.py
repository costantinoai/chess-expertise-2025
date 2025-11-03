#!/usr/bin/env python3
"""
MVPA Decoding — LaTeX Table Generation (Experts vs Novices)
============================================================

This script generates a publication-ready LaTeX table summarizing multi-voxel
pattern analysis (MVPA) decoding accuracy results using support vector machine
(SVM) classifiers, comparing expert and novice groups across brain regions for
two classification targets: Checkmate and Strategy.

METHODS (Academic Manuscript Section)
--------------------------------------
Summary tables were generated from ROI-level MVPA decoding results. For each
classification target (checkmate: check vs non-check boards; strategy: 10-way
classification of strategic board characteristics), linear SVM classifiers with
L2 regularization were trained using leave-one-run-out cross-validation (LOOCV)
within each participant. Decoding accuracies were averaged across cross-validation
folds to obtain per-subject, per-ROI accuracy estimates.

Group-level statistics were computed by comparing expert and novice accuracies
within each ROI. For each target and ROI, we report:

1. **Group difference (Δacc)**: Computed as (Expert mean - Novice mean) with 95%
   confidence interval.

2. **Statistical significance**: Results from Welch's two-sample t-test
   (scipy.stats.ttest_ind with equal_var=False) comparing expert and novice
   accuracies. P-values are corrected for multiple comparisons using the
   Benjamini-Hochberg false discovery rate (FDR) procedure (α=0.05) across ROIs
   within each target.

The table presents both classification targets side-by-side for easy comparison
across regions.

Inputs
------
- mvpa_group_stats.pkl (in *_mvpa_group/): Nested dictionary containing decoding stats:
  - 'svm': Dict mapping classification targets to statistical blocks
    - 'welch_expert_vs_novice': Welch t-test results with FDR q-values
    - 'chance': Chance-level performance (float)

Outputs
-------
- tables/mvpa_decoding_summary.tex: Combined LaTeX table for both decoding targets
- tables/mvpa_decoding_summary.csv: CSV version

Dependencies
------------
- common.bids_utils: load_roi_metadata
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-mvpa/82_table_mvpa_decoding.py

Analysis 3b from manuscript: Main Tables, Methods Sec 3.7.3
"""

import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from common import setup_script, log_script_end
from common.bids_utils import load_roi_metadata
from common.tables import generate_styled_table
from common.formatters import format_p_cell, format_ci, shorten_roi_name
from common import CONFIG

# ============================================================================
# Configuration & Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='mvpa_group',
    output_subdirs=['tables'],
    log_name='tables_mvpa_decoding.log',
)
RESULTS_DIR = results_dir
tables_dir = dirs['tables']

# ============================================================================
# Load MVPA Group Statistics
# ============================================================================

logger.info("Loading MVPA group statistics from pickle file...")

with open(RESULTS_DIR / 'mvpa_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

# Load ROI metadata
roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])

# Extract SVM decoding targets
svm_data = index.get('svm', {})
available_targets = sorted(svm_data.keys())
logger.info(f"Available SVM targets in data: {available_targets}")

# Map target names to the three we need for the table
target_mapping = {
    'visual_similarity': 'visual_similarity',
    'strategy': 'strategy',
    'checkmate': 'checkmate'
}

# Use the targets as they appear in the data
expected_targets = []
for data_key in target_mapping.keys():
    if data_key in svm_data:
        expected_targets.append(data_key)
    else:
        logger.error(f"Could not find SVM target: {data_key}")
        raise ValueError(f"Expected SVM target '{data_key}' not found in data")

logger.info(f"Using SVM targets: {expected_targets}")

# ============================================================================
# Build Combined DataFrame
# ============================================================================

logger.info("Building combined table for all SVM targets...")

# Map target keys to display names
target_display_names = {
    'visual_similarity': 'Visual Similarity',
    'strategy': 'Strategy',
    'checkmate': 'Checkmate'
}

# Determine the order for the table (Visual, Strategy, Checkmate - following centrally defined order)
target_order = []
visual_tgt = None
strategy_tgt = None
checkmate_tgt = None

for tgt in expected_targets:
    if tgt == 'checkmate':
        checkmate_tgt = tgt
    elif tgt == 'strategy':
        strategy_tgt = tgt
    elif tgt == 'visual_similarity':
        visual_tgt = tgt

# Build ordered list: Visual, Strategy, Checkmate (centrally defined order)
target_order = []
if visual_tgt:
    target_order.append(visual_tgt)
if strategy_tgt:
    target_order.append(strategy_tgt)
if checkmate_tgt:
    target_order.append(checkmate_tgt)

expected_targets = target_order
logger.info(f"Reordered targets for table (Visual, Strategy, Checkmate): {expected_targets}")

# Extract data for each target
combined_data = []

# Get ROI labels from the first target (all should have the same ROIs)
first_target = expected_targets[0]
welch_df = svm_data[first_target]['welch_expert_vs_novice']
roi_labels = welch_df['ROI_Label'].tolist()

# Merge with ROI metadata to get pretty names
roi_df = welch_df.merge(
    roi_info[['roi_id', 'pretty_name']],
    left_on='ROI_Label',
    right_on='roi_id',
    how='left'
)

for roi_idx, roi_label in enumerate(roi_labels):
    row = {}

    # Get pretty ROI name
    pretty_name = roi_df.iloc[roi_idx]['pretty_name'].replace('\n', ' ')
    row['ROI'] = pretty_name

    # Add data for each target
    for tgt_idx, tgt in enumerate(expected_targets):
        welch_df_tgt = svm_data[tgt]['welch_expert_vs_novice']

        # Get stats for this ROI
        roi_stats = welch_df_tgt[welch_df_tgt['ROI_Label'] == roi_label].iloc[0]

        # Extract values
        delta_acc = roi_stats['mean_diff']
        ci_low = roi_stats['ci95_low']
        ci_high = roi_stats['ci95_high']
        p_raw = roi_stats['p_val']
        p_fdr = roi_stats['p_val_fdr']

        # Store in row with target-specific column names (use index for consistent naming)
        prefix = f'tgt{tgt_idx}'
        row[f'{prefix}_Δacc'] = delta_acc
        row[f'{prefix}_CI_low'] = ci_low
        row[f'{prefix}_CI_high'] = ci_high
        row[f'{prefix}_p'] = p_raw
        row[f'{prefix}_pFDR'] = p_fdr
        row[f'{prefix}_name'] = target_display_names[tgt]

    combined_data.append(row)

df_combined = pd.DataFrame(combined_data)

# ============================================================================
# Generate LaTeX Table (using centralized styling)
# ============================================================================

logger.info("Generating LaTeX table via centralized style system...")

# Build clean DataFrame with formatted columns for publication
rows = []
for _, r in df_combined.iterrows():
    row = {'ROI': shorten_roi_name(r['ROI'])}
    for tgt_idx in range(len(expected_targets)):
        prefix = f'tgt{tgt_idx}'
        # Use consistent per-target names
        group = ['Visual Similarity', 'Strategy', 'Check vs Non-Check'][tgt_idx] if tgt_idx < 3 else f'Target{tgt_idx+1}'
        row[f'Δacc_{group}'] = float(r[f'{prefix}_Δacc']) if pd.notna(r[f'{prefix}_Δacc']) else float('nan')
        if pd.notna(r[f'{prefix}_CI_low']) and pd.notna(r[f'{prefix}_CI_high']):
            row[f'95% CI_{group}'] = format_ci(float(r[f'{prefix}_CI_low']), float(r[f'{prefix}_CI_high']), precision=3, latex=False)
        else:
            row[f'95% CI_{group}'] = '[--, --]'
        row[f'p_{group}'] = format_p_cell(r[f'{prefix}_p'])
        row[f'pFDR_{group}'] = format_p_cell(r[f'{prefix}_pFDR'])
    rows.append(row)

df_out = pd.DataFrame(rows)

# Define multicolumn header groups
groups = ['Visual Similarity', 'Strategy', 'Check vs Non-Check']
multicolumn = {
    g: [f'Δacc_{g}', f'95% CI_{g}', f'p_{g}', f'pFDR_{g}'] for g in groups
}

# Generate and save LaTeX table (saves to both results and manuscript folders)
latex_path = tables_dir / 'mvpa_decoding_summary.tex'
generate_styled_table(
    df=df_out,
    output_path=latex_path,
    caption='ROI-level decoding. Expert–novice difference in accuracy (Δacc) for Visual Similarity, Strategy, and Check vs Non-Check; 95% CIs and both raw and FDR-corrected p-values (Welch t-tests per ROI).',
    label='tab:svm_roi_summary',
    multicolumn_headers=multicolumn,
    column_format='lSccc|Sccc|Sccc',
    logger=logger,
    manuscript_name='decoding_main_dims.tex',
)

# Save CSV version for reference
csv_path = tables_dir / 'mvpa_decoding_summary.csv'
df_out.to_csv(csv_path, index=False)
logger.info(f"CSV table saved to: {csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
