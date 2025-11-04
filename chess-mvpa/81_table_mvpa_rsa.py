#!/usr/bin/env python3
"""
MVPA RSA — LaTeX Table Generation (Experts vs Novices)
=======================================================

This script generates a publication-ready LaTeX table summarizing multi-voxel
pattern analysis (MVPA) representational similarity analysis (RSA) results,
comparing neural similarity structure between experts and novices across brain
regions for three RSA targets: Check vs Non-Check, Strategy, and Visual Similarity.

METHODS (Academic Manuscript Section)
--------------------------------------
Summary tables were generated from ROI-level MVPA RSA correlation results.
For each RSA target (theoretical model), neural RDMs were correlated with model
RDMs within each of 22 bilateral cortical regions. Group-level statistics were
computed by comparing expert and novice Fisher z-transformed correlation values.

For each target and ROI, we report:

1. **Group difference (Δr)**: Computed as (Expert mean - Novice mean) with 95%
   confidence interval, back-transformed from Fisher z-space.

2. **Statistical significance**: Results from Welch's two-sample t-test
   (scipy.stats.ttest_ind with equal_var=False) on Fisher z-transformed
   correlations. P-values are corrected for multiple comparisons using the
   Benjamini-Hochberg false discovery rate (FDR) procedure (α=0.05) across ROIs
   within each target.

The table presents all three RSA targets (Check vs Non-Check, Strategy, Visual
Similarity) side-by-side for easy comparison across regions.

Inputs
------
- mvpa_group_stats.pkl (in *_mvpa_group/): Nested dictionary containing (from RSA group analysis):
  - 'rsa_corr': Dict mapping RSA targets to statistical blocks
    - 'welch_expert_vs_novice': Welch t-test results with FDR q-values

Outputs
-------
- tables/mvpa_rsa_summary.tex: Combined LaTeX table for all three RSA targets
- tables/mvpa_rsa_summary.csv: CSV version

Dependencies
------------
- common.bids_utils: load_roi_metadata
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder
- common.formatters: format_ci, format_p_value

Usage
-----
python chess-mvpa/81_table_mvpa_rsa.py

Analysis 3a from manuscript: Main Tables, Methods Sec 3.7.2
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
from common.tables import generate_styled_table, build_c_only_colspec
from common.formatters import format_p_cell, format_ci, shorten_roi_name
from common import CONFIG

# ============================================================================
# Configuration & Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='mvpa_group',
    output_subdirs=['tables'],
    log_name='tables_mvpa_rsa.log',
)
RESULTS_DIR = results_dir
tables_dir = dirs['tables']

# ============================================================================
# Load MVPA Group Statistics
# ============================================================================

logger.info("Loading MVPA group statistics from pickle file...")
with open(RESULTS_DIR / 'mvpa_group_stats.pkl', 'rb') as f:
    index = pickle.load(f)

# Extract RSA targets from data
rsa_data = index.get('rsa_corr', {})
available_targets = sorted(rsa_data.keys())
logger.info(f"Available RSA targets in data: {available_targets}")

# Map target names to the three we need for the table
# Based on CONFIG['MODEL_LABELS'], targets are: check, visual, strategy
target_mapping = {
    'checkmate': 'check',
    'visual_similarity': 'visual',
    'strategy': 'strategy'
}

# Use the targets as they appear in the data
expected_targets = []
for data_key, display_key in target_mapping.items():
    if data_key in rsa_data:
        expected_targets.append(data_key)
    elif display_key in rsa_data:
        expected_targets.append(display_key)
    else:
        logger.error(f"Could not find RSA target for {display_key} (tried {data_key})")
        raise ValueError(f"Expected RSA target not found in data")

logger.info(f"Using RSA targets: {expected_targets}")

# ============================================================================
# Build Combined DataFrame
# ============================================================================

logger.info("Building combined table for all RSA targets...")

# Map target keys to display names (works with both naming conventions)
target_display_names = {
    'checkmate': 'Check vs Non-Check',
    'check': 'Check vs Non-Check',
    'strategy': 'Strategy',
    'visual_similarity': 'Visual Similarity',
    'visual': 'Visual Similarity'
}

# Determine the order for the table (Visual, Strategy, Check)
# Following the centrally defined order in CONFIG
target_order = []
check_tgt = None
strategy_tgt = None
visual_tgt = None

for tgt in expected_targets:
    if tgt in ['checkmate', 'check']:
        check_tgt = tgt
    elif tgt == 'strategy':
        strategy_tgt = tgt
    elif tgt in ['visual_similarity', 'visual']:
        visual_tgt = tgt

# Build ordered list: Visual, Strategy, Check (centrally defined order)
target_order = []
if visual_tgt:
    target_order.append(visual_tgt)
if strategy_tgt:
    target_order.append(strategy_tgt)
if check_tgt:
    target_order.append(check_tgt)

expected_targets = target_order
logger.info(f"Reordered targets for table (Visual, Strategy, Check): {expected_targets}")

# Extract data for each target
# We'll build a DataFrame with columns: ROI, Check_Δr, Check_CI, Check_p, ...
combined_data = []

# Get ROI labels from the first target (all should have the same ROIs)
first_target = expected_targets[0]
welch_df = rsa_data[first_target]['welch_expert_vs_novice']
roi_labels = welch_df['ROI_Label'].tolist()

for roi_idx, roi_label in enumerate(roi_labels):
    row = {}
    # Use ROI name from stats (no external metadata dependency)
    pretty_name = welch_df[welch_df['ROI_Label'] == roi_label]['ROI_Name'].iloc[0]
    row['ROI'] = shorten_roi_name(str(pretty_name).replace('\n', ' '))

    # Add data for each target
    for tgt_idx, tgt in enumerate(expected_targets):
        welch_df_tgt = rsa_data[tgt]['welch_expert_vs_novice']

        # Get stats for this ROI
        roi_stats = welch_df_tgt[welch_df_tgt['ROI_Label'] == roi_label].iloc[0]

        # Extract values
        delta_r = roi_stats['mean_diff']
        ci_low = roi_stats['ci95_low']
        ci_high = roi_stats['ci95_high']
        p_raw = roi_stats['p_val']
        p_fdr = roi_stats['p_val_fdr']

        # Store in row with target-specific column names (use index for consistent naming)
        prefix = f'tgt{tgt_idx}'
        row[f'{prefix}_Δr'] = delta_r
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
    for tgt_idx, tgt in enumerate(expected_targets):
        prefix = f'tgt{tgt_idx}'
        group = target_display_names[tgt]
        row[f'Δr_{group}'] = float(r[f'{prefix}_Δr']) if pd.notna(r[f'{prefix}_Δr']) else float('nan')
        if pd.notna(r[f'{prefix}_CI_low']) and pd.notna(r[f'{prefix}_CI_high']):
            row[f'95% CI_{group}'] = format_ci(float(r[f'{prefix}_CI_low']), float(r[f'{prefix}_CI_high']), precision=3, latex=False, use_numrange=True)
        else:
            row[f'95% CI_{group}'] = '{--}{--}'
        row[f'p_{group}'] = format_p_cell(r[f'{prefix}_p'])
        row[f'pFDR_{group}'] = format_p_cell(r[f'{prefix}_pFDR'])
    rows.append(row)

df_out = pd.DataFrame(rows)

# Define multicolumn header groups
multicolumn = {
    target_display_names[t]: [
        f'Δr_{target_display_names[t]}',
        f'95% CI_{target_display_names[t]}',
        f'p_{target_display_names[t]}',
        f'pFDR_{target_display_names[t]}'
    ] for t in expected_targets
}

# Generate and save LaTeX table (saves to both results and manuscript folders)
latex_path = tables_dir / 'mvpa_rsa_summary.tex'
generate_styled_table(
    df=df_out,
    output_path=latex_path,
    caption='ROI-level RSA. Expert–novice difference in correlation ($\\delta r$) for Visual Similarity, Strategy, and Check vs Non-Check; 95% CIs and both raw and FDR-corrected $p$-values (Welch $t$-tests per ROI).',
    label='tab:rsa_roi_summary',
    multicolumn_headers=multicolumn,
    column_format=build_c_only_colspec(df_out, multicolumn),
    logger=logger,
    manuscript_name='rsa_main_dims.tex',
)

# Save CSV version for reference
csv_path = tables_dir / 'mvpa_rsa_summary.csv'
df_out.to_csv(csv_path, index=False)
logger.info(f"CSV table saved to: {csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
