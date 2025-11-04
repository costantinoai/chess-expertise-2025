#!/usr/bin/env python3
"""
Split-Half Reliability — LaTeX Table Generation
================================================

Generates a publication-ready LaTeX table summarizing the split-half
reliability of behavioral representational dissimilarity matrices (RDMs) for
expert and novice chess players. Uses the centralized table generator for
consistent formatting, validation, and LaTeX compile checks.

Tables Produced
---------------
- File: results/behavioral_split_half/tables/split_half_reliability.tex
- Columns: Group, Condition, Mean, 95% CI, p
- Formatting: booktabs, consistent math headers, c-only alignment

Inputs
------
- results/behavioral_split_half/reliability_metrics.pkl
  Keys:
    - 'experts_within': {mean_r_full, ci_r_full, p_boot_full}
    - 'novices_within': {mean_r_full, ci_r_full, p_boot_full}
    - 'between_groups': {mean_r_full, ci_r_full, p_boot_full}
    - 'experts_vs_novices_diff': {ci_delta_full, p_boot_delta_full}
    - 'between_groups_diff': {ci_delta_full, p_boot_delta_full} (optional)

Outputs
-------
- tables/split_half_reliability.tex (LaTeX)
- tables/split_half_reliability.csv (CSV)
"""

import os
import sys
from pathlib import Path
import pickle
import pandas as pd

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from common import setup_script, log_script_end
from common.tables import generate_styled_table
from common.formatters import format_ci, format_p_cell


# ============================================================================
# Configuration & Setup
# ============================================================================

results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='behavioral_split_half',
    output_subdirs=['tables'],
    log_name='tables_split_half.log',
)
RESULTS_DIR = results_dir
tables_dir = dirs['tables']


# ============================================================================
# Load Reliability Results
# ============================================================================

logger.info("Loading split-half reliability metrics from pickle file...")
with open(RESULTS_DIR / 'reliability_metrics.pkl', 'rb') as f:
    results = pickle.load(f)


# ============================================================================
# Build Table via Centralized Generator
# ============================================================================

logger.info("Generating split-half reliability LaTeX table...")

def _ci(ci):
    try:
        lo = float(ci[0]); hi = float(ci[1])
        import math
        if math.isnan(lo) or math.isnan(hi):
            return '{--}{--}'
        return format_ci(lo, hi, precision=3, latex=False, use_numrange=True)
    except Exception:
        return '{--}{--}'

def _mean(v):
    try:
        return float(v)
    except Exception:
        return float('nan')

def _p(v):
    try:
        return format_p_cell(v)
    except Exception:
        return '--'

exp_within = results.get('experts_within', {})
nov_within = results.get('novices_within', {})
between = results.get('between_groups', {})
diff = results.get('experts_vs_novices_diff', {})
between_diff = results.get('between_groups_diff', {})

rows = [
    {
        'Group': 'Experts',
        'Condition': 'Within',
        'Mean': _mean(exp_within.get('mean_r_full')),
        '95% CI': _ci(exp_within.get('ci_r_full', (float('nan'), float('nan')))),
        'p': _p(exp_within.get('p_boot_full')),
    },
    {
        'Group': 'Experts',
        'Condition': 'Between',
        'Mean': _mean(between.get('mean_r_full')),
        '95% CI': _ci(between.get('ci_r_full', (float('nan'), float('nan')))),
        'p': _p(between.get('p_boot_full')),
    },
    {
        'Group': 'Novices',
        'Condition': 'Within',
        'Mean': _mean(nov_within.get('mean_r_full')),
        '95% CI': _ci(nov_within.get('ci_r_full', (float('nan'), float('nan')))),
        'p': _p(nov_within.get('p_boot_full')),
    },
    {
        'Group': 'Novices',
        'Condition': 'Between',
        'Mean': _mean(between.get('mean_r_full')),
        '95% CI': _ci(between.get('ci_r_full', (float('nan'), float('nan')))),
        'p': _p(between.get('p_boot_full')),
    },
    {
        'Group': 'Between-group comparison',
        'Condition': 'Within: Experts vs. Novices',
        'Mean': None,
        '95% CI': _ci(diff.get('ci_delta_full', (float('nan'), float('nan')))),
        'p': _p(diff.get('p_boot_delta_full')),
    },
    {
        'Group': 'Between-group comparison',
        'Condition': 'Between: Experts vs. Novices',
        'Mean': None,
        '95% CI': _ci(between_diff.get('ci_delta_full', (float('nan'), float('nan')))),
        'p': _p(between_diff.get('p_boot_delta_full')),
    },
]

df = pd.DataFrame(rows)
df['Mean'] = df['Mean'].map(
    lambda x: r'\textemdash' if (x is None or (isinstance(x, float) and (x != x))) else x
)

tex_path = tables_dir / 'split_half_reliability.tex'
generate_styled_table(
    df=df,
    output_path=tex_path,
    caption='Summary statistics for Experts and Novices (within- and between-condition), with 95% confidence intervals and p-values.',
    label='supptab:bh_splithalf',
    column_format='llccc',
    logger=logger,
    manuscript_name='bh_rdm_splithalf.tex',
)

# CSV copy for reference
df.to_csv(tables_dir / 'split_half_reliability.csv', index=False)

# ============================================================================
# Finish
# ============================================================================

logger.info(f"Saved split-half reliability LaTeX table: {tex_path}")
log_script_end(logger)

