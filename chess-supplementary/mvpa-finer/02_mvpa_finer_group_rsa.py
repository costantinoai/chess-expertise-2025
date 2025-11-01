#!/usr/bin/env python3
"""
MVPA-Finer Group Analysis — RSA Correlations (ROI-level, checkmate-only targets)

Performs group-level tests on subject-level ROI RSA correlations for fine
dimensions computed on checkmate boards only (still Glasser-22 atlas). Uses the
latest MATLAB outputs under CONFIG['BIDS_MVPA'] matching CONFIG['MVPA_PATTERN_CM_ONLY'] and writes
timestamped results in this package.
"""

import os
import sys
from pathlib import Path
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add repo root for 'common' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Add chess-mvpa to import path to reuse its modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chess-mvpa')))
script_dir = Path(__file__).parent

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import (
    get_participants_with_expertise,
    load_roi_metadata,
)
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs

from modules.mvpa_io import (
    find_subject_tsvs,
    build_group_dataframe,
)
from modules.mvpa_group import (
    compute_per_roi_group_comparison,
    compute_per_roi_vs_chance_tests,
    split_data_by_target_and_group,
)


config, out_dir, logger = setup_analysis(
    analysis_name="mvpa_finer_group_rsa",
    results_base=script_dir / 'results',
    script_file=__file__,
)

# Use MVPA RSA directory which contains Glasser-22 ROIs and all fine-grained targets
mvpa_dir = CONFIG['BIDS_MVPA_RSA']
if not mvpa_dir.exists():
    raise FileNotFoundError(f"Missing MVPA RSA directory: {mvpa_dir}")
logger.info(f"Using MVPA finer source: {mvpa_dir}")

participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG['BIDS_PARTICIPANTS'], bids_root=CONFIG['BIDS_ROOT']
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
default_roi_names, _ = get_roi_names_and_colors(CONFIG['ROI_GLASSER_22'])

# Find all subject-level TSV files containing RSA correlation coefficients
# These files contain both regular targets and "_half" targets (checkmate-only stimuli)
files = find_subject_tsvs(mvpa_dir)
method = 'rsa_corr'
logger.info(f"[{method}] Found {len(files)} subject TSVs")

df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ['participant_id', 'expert', 'target']]

chance_level = float(CONFIG.get('CHANCE_LEVEL_RSA', 0.0))
targets = sorted(df['target'].dropna().unique())
method_results = {}

for tgt in targets:
    expert_data, novice_data = split_data_by_target_and_group(df, tgt, roi_names)
    group_comparison = compute_per_roi_group_comparison(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        alpha=CONFIG['ALPHA_FDR'],
        confidence_level=0.95,
    )
    expert_vs_chance, novice_vs_chance = compute_per_roi_vs_chance_tests(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        chance_level=chance_level,
        alpha=CONFIG['ALPHA_FDR'],
        alternative='greater',
        confidence_level=0.95,
    )
    method_results[tgt] = {
        'welch_expert_vs_novice': group_comparison['test_results'],
        'experts_vs_chance': expert_vs_chance['test_results'],
        'novices_vs_chance': novice_vs_chance['test_results'],
        'chance': chance_level,
        'experts_desc': group_comparison['expert_desc'],
        'novices_desc': group_comparison['novice_desc'],
    }

for tgt, blocks in method_results.items():
    write_group_stats_outputs(out_dir, method, tgt, blocks)

with open(out_dir / 'mvpa_group_stats.pkl', 'wb') as f:
    pickle.dump({'rsa_corr': method_results}, f)

logger.info("Saved group statistics artifacts (MVPA finer RSA)")
log_script_end(logger)
