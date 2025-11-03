#!/usr/bin/env python3
"""
MVPA-Finer Group Analysis — Decoding (ROI-level, checkmate-only targets)

Performs group-level tests on subject-level ROI SVM decoding accuracies for
fine dimensions computed on checkmate boards only.
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

import numpy as np

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import (
    get_participants_with_expertise,
    load_roi_metadata,
    load_stimulus_metadata,
    derive_target_chance_from_stimuli,
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

config, out_dir, logger = setup_analysis(
    analysis_name="mvpa_finer_group_decoding",
    results_base=script_dir / "results",
    script_file=__file__,
)

# Locate the subject-level MVPA decoding directory. Subject-level SVM decoding was
# performed in MATLAB; this script reads those results and performs group statistics.
# Data is organized by subject with TSV files containing all targets including "_half" targets.
mvpa_dir = CONFIG["BIDS_MVPA_DECODING"]
if not mvpa_dir.exists():
    raise FileNotFoundError(f"Missing MVPA decoding directory: {mvpa_dir}")
logger.info(f"Using MVPA finer source: {mvpa_dir}")

# Load participant list with expertise labels (expert=True/False) for group assignment
participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

# Load ROI metadata (names, hemisphere, colors) for labeling and plotting
roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])
default_roi_names, _ = get_roi_names_and_colors(CONFIG["ROI_GLASSER_22"])

artifact_index = {}

# Process SVM decoding results (linear SVMs with L2 regularization, C=1.0)
# Find all subject-level TSV files (one per subject, containing accuracies for all ROIs and targets)
# These files contain both regular targets and "_half" targets (checkmate-only stimuli)
method = "svm"
files = find_subject_tsvs(mvpa_dir)
logger.info(f"[{method}] Found {len(files)} subject TSVs")

# Load subject TSVs and consolidate into a single DataFrame with columns:
# subject, expert (bool), target (classification task), and one column per ROI (accuracy).
# Each row = one subject × one target combination.
df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ["participant_id", "expert", "target"]]

# Determine chance-level accuracy for each classification target based on stimulus design.
# For targets ending in "_half", chance is computed from checkmate stimuli only.
# Binary tasks have chance=0.5. Multi-class tasks have chance=1/n_classes.
stim = load_stimulus_metadata(return_all=True)
targets = sorted(df['target'].dropna().unique())

# Build chance map: for *_half targets, filter to checkmate stimuli; otherwise use full set
chance_map = {}
for tgt in targets:
    if tgt.endswith('_half'):
        # Filter stimuli to checkmate boards only (20 stimuli)
        checkmate_stim = stim[stim['check'] == 'checkmate'].copy()
        # Remove '_half' suffix to get base target name for derive_target_chance_from_stimuli
        base_target = tgt[:-5]
        chance_map[tgt] = derive_target_chance_from_stimuli([base_target], stimuli_df=checkmate_stim).get(base_target, np.nan)
    else:
        # Use full stimulus set
        chance_map[tgt] = derive_target_chance_from_stimuli([tgt], stimuli_df=stim).get(tgt, np.nan)

# For each target and each ROI, perform three statistical tests:
# 1. Experts vs chance: one-sample one-tailed t-test (µ_expert > chance)
# 2. Novices vs chance: one-sample one-tailed t-test (µ_novice > chance)
# 3. Experts vs novices: Welch two-sample two-tailed t-test
# Apply FDR correction across ROIs within each test type and target.
method_results = {}

for tgt in targets:
    # Extract expert and novice data for this target
    expert_data, novice_data = split_data_by_target_and_group(df, tgt, roi_names)

    # Get target-specific chance level (varies by task)
    chance_level = float(chance_map.get(tgt, np.nan))

    # Perform between-group comparison (expert vs novice) with descriptives
    group_comparison = compute_per_roi_group_comparison(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        alpha=CONFIG['ALPHA_FDR'],
        confidence_level=0.95,
    )

    # Perform within-group vs-chance tests with descriptives
    expert_vs_chance, novice_vs_chance = compute_per_roi_vs_chance_tests(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        chance_level=chance_level,
        alpha=CONFIG['ALPHA_FDR'],
        alternative='greater',
        confidence_level=0.95,
    )

    # Package results for output functions
    method_results[tgt] = {
        'welch_expert_vs_novice': group_comparison['test_results'],
        'experts_vs_chance': expert_vs_chance['test_results'],
        'novices_vs_chance': novice_vs_chance['test_results'],
        'chance': chance_level,
        'experts_desc': group_comparison['expert_desc'],
        'novices_desc': group_comparison['novice_desc'],
    }

# Save results as human-readable CSVs (one per target per test type) for inspection
# and as pickle files for downstream plotting scripts.
for tgt, blocks in method_results.items():
    write_group_stats_outputs(out_dir, method, tgt, blocks)

artifact_index[method] = method_results

with open(out_dir / "mvpa_group_stats.pkl", "wb") as f:
    pickle.dump(artifact_index, f)

logger.info("Saved group statistics artifacts (MVPA finer decoding)")
log_script_end(logger)
logger.info(f"All outputs saved to: {out_dir}")
