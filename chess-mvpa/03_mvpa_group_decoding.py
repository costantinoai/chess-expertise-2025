#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA Group Analysis — Decoding (ROI-level)

METHODS
=======

Rationale
---------
Multivariate pattern analysis (MVPA) assesses whether spatial patterns of
brain activity encode task-relevant information. To test whether chess
expertise modulates neural encoding of chess positions, we performed
group-level statistical tests on subject-level decoding accuracies across
regions of interest (ROIs).

Data
----
Subject-level decoding accuracies were computed using linear support vector
machines (SVMs) trained and tested on trial-wise beta estimates extracted
from 22 bilateral cortical regions (Glasser multimodal parcellation). Subject-
level analyses were performed separately for each classification target and
each ROI. Results are stored as TSV files in BIDS/derivatives/mvpa/svm/, with
one row per subject and one column per ROI.

Participants: N=40 (20 experts, 20 novices).
Stimuli: 40 chess board positions (20 check, 20 non-check).
ROIs: 22 bilateral cortical regions.

Chance-level accuracy was derived from the stimulus design and target
definition. For binary classifications (e.g., check vs non-check), chance =
0.5. For multi-class classifications, chance = 1 / n_classes.

Group-Level Statistical Testing
--------------------------------
Subject-level accuracies were grouped by expertise (experts vs novices) for
each ROI and each classification target. Three statistical tests were
conducted per ROI:

1. **Experts vs Chance**: One-sample one-tailed t-test testing whether expert
   mean accuracy significantly exceeded the theoretical chance level.
   Null hypothesis: μ_expert ≤ chance.
   Implementation: scipy.stats.ttest_1samp with alternative='greater'.

2. **Novices vs Chance**: One-sample one-tailed t-test testing whether novice
   mean accuracy significantly exceeded the theoretical chance level.
   Null hypothesis: μ_novice ≤ chance.
   Implementation: scipy.stats.ttest_1samp with alternative='greater'.

3. **Experts vs Novices**: Welch two-sample two-tailed t-test comparing expert
   and novice mean accuracies. Welch's t-test does not assume equal variances.
   Null hypothesis: μ_expert = μ_novice.
   Implementation: scipy.stats.ttest_ind with equal_var=False.

For one-sample tests, directional hypotheses yield one-tailed p-values. For
two-sample tests, non-directional hypotheses yield two-tailed p-values.

False Discovery Rate (FDR) Correction
--------------------------------------
To control for multiple comparisons across the 22 ROIs, we applied the
Benjamini-Hochberg FDR correction separately within each classification target
and each test type (expert vs chance, novice vs chance, expert vs novice).

Implementation: statsmodels.stats.multitest.multipletests with method='fdr_bh'
and alpha=0.05.

FDR correction was performed independently for each test type because each
addresses a distinct hypothesis family. ROIs were considered significant if
their FDR-corrected q-value was below 0.05.

Statistical Assumptions and Limitations
----------------------------------------
- **Normality**: t-tests assume normally distributed accuracies within each
  group and ROI. With n=20 per group, the central limit theorem provides
  robustness to moderate deviations.
- **Independence**: Subject-level accuracies are assumed independent. Scanner
  drift and shared task strategies may introduce correlated noise, but
  individual-level GLM estimation mitigates this concern.
- **Equal variances**: Welch's t-test relaxes the equal variance assumption.
- **Spatial dependence**: ROIs are anatomically adjacent and functionally
  connected, violating independence. FDR correction partially accounts for
  this by controlling the expected proportion of false positives.

Outputs
-------
All results are saved to results/<timestamp>_mvpa_group_decoding/:
- <target>_experts_vs_chance.csv: Expert vs chance statistics per ROI
- <target>_novices_vs_chance.csv: Novice vs chance statistics per ROI
- <target>_experts_vs_novices.csv: Group comparison statistics per ROI
- mvpa_group_stats.pkl: Complete results dictionary (for plotting scripts)
- 03_mvpa_group_decoding.py: Copy of this script
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

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
from common.io_utils import resolve_latest_dir

from modules.mvpa_io import (
    find_subject_tsvs,
    load_subject_tsv,
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

# Specify which MVPA run to analyze. If None, automatically use the most recent
# Glasser-22 bilateral analysis (timestamped directories). This allows rerunning
# the analysis on different preprocessing versions without code changes.
MVPA_DIR_NAME = None


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

config, out_dir, logger = setup_analysis(
    analysis_name="mvpa_group_decoding",
    results_base=script_dir / "results",
    script_file=__file__,
)

# Locate the subject-level MVPA results directory. Subject-level SVM decoding was
# performed in MATLAB; this script reads those results and performs group statistics.
mvpa_dir = resolve_latest_dir(CONFIG["BIDS_MVPA"], pattern="*_glasser_cortices_bilateral", specific_name=MVPA_DIR_NAME)
logger.info(f"Using MVPA source: {mvpa_dir}")

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
method = "svm"
method_dir = mvpa_dir / method
if not method_dir.exists():
    raise FileNotFoundError(f"Missing method directory: {method_dir}")

# Find all subject-level TSV files (one per subject, containing accuracies for all ROIs)
files = find_subject_tsvs(method_dir)
logger.info(f"[{method}] Found {len(files)} subject TSVs")

# Load subject TSVs and consolidate into a single DataFrame with columns:
# subject, expert (bool), target (classification task), and one column per ROI (accuracy).
# Each row = one subject × one target combination.
df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ["subject", "expert", "target"]]

# Determine chance-level accuracy for each classification target based on stimulus design.
# Binary tasks (e.g., check vs non-check) have chance=0.5. Multi-class tasks have chance=1/n_classes.
# This is derived from stimulus metadata (counting stimuli per category).
stim = load_stimulus_metadata(return_all=True)
targets = sorted(df['target'].dropna().unique())
chance_map = derive_target_chance_from_stimuli(targets, stimuli_df=stim)

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

logger.info("Saved group statistics artifacts (decoding)")
log_script_end(logger)
logger.info(f"All outputs saved to: {out_dir}")
